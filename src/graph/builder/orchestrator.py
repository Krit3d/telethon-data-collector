import asyncio
import logging

from sqlalchemy import update

from src.config.config import Settings
from src.db.database import Database
from src.db.models import Content
from src.graph.client import Neo4jClient
from src.graph.builder.reader import PostBatchContext, Reader
from src.graph.builder.splitter import TextSplitter
from src.graph.builder.extractor import GraphExtractor
from src.graph.builder.aligner import Aligner
from src.graph.builder.vectorizer import EntityVectorizer
from src.graph.builder.writer import GraphWriter

logger = logging.getLogger(__name__)


class KagBuilderOrchestrator:

    def __init__(self, settings: Settings, db: Database, neo4j_client: Neo4jClient) -> None:
        self._settings = settings
        self._db = db
        self._neo4j_client = neo4j_client

        self._reader = Reader(self._db)
        self._splitter = TextSplitter()
        self._extractor = GraphExtractor(self._settings)
        self._aligner = Aligner(self._settings, self._neo4j_client)
        self._vectorizer = EntityVectorizer(self._settings)
        self._writer = GraphWriter(self._settings, self._neo4j_client, self._db)

        self._semaphore = asyncio.Semaphore(settings.graph_concurrency)
        self._running = True

    async def process_post(self, context: PostBatchContext) -> None:
        async with self._semaphore:
            try:
                chunks = self._splitter.prepare_and_split(context.content, context.transcription)

                for chunk in chunks:
                    extraction_result = await self._extractor.extract(context, chunk.text)
                    aligned_result = await self._aligner.align(extraction_result, context)

                    await asyncio.gather(
                        self._vectorizer.vectorize_and_upsert_entities(aligned_result),
                        self._writer.write_extraction_result(aligned_result, context),
                    )
            except Exception:
                logger.exception("Post %d processing failed, marking as failed", context.content_id)
                async with self._db.async_session() as session:
                    async with session.begin():
                        await session.execute(
                            update(Content)
                            .where(Content.id == context.content_id)
                            .values(graph_status=3)
                        )

    async def run_pipeline(self, poll_interval: float = 3.0) -> None:
        iteration = 0
        while self._running:
            try:
                if iteration % 10 == 0:
                    recovered = await self._reader.recover_stale_claims(timeout_minutes=30)
                    if recovered:
                        logger.info("Recovered %d stale claims", recovered)

                batch = await self._reader.fetch_pending_batch(
                    batch_size=self._settings.graph_batch_size,
                    worker_id=self._settings.worker_id,
                    total_workers=self._settings.total_workers,
                    priority_mode=self._settings.extraction_priority_mode,
                )

                if not batch:
                    await asyncio.sleep(poll_interval)
                    iteration += 1
                    continue

                tasks = [self.process_post(ctx) for ctx in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for ctx, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.error("Post %d failed with error: %s", ctx.content_id, result)

                iteration += 1
            except asyncio.CancelledError:
                logger.info("Pipeline cancelled")
                break
            except Exception:
                logger.exception("Pipeline iteration %d failed unexpectedly", iteration)
                await asyncio.sleep(poll_interval)
                iteration += 1

    def stop(self) -> None:
        self._running = False
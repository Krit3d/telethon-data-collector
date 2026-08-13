import asyncio
import logging
import time

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

        self._queue: asyncio.Queue[PostBatchContext] = asyncio.Queue(maxsize=200)
        self._producer_task: asyncio.Task | None = None
        self._consumer_tasks: list[asyncio.Task] = []
        self._running = True

    async def process_post(self, context: PostBatchContext) -> None:
        t_post = time.perf_counter()
        try:
            logger.info("Processing post %d (author: %s, platform: %s)...", context.content_id, context.author_title, context.platform)

            t0 = time.perf_counter()
            chunks = self._splitter.prepare_and_split(context.content, context.transcription)
            splitter_elapsed = time.perf_counter() - t0

            total_extractor = 0.0
            total_aligner = 0.0
            total_writer = 0.0

            for chunk in chunks:
                t1 = time.perf_counter()
                extraction_result = await self._extractor.extract(context, chunk.text)
                total_extractor += time.perf_counter() - t1

                t2 = time.perf_counter()
                aligned_result = await self._aligner.align(extraction_result, context)
                total_aligner += time.perf_counter() - t2

                t3 = time.perf_counter()
                await asyncio.gather(
                    self._vectorizer.vectorize_and_upsert_entities(aligned_result),
                    self._writer.write_extraction_result(aligned_result, context),
                )
                total_writer += time.perf_counter() - t3

            total_elapsed = time.perf_counter() - t_post
            logger.debug(
                "[Post %d] Processed in %.2fs (Splitter: %.2fs, Extractor: %.2fs, Aligner: %.2fs, Writer: %.2fs)",
                context.content_id, total_elapsed,
                splitter_elapsed, total_extractor, total_aligner, total_writer,
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

    async def _producer_loop(self, poll_interval: float = 3.0) -> None:
        iteration = 0
        while self._running:
            try:
                if iteration % 10 == 0:
                    recovered = await self._reader.recover_stale_claims(timeout_minutes=30)
                    if recovered:
                        logger.info("Recovered %d stale claims", recovered)

                qsize = self._queue.qsize()
                if qsize < self._queue.maxsize:
                    t0 = time.perf_counter()
                    batch = await self._reader.fetch_pending_batch(
                        batch_size=self._settings.graph_batch_size,
                        worker_id=self._settings.worker_id,
                        total_workers=self._settings.total_workers,
                        priority_mode=self._settings.extraction_priority_mode,
                    )
                    fetch_elapsed = time.perf_counter() - t0

                    if not batch:
                        await asyncio.sleep(poll_interval)
                        iteration += 1
                        continue

                    for ctx in batch:
                        await self._queue.put(ctx)

                    logger.debug(
                        "Produced %d posts into queue (qsize=%d, fetch=%.2fs)",
                        len(batch), self._queue.qsize(), fetch_elapsed,
                    )
                else:
                    logger.debug("Queue full (qsize=%d), backing off", qsize)
                    await asyncio.sleep(0.1)

                iteration += 1
            except asyncio.CancelledError:
                logger.info("Producer loop cancelled")
                break
            except Exception:
                logger.exception("Producer loop iteration %d failed", iteration)
                await asyncio.sleep(poll_interval)
                iteration += 1

    async def _consumer_worker(self, worker_id: int) -> None:
        while self._running:
            try:
                context = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self.process_post(context)
            except Exception:
                logger.exception("Consumer %d: post %d failed unexpectedly", worker_id, context.content_id)
            finally:
                self._queue.task_done()

    async def run_pipeline(self, poll_interval: float = 3.0) -> None:
        self._producer_task = asyncio.create_task(self._producer_loop(poll_interval))
        self._consumer_tasks = [
            asyncio.create_task(self._consumer_worker(i))
            for i in range(self._settings.graph_concurrency)
        ]

        try:
            await asyncio.gather(self._producer_task, *self._consumer_tasks)
        except asyncio.CancelledError:
            logger.info("Pipeline cancelled")
        finally:
            self._producer_task.cancel()
            for t in self._consumer_tasks:
                t.cancel()
            await asyncio.gather(self._producer_task, *self._consumer_tasks, return_exceptions=True)

    def stop(self) -> None:
        self._running = False
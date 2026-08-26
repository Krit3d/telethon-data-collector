import asyncio
import logging
import time
from datetime import datetime, timezone

from sqlalchemy import update

from src.config.config import Settings
from src.db.database import Database
from src.db.models import Content
from src.graph.client import Neo4jClient
from src.graph.builder.reader import PostBatchContext, Reader
from src.graph.builder.splitter import TextSplitter
from src.graph.builder.extractor import GraphExtractor, LLMInfrastructureError
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
            try:
                chunks = self._splitter.prepare_and_split(context.content, context.transcription)
            except Exception as e:
                splitter_elapsed = time.perf_counter() - t0
                logger.error("[Post %d] Splitter FAILED after %.2fs | error: %s", context.content_id, splitter_elapsed, e)
                raise
            splitter_elapsed = time.perf_counter() - t0
            logger.debug("[Post %d] Splitter done in %.2fs | chunks: %d", context.content_id, splitter_elapsed, len(chunks))

            total_extractor = 0.0
            total_aligner = 0.0
            total_writer = 0.0

            for chunk_idx, chunk in enumerate(chunks):
                t1 = time.perf_counter()
                try:
                    extraction_result = await self._extractor.extract(
                        context,
                        caption_text=chunk.caption_text,
                        transcription_text=chunk.transcription_text,
                    )
                except Exception as e:
                    extractor_elapsed = time.perf_counter() - t1
                    logger.error("[Post %d] Extractor FAILED after %.2fs (chunk %d) | error: %s", context.content_id, extractor_elapsed, chunk_idx, e)
                    raise
                extractor_elapsed = time.perf_counter() - t1
                total_extractor += extractor_elapsed
                logger.debug("[Post %d] Extractor done in %.2fs (chunk %d)", context.content_id, extractor_elapsed, chunk_idx)

                t2 = time.perf_counter()
                try:
                    aligned_result = await self._aligner.align(extraction_result, context)
                except Exception as e:
                    aligner_elapsed = time.perf_counter() - t2
                    logger.error("[Post %d] Aligner FAILED after %.2fs (chunk %d) | error: %s", context.content_id, aligner_elapsed, chunk_idx, e)
                    raise
                aligner_elapsed = time.perf_counter() - t2
                total_aligner += aligner_elapsed
                logger.debug("[Post %d] Aligner done in %.2fs (chunk %d)", context.content_id, aligner_elapsed, chunk_idx)

                t3 = time.perf_counter()
                try:
                    await asyncio.gather(
                        self._vectorizer.vectorize_and_upsert_entities(aligned_result),
                        self._writer.write_extraction_chunk(aligned_result, context, is_first_chunk=chunk_idx == 0),
                    )
                except Exception as e:
                    writer_elapsed = time.perf_counter() - t3
                    logger.error("[Post %d] Writer/Vectorizer FAILED after %.2fs (chunk %d) | error: %s", context.content_id, writer_elapsed, chunk_idx, e)
                    raise
                writer_elapsed = time.perf_counter() - t3
                total_writer += writer_elapsed
                logger.debug("[Post %d] Writer/Vectorizer done in %.2fs (chunk %d)", context.content_id, writer_elapsed, chunk_idx)

            total_elapsed = time.perf_counter() - t_post
            logger.debug(
                "[Post %d] Processed in %.2fs (Splitter: %.2fs, Extractor: %.2fs, Aligner: %.2fs, Writer: %.2fs)",
                context.content_id, total_elapsed,
                splitter_elapsed, total_extractor, total_aligner, total_writer,
            )
        except Exception as e:
            is_infra_error = isinstance(e, LLMInfrastructureError)
            if is_infra_error:
                logger.warning(
                    "Post %d transient LLM infrastructure failure, resetting status for retry: %s",
                    context.content_id, e,
                )
                target_status = 0
            else:
                logger.exception("Post %d processing failed, marking as failed: %s", context.content_id, e)
                target_status = 3
            async with self._db.async_session() as session:
                async with session.begin():
                    await session.execute(
                        update(Content)
                        .where(Content.id == context.content_id)
                        .values(graph_status=target_status, updated_at=datetime.now(timezone.utc))
                    )
            if is_infra_error:
                await asyncio.sleep(2.0)

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
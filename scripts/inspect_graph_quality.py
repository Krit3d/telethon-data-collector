"""
Script for checking graph entities quality.

Command:
docker compose -f docker-compose.api.yml run --rm -v ./scripts:/app/scripts api python -m scripts.inspect_graph_quality
"""

import random as _random
import warnings
import logging
from pathlib import Path
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from qdrant_client import AsyncQdrantClient
from src.config.config import load_settings
from src.db.models import Account, Content

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)


async def main() -> None:
    settings = load_settings()
    engine = create_async_engine(settings.db_url, echo=False, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        total_verified = (await session.scalar(
            select(func.count(Content.id)).join(Content.account).where(Account.status == 'verified')
        )) or 0

        extracted_verified = (await session.scalar(
            select(func.count(Content.id))
            .join(Content.account)
            .where(Account.status == 'verified', Content.is_graph_extracted == True)
        )) or 0

        pct = (extracted_verified / total_verified * 100) if total_verified > 0 else 0.0

        print()
        print('Пункт 1: Статистика SQLAlchemy')
        print(f'  1. Верифицированные посты (общее число): {total_verified}')
        print(f'  2. Верифицированные посты с извлеченным графом: {extracted_verified}')
        print(f'  3. Покрытие экстракции графа: {pct:.2f}%')

        graph_name = settings.graph_name

        total_nodes_result = await session.execute(
            text(f'SELECT count(*) FROM {graph_name}._ag_label_vertex')
        )
        total_nodes = total_nodes_result.scalar() or 0

        total_edges_result = await session.execute(
            text(f'SELECT count(*) FROM {graph_name}._ag_label_edge')
        )
        total_edges = total_edges_result.scalar() or 0

        isolated_result = await session.execute(
            text(
                f'SELECT count(*) FROM {graph_name}._ag_label_vertex v '
                f'WHERE NOT EXISTS (SELECT 1 FROM {graph_name}._ag_label_edge e WHERE e.start_id = v.id) '
                f'AND NOT EXISTS (SELECT 1 FROM {graph_name}._ag_label_edge e WHERE e.end_id = v.id)'
            )
        )
        isolated = isolated_result.scalar() or 0

        isolated_ratio = (isolated / total_nodes * 100) if total_nodes > 0 else 0.0

        print()
        print('Пункт 2: Статистика Графовой БД')
        print(f'  1. Всего узлов (вершин): {total_nodes}')
        print(f'  2. Всего связей (ребер): {total_edges}')
        print(f'  3. Изолированные узлы: {isolated}')
        print(f'  4. Доля изолированных узлов: {isolated_ratio:.2f}%')

        qdrant_url = settings.qdrant_url
        qdrant_api_key = settings.qdrant_api_key
        collection = 'social_entities'

        print()
        print('Пункт 3: Статистика Qdrant')
        print(f'  1. Коллекция: {collection}')

        if qdrant_url:
            client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            try:
                collection_info = await client.get_collection(collection)
                points_count = collection_info.points_count
                print(f'  2. Всего точек (векторов): {points_count}')

                scroll_result = await client.scroll(
                    collection_name=collection,
                    limit=5,
                    with_payload=True,
                    with_vectors=True,
                )
                points, _ = scroll_result

                if points:
                    for point in points:
                        payload_keys = list(point.payload.keys()) if point.payload else []
                        vector = point.vector
                        vector_present = False
                        vector_dims = 0
                        if vector is not None:
                            if isinstance(vector, dict):
                                vector_present = True
                                for v in vector.values():
                                    if isinstance(v, (list, dict)) and len(v) > 0:
                                        vector_dims = len(v)
                                        break
                            elif isinstance(vector, (list, dict)):
                                vector_present = len(vector) > 0
                                if vector_present:
                                    vector_dims = len(vector)
                        print(f'  -> Точка {point.id}: Ключи payload: {payload_keys}, Вектор присутствует: {vector_present}, Размерность вектора: {vector_dims}')
                else:
                    print('  -> В коллекции нет точек')
            except Exception as exc:
                print(f'  2. Всего точек: Ошибка запроса к Qdrant: {exc}')
            finally:
                await client.close()
        else:
            print(f'  2. Всего точек: URL Qdrant не настроен')

        avg_entities_per_post = (total_nodes / extracted_verified) if extracted_verified > 0 else 0.0
        avg_relations_per_post = (total_edges / extracted_verified) if extracted_verified > 0 else 0.0

        zero_entities_result = await session.execute(
            text(
                f'SELECT count(*) FROM {graph_name}."Event" v '
                f'WHERE ((v.properties::varchar)::jsonb ->> \'id\') '
                f'LIKE \'event_publication_%\' '
                f'AND NOT EXISTS (SELECT 1 FROM {graph_name}._ag_label_edge e WHERE e.start_id = v.id)'
            )
        )
        zero_entities_count = zero_entities_result.scalar() or 0
        zero_entities_pct = (zero_entities_count / extracted_verified * 100) if extracted_verified > 0 else 0.0

        print()
        print('Пункт 4: Другие показатели (проверка здоровья графа)')
        print(f'  1. Среднее количество сущностей на пост: {avg_entities_per_post:.2f}')
        print(f'  2. Среднее количество связей на пост: {avg_relations_per_post:.2f}')
        print(f'  3. Доля постов с 0 извлеченных сущностей/тем: {zero_entities_pct:.2f}%')

        label_rows = (await session.execute(
            text(
                f'SELECT l.name AS label, count(*) AS cnt '
                f'FROM {graph_name}._ag_label_vertex v '
                f'JOIN ag_catalog.ag_label l ON l.id = (v.id::text::bigint >> 48) '
                f'JOIN ag_catalog.ag_graph g ON g.graphid = l.graph '
                f'WHERE g.name = \'{graph_name}\' '
                f'GROUP BY l.name ORDER BY cnt DESC LIMIT 20'
            )
        )).all()

        name_rows = (await session.execute(
            text(
                f'SELECT (v.properties::varchar)::jsonb ->> \'name\' AS name, '
                f'count(*) AS cnt '
                f'FROM {graph_name}._ag_label_vertex v '
                f'GROUP BY name ORDER BY cnt DESC LIMIT 20'
            )
        )).all()

        rel_rows = (await session.execute(
            text(
                f'SELECT l.name AS relation_type, count(*) AS cnt '
                f'FROM {graph_name}._ag_label_edge e '
                f'JOIN ag_catalog.ag_label l ON l.id = (e.id::text::bigint >> 48) '
                f'JOIN ag_catalog.ag_graph g ON g.graphid = l.graph '
                f'WHERE g.name = \'{graph_name}\' '
                f'GROUP BY l.name ORDER BY cnt DESC LIMIT 20'
            )
        )).all()

        triplet_rows = (await session.execute(
            text(
                f'SELECT la.name AS label_a, '
                f'(a.properties::varchar)::jsonb ->> \'name\' AS name_a, '
                f'lr.name AS label_r, '
                f'lb.name AS label_b, '
                f'(b.properties::varchar)::jsonb ->> \'name\' AS name_b '
                f'FROM {graph_name}._ag_label_edge r '
                f'JOIN {graph_name}._ag_label_vertex a ON r.start_id = a.id '
                f'JOIN {graph_name}._ag_label_vertex b ON r.end_id = b.id '
                f'JOIN ag_catalog.ag_label la ON la.id = (a.id::text::bigint >> 48) '
                f'JOIN ag_catalog.ag_label lr ON lr.id = (r.id::text::bigint >> 48) '
                f'JOIN ag_catalog.ag_label lb ON lb.id = (b.id::text::bigint >> 48) '
                f'JOIN ag_catalog.ag_graph g ON g.graphid = lr.graph '
                f'WHERE g.name = \'{graph_name}\' LIMIT 200'
            )
        )).all()

        if len(triplet_rows) >= 10:
            triplet_sample = _random.sample(triplet_rows, 10)
        else:
            triplet_sample = triplet_rows[:]

        try:
            recent_rows = (await session.execute(
                text(
                    f'SELECT la.name AS label_a, '
                    f'(a.properties::varchar)::jsonb ->> \'name\' AS name_a, '
                    f'lr.name AS label_r, '
                    f'lb.name AS label_b, '
                    f'(b.properties::varchar)::jsonb ->> \'name\' AS name_b '
                    f'FROM {graph_name}._ag_label_edge r '
                    f'JOIN {graph_name}._ag_label_vertex a ON r.start_id = a.id '
                    f'JOIN {graph_name}._ag_label_vertex b ON r.end_id = b.id '
                    f'JOIN ag_catalog.ag_label la ON la.id = (a.id::text::bigint >> 48) '
                    f'JOIN ag_catalog.ag_label lr ON lr.id = (r.id::text::bigint >> 48) '
                    f'JOIN ag_catalog.ag_label lb ON lb.id = (b.id::text::bigint >> 48) '
                    f'JOIN ag_catalog.ag_graph g ON g.graphid = lr.graph '
                    f'WHERE g.name = \'{graph_name}\' ORDER BY r.id DESC LIMIT 5'
                )
            )).all()
        except Exception:
            recent_rows = (await session.execute(
                text(
                    f'SELECT la.name AS label_a, '
                    f'(a.properties::varchar)::jsonb ->> \'name\' AS name_a, '
                    f'lr.name AS label_r, '
                    f'lb.name AS label_b, '
                    f'(b.properties::varchar)::jsonb ->> \'name\' AS name_b '
                    f'FROM {graph_name}._ag_label_edge r '
                    f'JOIN {graph_name}._ag_label_vertex a ON r.start_id = a.id '
                    f'JOIN {graph_name}._ag_label_vertex b ON r.end_id = b.id '
                    f'JOIN ag_catalog.ag_label la ON la.id = (a.id::text::bigint >> 48) '
                    f'JOIN ag_catalog.ag_label lr ON lr.id = (r.id::text::bigint >> 48) '
                    f'JOIN ag_catalog.ag_label lb ON lb.id = (b.id::text::bigint >> 48) '
                    f'JOIN ag_catalog.ag_graph g ON g.graphid = lr.graph '
                    f'WHERE g.name = \'{graph_name}\' LIMIT 5'
                )
            )).all()

        report_path = Path(__file__).parent / 'graph_collapse_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('=== Отчет детектора коллапса модели ===\n\n')

            f.write('--- Топ-20 меток узлов (Labels) ---\n')
            for row in label_rows:
                lbl = str(row.label) if row.label is not None else ''
                cnt = str(row.cnt) if row.cnt is not None else '0'
                f.write(f'  {lbl:<25} {cnt:<10}\n')
            f.write('\n')

            f.write('--- Топ-20 имен узлов (Names) ---\n')
            for row in name_rows:
                name = str(row.name) if row.name is not None else '<NULL>'
                cnt = str(row.cnt) if row.cnt is not None else '0'
                f.write(f'  {name:<40} {cnt:<10}\n')
            f.write('\n')

            f.write('--- Топ-20 типов связей (Relations) ---\n')
            for row in rel_rows:
                rel = str(row.relation_type) if row.relation_type is not None else ''
                cnt = str(row.cnt) if row.cnt is not None else '0'
                f.write(f'  {rel:<30} {cnt:<10}\n')
            f.write('\n')

            f.write('--- 10 случайных триплетов графа ---\n')
            for row in triplet_sample:
                sl = str(row.label_a) if row.label_a is not None else ''
                sn = str(row.name_a) if row.name_a is not None else '<NULL>'
                rl = str(row.label_r) if row.label_r is not None else ''
                tl = str(row.label_b) if row.label_b is not None else ''
                tn = str(row.name_b) if row.name_b is not None else '<NULL>'
                f.write(f'  ({sl}, {sn}) - [{rl}] -> ({tl}, {tn})\n')
            f.write('\n')

            f.write('--- 5 последних успешных связей ---\n')
            for row in recent_rows:
                sl = str(row.label_a) if row.label_a is not None else ''
                sn = str(row.name_a) if row.name_a is not None else '<NULL>'
                rl = str(row.label_r) if row.label_r is not None else ''
                tl = str(row.label_b) if row.label_b is not None else ''
                tn = str(row.name_b) if row.name_b is not None else '<NULL>'
                f.write(f'  ({sl}, {sn}) - [{rl}] -> ({tl}, {tn})\n')

    await engine.dispose()


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

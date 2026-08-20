from __future__ import annotations

from src.graph.client import Neo4jClient


SCHEMA_QUERIES: list[str] = [
    "CREATE CONSTRAINT constraint_actor_id IF NOT EXISTS FOR (a:Actor) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT constraint_post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT constraint_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT constraint_organization_id IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE",
    "CREATE CONSTRAINT constraint_product_id IF NOT EXISTS FOR (pr:Product) REQUIRE pr.id IS UNIQUE",
    "CREATE CONSTRAINT constraint_concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT constraint_concept_code IF NOT EXISTS FOR (c:Concept) REQUIRE c.code IS UNIQUE",
    "CREATE CONSTRAINT constraint_event_id IF NOT EXISTS FOR (ev:Event) REQUIRE ev.id IS UNIQUE",
    "CREATE CONSTRAINT constraint_micro_concept_id IF NOT EXISTS FOR (mc:MicroConcept) REQUIRE mc.id IS UNIQUE",
    "CREATE CONSTRAINT constraint_hashtag_id IF NOT EXISTS FOR (h:Hashtag) REQUIRE h.id IS UNIQUE",
    "CREATE INDEX idx_actor_name_lower IF NOT EXISTS FOR (n:Actor) ON (n.name_lower)",
    "CREATE INDEX idx_entity_name_lower IF NOT EXISTS FOR (n:Entity) ON (n.name_lower)",
    "CREATE INDEX idx_org_name_lower IF NOT EXISTS FOR (n:Organization) ON (n.name_lower)",
    "CREATE INDEX idx_product_name_lower IF NOT EXISTS FOR (n:Product) ON (n.name_lower)",
    "CREATE INDEX idx_event_name_lower IF NOT EXISTS FOR (n:Event) ON (n.name_lower)",
    "CREATE INDEX idx_microconcept_name_lower IF NOT EXISTS FOR (n:MicroConcept) ON (n.name_lower)",
    "CREATE INDEX idx_hashtag_name_lower IF NOT EXISTS FOR (n:Hashtag) ON (n.name_lower)",
    "CREATE INDEX idx_concept_name_lower IF NOT EXISTS FOR (n:Concept) ON (n.name_lower)",
    "CREATE INDEX idx_actor_handle IF NOT EXISTS FOR (a:Actor) ON (a.handle)",
    "CREATE INDEX idx_post_published IF NOT EXISTS FOR (p:Post) ON (p.published_at)",
    "CREATE INDEX idx_post_account_id IF NOT EXISTS FOR (p:Post) ON (p.account_id)",
    "CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS FOR (n:Entity|Actor|Organization|Product|Event|MicroConcept|Concept|Hashtag) ON EACH [n.name]",
]


async def init_neo4j_schema(client: Neo4jClient) -> None:
    for query in SCHEMA_QUERIES:
        await client.execute_write(query)

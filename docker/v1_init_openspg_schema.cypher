CREATE CONSTRAINT constr_actor_id IF NOT EXISTS FOR (n:Actor) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT constr_post_id IF NOT EXISTS FOR (n:Post) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT constr_entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT constr_org_id IF NOT EXISTS FOR (n:Organization) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT constr_product_id IF NOT EXISTS FOR (n:Product) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT constr_event_id IF NOT EXISTS FOR (n:Event) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT constr_microconcept_id IF NOT EXISTS FOR (n:MicroConcept) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT constr_concept_id IF NOT EXISTS FOR (n:Concept) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT constr_concept_code IF NOT EXISTS FOR (n:Concept) REQUIRE n.code IS UNIQUE;
CREATE CONSTRAINT constr_hashtag_id IF NOT EXISTS FOR (n:Hashtag) REQUIRE n.id IS UNIQUE;

CREATE INDEX idx_actor_name_lower IF NOT EXISTS FOR (n:Actor) ON (n.name_lower);
CREATE INDEX idx_entity_name_lower IF NOT EXISTS FOR (n:Entity) ON (n.name_lower);
CREATE INDEX idx_org_name_lower IF NOT EXISTS FOR (n:Organization) ON (n.name_lower);
CREATE INDEX idx_product_name_lower IF NOT EXISTS FOR (n:Product) ON (n.name_lower);
CREATE INDEX idx_event_name_lower IF NOT EXISTS FOR (n:Event) ON (n.name_lower);
CREATE INDEX idx_microconcept_name_lower IF NOT EXISTS FOR (n:MicroConcept) ON (n.name_lower);
CREATE INDEX idx_hashtag_name_lower IF NOT EXISTS FOR (n:Hashtag) ON (n.name_lower);
CREATE INDEX idx_concept_name_lower IF NOT EXISTS FOR (n:Concept) ON (n.name_lower);
CREATE INDEX idx_post_published_at IF NOT EXISTS FOR (n:Post) ON (n.published_at);
CREATE INDEX idx_post_account_id IF NOT EXISTS FOR (n:Post) ON (n.account_id);

CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS FOR (n:Entity|Actor|Organization|Product|Event|MicroConcept|Concept|Hashtag) ON EACH [n.name];
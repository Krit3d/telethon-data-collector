from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.graph.utils import build_node_id, clean_name_lower, is_author_entity, is_garbage_value

import logging

logger = logging.getLogger(__name__)


class EntityType(StrEnum):
    Actor = "Actor"
    Post = "Post"
    Entity = "Entity"
    Organization = "Organization"
    Product = "Product"
    Concept = "Concept"
    Event = "Event"
    MicroConcept = "MicroConcept"
    Hashtag = "Hashtag"


class RelationType(StrEnum):
    PUBLISHED = "PUBLISHED"
    MENTIONS = "MENTIONS"
    ABOUT = "ABOUT"
    BELONGS_TO = "BELONGS_TO"
    COVERS_TOPIC = "COVERS_TOPIC"
    COAUTHOR = "COAUTHOR"
    WORKS_AT = "WORKS_AT"
    TAGGED_WITH = "TAGGED_WITH"
    MAPS_TO = "MAPS_TO"
    RELATED_TO = "RELATED_TO"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    USES_TECH = "USES_TECH"
    PRODUCES = "PRODUCES"
    PARENT_OF = "PARENT_OF"


class PlatformType(StrEnum):
    instagram = "instagram"
    telegram = "telegram"
    youtube = "youtube"
    tiktok = "tiktok"
    threads = "threads"


class PostType(StrEnum):
    post = "post"
    reel = "reel"
    short = "short"
    tiktok = "tiktok"
    video = "video"


class ToneType(StrEnum):
    analytical = "analytical"
    expert = "expert"
    provocative = "provocative"
    educational = "educational"
    entertainment = "entertainment"
    casual = "casual"


class HormoneType(StrEnum):
    dopamine = "dopamine"
    oxytocin = "oxytocin"
    serotonin = "serotonin"
    cortisol = "cortisol"
    adrenaline = "adrenaline"
    endorphin = "endorphin"


class SentimentType(StrEnum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


_SENTIMENT_VALUES: frozenset[str] = frozenset({member.value for member in SentimentType})


class EntityCategory(StrEnum):
    technology = "technology"
    method = "method"
    person = "person"
    term = "term"
    general = "general"


class OrgType(StrEnum):
    company = "company"
    brand = "brand"
    agency = "agency"
    media = "media"
    non_profit = "non_profit"


class ProductType(StrEnum):
    software = "software"
    gadget = "gadget"
    course = "course"
    app = "app"
    physical_good = "physical_good"
    service = "service"


class EventType(StrEnum):
    conference = "conference"
    festival = "festival"
    competition = "competition"
    incident = "incident"


PRODUCT_TYPE_VALUES: frozenset[str] = frozenset({e.value for e in ProductType})
ORG_TYPE_VALUES: frozenset[str] = frozenset({e.value for e in OrgType})
ENTITY_CATEGORY_VALUES: frozenset[str] = frozenset({e.value for e in EntityCategory})
EVENT_TYPE_VALUES: frozenset[str] = frozenset({e.value for e in EventType})

PREFIX_TO_LABEL_MAP: dict[str, EntityType] = {
    "event_publication_": EntityType.Post,
    "microconcept_": EntityType.MicroConcept,
    "organization_": EntityType.Organization,
    "hashtag_": EntityType.Hashtag,
    "concept_": EntityType.Concept,
    "product_": EntityType.Product,
    "entity_": EntityType.Entity,
    "actor_": EntityType.Actor,
    "event_": EntityType.Event,
}


class RoleType(StrEnum):
    founder = "founder"
    executive = "executive"
    employee = "employee"
    ambassador = "ambassador"
    advisor = "advisor"


class RelationSubtype(StrEnum):
    creator = "creator"
    promoter = "promoter"
    affiliate = "affiliate"
    vendor = "vendor"
    publisher = "publisher"
    distributor = "distributor"
    sponsor = "sponsor"


RELATION_DOMAIN_RANGE_MAP: dict[RelationType, dict[str, set[EntityType]]] = {
    RelationType.PUBLISHED: {
        "sources": {EntityType.Actor},
        "targets": {EntityType.Post},
    },
    RelationType.MENTIONS: {
        "sources": {EntityType.Post},
        "targets": {
            EntityType.Entity,
            EntityType.Organization,
            EntityType.Product,
            EntityType.Event,
        },
    },
    RelationType.ABOUT: {
        "sources": {EntityType.Post},
        "targets": {EntityType.MicroConcept},
    },
    RelationType.BELONGS_TO: {
        "sources": {
            EntityType.Entity,
            EntityType.Product,
            EntityType.Organization,
            EntityType.Event,
            EntityType.MicroConcept,
        },
        "targets": {EntityType.MicroConcept, EntityType.Concept},
    },
    RelationType.COVERS_TOPIC: {
        "sources": {EntityType.Actor},
        "targets": {EntityType.Concept, EntityType.MicroConcept},
    },
    RelationType.COAUTHOR: {
        "sources": {EntityType.Actor},
        "targets": {EntityType.Actor},
    },
    RelationType.WORKS_AT: {
        "sources": {EntityType.Actor},
        "targets": {EntityType.Organization},
    },
    RelationType.TAGGED_WITH: {
        "sources": {EntityType.Post},
        "targets": {EntityType.Hashtag},
    },
    RelationType.MAPS_TO: {
        "sources": {EntityType.Hashtag},
        "targets": {
            EntityType.Entity,
            EntityType.Product,
            EntityType.Organization,
            EntityType.Event,
            EntityType.MicroConcept,
        },
    },
    RelationType.RELATED_TO: {
        "sources": {EntityType.Entity},
        "targets": {EntityType.Entity, EntityType.Organization, EntityType.Product},
    },
    RelationType.PARTICIPATED_IN: {
        "sources": {EntityType.Actor},
        "targets": {EntityType.Event},
    },
    RelationType.USES_TECH: {
        "sources": {EntityType.Actor, EntityType.Post},
        "targets": {EntityType.Product, EntityType.Entity},
    },
    RelationType.PRODUCES: {
        "sources": {EntityType.Organization, EntityType.Actor},
        "targets": {EntityType.Product},
    },
    RelationType.PARENT_OF: {
        "sources": {EntityType.Concept},
        "targets": {EntityType.Concept},
    },
}


def validate_and_repair_relation(
    source_id: str,
    source_label: EntityType,
    target_id: str,
    target_label: EntityType,
    relation_type: RelationType,
) -> tuple[str, EntityType, str, EntityType, RelationType] | None:
    if source_id == target_id:
        return None
    rule = RELATION_DOMAIN_RANGE_MAP.get(relation_type)
    if rule is None:
        return None
    if relation_type == RelationType.BELONGS_TO:
        if source_label in {EntityType.Entity, EntityType.Product, EntityType.Organization, EntityType.Event} and target_label == EntityType.MicroConcept:
            return (source_id, source_label, target_id, target_label, relation_type)
        if source_label == EntityType.MicroConcept and target_label == EntityType.Concept:
            return (source_id, source_label, target_id, target_label, relation_type)
        if target_label in {EntityType.Entity, EntityType.Product, EntityType.Organization, EntityType.Event} and source_label == EntityType.MicroConcept:
            return (target_id, target_label, source_id, source_label, relation_type)
        if target_label == EntityType.MicroConcept and source_label == EntityType.Concept:
            return (target_id, target_label, source_id, source_label, relation_type)
        return None
    if source_label in rule["sources"] and target_label in rule["targets"]:
        return (source_id, source_label, target_id, target_label, relation_type)
    if target_label in rule["sources"] and source_label in rule["targets"]:
        return (target_id, target_label, source_id, source_label, relation_type)
    return None


def extract_entity_subtype(label: EntityType | str, properties: dict[str, Any]) -> tuple[str | None, str | None]:
    if label == EntityType.Entity:
        val = properties.get("entity_type")
        if val is not None:
            return ("entity_type", str(val))
    if label == EntityType.Organization:
        val = properties.get("org_type")
        if val is not None:
            return ("org_type", str(val))
    if label == EntityType.Product:
        val = properties.get("product_type")
        if val is not None:
            return ("product_type", str(val))
    if label == EntityType.Event:
        val = properties.get("event_type")
        if val is not None:
            return ("event_type", str(val))
    return (None, None)


class ActorNode(BaseModel):
    id: str
    account_id: int
    name: str
    name_lower: str | None = None
    handle: str
    location_name: str | None = None
    platform: PlatformType
    platform_id: str
    primary_language: str | None = None
    primary_tone: ToneType | None = None
    secondary_tone: ToneType | None = None
    primary_hormone: HormoneType | None = None
    secondary_hormone: HormoneType | None = None

    @model_validator(mode="after")
    def _fill_name_lower(self) -> ActorNode:
        self.name_lower = clean_name_lower(self.name)
        return self


class PostNode(BaseModel):
    id: str
    account_id: int
    content_id: int
    published_at: int
    platform: PlatformType
    post_type: PostType = PostType.post
    language: str | None = None
    tone: ToneType | None = None
    secondary_tone: ToneType | None = None
    primary_hormone: HormoneType | None = None
    secondary_hormone: HormoneType | None = None
    score_dopamine: float = Field(default=0.0, ge=0.0, le=1.0)
    score_oxytocin: float = Field(default=0.0, ge=0.0, le=1.0)
    score_serotonin: float = Field(default=0.0, ge=0.0, le=1.0)
    score_cortisol: float = Field(default=0.0, ge=0.0, le=1.0)
    score_adrenaline: float = Field(default=0.0, ge=0.0, le=1.0)
    score_endorphin: float = Field(default=0.0, ge=0.0, le=1.0)
    is_video: bool = False
    is_spam_or_gambling: bool = False


class EntityNode(BaseModel):
    id: str | None = None
    name: str
    name_lower: str | None = None
    entity_type: EntityCategory
    mentions_count: int = 1

    @model_validator(mode="after")
    def _fill_name_lower(self) -> EntityNode:
        self.name_lower = clean_name_lower(self.name)
        return self


class OrganizationNode(BaseModel):
    id: str | None = None
    name: str
    name_lower: str | None = None
    org_type: OrgType
    mentions_count: int = 1

    @model_validator(mode="after")
    def _fill_name_lower(self) -> OrganizationNode:
        self.name_lower = clean_name_lower(self.name)
        return self


class ProductNode(BaseModel):
    id: str | None = None
    name: str
    name_lower: str | None = None
    product_type: ProductType
    mentions_count: int = 1

    @model_validator(mode="after")
    def _fill_name_lower(self) -> ProductNode:
        self.name_lower = clean_name_lower(self.name)
        return self


class EventNode(BaseModel):
    id: str | None = None
    name: str
    name_lower: str | None = None
    event_type: EventType
    mentions_count: int = 1

    @model_validator(mode="after")
    def _fill_name_lower(self) -> EventNode:
        self.name_lower = clean_name_lower(self.name)
        return self


class MicroConceptNode(BaseModel):
    id: str | None = None
    name: str
    name_lower: str | None = None
    is_classified: bool = False

    @model_validator(mode="after")
    def _fill_name_lower(self) -> MicroConceptNode:
        self.name_lower = clean_name_lower(self.name)
        return self


class ConceptNode(BaseModel):
    id: str
    code: str
    name: str
    name_lower: str | None = None
    tier_1: str
    tier_2: str | None = None
    tier_3: str | None = None
    tier_4: str | None = None
    extension: str | None = None

    @model_validator(mode="after")
    def _fill_name_lower(self) -> ConceptNode:
        self.name_lower = clean_name_lower(self.name)
        return self


class HashtagNode(BaseModel):
    id: str
    name: str
    name_lower: str | None = None
    mentions_count: int = 1

    @model_validator(mode="after")
    def _fill_name_lower(self) -> HashtagNode:
        self.name_lower = clean_name_lower(self.name)
        return self


class HashtagItem(BaseModel):
    raw: str
    normalized: str


class ExtractedEntity(BaseModel):
    id: str | None = None
    name: str
    name_lower: str | None = None
    label: EntityType
    properties: dict[str, Any] = {}
    microconcept: str | None = None
    confidence: float = 0.8
    embedding: list[float] | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _fill_name_lower(self) -> ExtractedEntity:
        self.name_lower = clean_name_lower(self.name)
        return self


class ExtractedRelation(BaseModel):
    source_id: str
    source_label: EntityType
    target_id: str
    target_label: EntityType
    relation_type: RelationType
    properties: dict[str, Any] = {}
    confidence: float | None = None

    @model_validator(mode="after")
    def _validate_and_repair(self) -> ExtractedRelation:
        repaired = validate_and_repair_relation(
            self.source_id,
            self.source_label,
            self.target_id,
            self.target_label,
            self.relation_type,
        )
        if repaired is None:
            raise ValueError(f"Invalid relation schema: {self.source_id} ({self.source_label}) -[{self.relation_type}]-> {self.target_id} ({self.target_label})")
        self.source_id, self.source_label, self.target_id, self.target_label, self.relation_type = repaired
        props_conf = self.properties.get("confidence")
        if props_conf is not None and not isinstance(props_conf, bool):
            try:
                parsed_conf = float(props_conf)
            except (TypeError, ValueError):
                parsed_conf = None
            if parsed_conf is not None:
                self.confidence = 1.0 if parsed_conf >= 0.9 else 0.8
        if self.confidence is not None:
            self.properties["confidence"] = self.confidence
        return self


class ExtractedPsychographics(BaseModel):
    language: str | None = None
    primary_tone: ToneType | None = None
    secondary_tone: ToneType | None = None
    tone_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    primary_hormone: HormoneType | None = None
    secondary_hormone: HormoneType | None = None
    score_dopamine: float = Field(default=0.0, ge=0.0, le=1.0)
    score_oxytocin: float = Field(default=0.0, ge=0.0, le=1.0)
    score_serotonin: float = Field(default=0.0, ge=0.0, le=1.0)
    score_cortisol: float = Field(default=0.0, ge=0.0, le=1.0)
    score_adrenaline: float = Field(default=0.0, ge=0.0, le=1.0)
    score_endorphin: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _derive_hormone_scores(self) -> ExtractedPsychographics:
        if self.primary_tone is None and self.secondary_tone is not None:
            self.primary_tone = self.secondary_tone
            self.secondary_tone = None
        elif self.secondary_tone is not None and self.secondary_tone == self.primary_tone:
            self.secondary_tone = None
        scores = (
            (self.score_dopamine, HormoneType.dopamine),
            (self.score_oxytocin, HormoneType.oxytocin),
            (self.score_serotonin, HormoneType.serotonin),
            (self.score_cortisol, HormoneType.cortisol),
            (self.score_adrenaline, HormoneType.adrenaline),
            (self.score_endorphin, HormoneType.endorphin),
        )
        sorted_scores = sorted(scores, key=lambda item: item[0], reverse=True)
        if sorted_scores[0][0] >= 0.05:
            self.primary_hormone = sorted_scores[0][1]
            if sorted_scores[1][0] >= 0.05:
                self.secondary_hormone = sorted_scores[1][1]
            else:
                self.secondary_hormone = None
        else:
            self.primary_hormone = None
            self.secondary_hormone = None
        return self


class OpenSPGExtractionResult(BaseModel):
    thinking: str = ""
    entities: list[ExtractedEntity] = []
    relations: list[ExtractedRelation] = []
    psychographics: ExtractedPsychographics
    is_spam_or_gambling: bool = False
    hashtags: list[HashtagItem] = []

    def sanitize_and_validate(
        self,
        allowed_ids: set[str] | None = None,
        forbidden_names: set[str] | None = None,
        author_title: str | None = None,
        author_handle: str | None = None,
    ) -> OpenSPGExtractionResult:
        phantom_names = frozenset({"<name>", "unknown", "null", "none", "n/a", ""})
        valid_ids = set(allowed_ids) if allowed_ids else set()
        clean_entities: list[ExtractedEntity] = []
        for e in self.entities:
            if e.name.lower() in phantom_names or len(e.name.strip()) < 2:
                continue
            if forbidden_names and e.name_lower in forbidden_names:
                continue
            if author_title and is_author_entity(e.name, author_title, author_handle or ""):
                continue
            if is_garbage_value(e.name, e.label):
                continue

            for key in ("role", "platform", "author", "post", "sentiment", "confidence", "type", "label", "name"):
                e.properties.pop(key, None)

            if e.label == EntityType.Product:
                val = e.properties.get("product_type")
                if val is None:
                    logger.warning(f"Entity '{e.name}' (label={e.label}): missing required 'product_type'")
                    continue
                try:
                    e.properties["product_type"] = ProductType(str(val).lower().strip())
                except ValueError:
                    logger.warning(f"Entity '{e.name}' (label={e.label}): invalid 'product_type' value: {val}")
                    continue
                e.properties = {k: v for k, v in e.properties.items() if k == "product_type"}
            elif e.label == EntityType.Organization:
                val = e.properties.get("org_type")
                if val is None:
                    logger.warning(f"Entity '{e.name}' (label={e.label}): missing required 'org_type'")
                    continue
                try:
                    e.properties["org_type"] = OrgType(str(val).lower().strip())
                except ValueError:
                    logger.warning(f"Entity '{e.name}' (label={e.label}): invalid 'org_type' value: {val}")
                    continue
                e.properties = {k: v for k, v in e.properties.items() if k == "org_type"}
            elif e.label == EntityType.Entity:
                val = e.properties.get("entity_type")
                if val is None:
                    logger.warning(f"Entity '{e.name}' (label={e.label}): missing required 'entity_type'")
                    continue
                try:
                    e.properties["entity_type"] = EntityCategory(str(val).lower().strip())
                except ValueError:
                    logger.warning(f"Entity '{e.name}' (label={e.label}): invalid 'entity_type' value: {val}")
                    continue
                e.properties = {k: v for k, v in e.properties.items() if k == "entity_type"}
            elif e.label == EntityType.Event:
                val = e.properties.get("event_type")
                if val is None:
                    logger.warning(f"Entity '{e.name}' (label={e.label}): missing required 'event_type'")
                    continue
                try:
                    e.properties["event_type"] = EventType(str(val).lower().strip())
                except ValueError:
                    logger.warning(f"Entity '{e.name}' (label={e.label}): invalid 'event_type' value: {val}")
                    continue
                e.properties = {k: v for k, v in e.properties.items() if k == "event_type"}
            elif e.label == EntityType.MicroConcept:
                e.properties = {"is_classified": bool(e.properties.get("is_classified", False))}
            elif e.label == EntityType.Hashtag:
                e.properties = {k: v for k, v in e.properties.items() if k in ("raw", "normalized")}
            else:
                e.properties.clear()

            if not e.id:
                e.id = build_node_id(e.label, e.name)
            clean_entities.append(e)
            valid_ids.add(e.id)

        self.entities = clean_entities

        for h in self.hashtags:
            valid_ids.add(build_node_id(EntityType.Hashtag, h.raw))
            valid_ids.add(build_node_id(EntityType.Hashtag, h.normalized))

        clean_relations: list[ExtractedRelation] = []
        for r in self.relations:
            source_is_concept = r.source_label == EntityType.Concept or r.source_id.startswith("concept_")
            target_is_concept = r.target_label == EntityType.Concept or r.target_id.startswith("concept_")
            source_valid = source_is_concept or r.source_id in valid_ids
            target_valid = target_is_concept or r.target_id in valid_ids
            if not (source_valid and target_valid):
                continue
            repaired = validate_and_repair_relation(
                r.source_id,
                r.source_label,
                r.target_id,
                r.target_label,
                r.relation_type,
            )
            if repaired is None:
                continue
            r.source_id, r.source_label, r.target_id, r.target_label, r.relation_type = repaired
            clean_relations.append(r)
        self.relations = clean_relations

        clean_hashtags: list[HashtagItem] = []
        for h in self.hashtags:
            if not h.raw or not h.normalized:
                continue
            if is_garbage_value(h.raw, EntityType.Hashtag) or is_garbage_value(h.normalized, EntityType.Hashtag):
                continue
            clean_hashtags.append(h)
        self.hashtags = clean_hashtags
        return self


VECTORIZABLE_ENTITY_LABELS: frozenset[EntityType] = frozenset({
    EntityType.Entity,
    EntityType.Organization,
    EntityType.Product,
    EntityType.Event,
})
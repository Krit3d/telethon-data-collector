from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.graph.utils import is_garbage_value


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
    shorts = "short"
    tiktok = "tiktok"
    video = "video"


class ToneType(StrEnum):
    analytical = "analytical"
    expert = "expert"
    provocative = "provocative"
    educational = "educational"
    entertainment = "entertainment"
    casual = "casual"
    hype_train = "hype_train"
    sell_courses = "sell_courses"


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
    release = "release"
    incident = "incident"
    festival = "festival"
    trend = "trend"


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
    RelationType.PARTICIPATED_IN: {
        "sources": {EntityType.Actor, EntityType.Entity},
        "targets": {EntityType.Event},
    },
    RelationType.WORKS_AT: {
        "sources": {EntityType.Actor, EntityType.Entity},
        "targets": {EntityType.Organization},
    },
    RelationType.PRODUCES: {
        "sources": {EntityType.Organization, EntityType.Actor},
        "targets": {EntityType.Product},
    },
    RelationType.USES_TECH: {
        "sources": {EntityType.Actor, EntityType.Post},
        "targets": {EntityType.Product, EntityType.Entity},
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
    RelationType.TAGGED_WITH: {
        "sources": {EntityType.Post},
        "targets": {EntityType.Hashtag},
    },
    RelationType.MAPS_TO: {
        "sources": {EntityType.Hashtag},
        "targets": {
            EntityType.Product,
            EntityType.Entity,
            EntityType.Organization,
            EntityType.Event,
        },
    },
    RelationType.BELONGS_TO: {
        "sources": {
            EntityType.Actor,
            EntityType.Entity,
            EntityType.Product,
            EntityType.Organization,
            EntityType.Event,
            EntityType.Hashtag,
            EntityType.MicroConcept,
        },
        "targets": {EntityType.MicroConcept, EntityType.Concept},
    },
    RelationType.PARENT_OF: {
        "sources": {EntityType.Concept},
        "targets": {EntityType.Concept},
    },
    RelationType.COVERS_TOPIC: {
        "sources": {EntityType.Actor},
        "targets": {EntityType.Concept, EntityType.MicroConcept},
    },
    RelationType.COAUTHOR: {
        "sources": {EntityType.Actor},
        "targets": {EntityType.Actor},
    },
    RelationType.RELATED_TO: {
        "sources": {EntityType.Entity},
        "targets": {EntityType.Entity, EntityType.Organization, EntityType.Product},
    },
    RelationType.PUBLISHED: {
        "sources": {EntityType.Actor},
        "targets": {EntityType.Post},
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
        return (source_id, source_label, target_id, target_label, RelationType.RELATED_TO)
    if source_label in rule["sources"] and target_label in rule["targets"]:
        return (source_id, source_label, target_id, target_label, relation_type)
    if (
        relation_type == RelationType.PARTICIPATED_IN
        and source_label == EntityType.Organization
        and target_label in {EntityType.Actor, EntityType.Entity}
    ):
        return (target_id, target_label, source_id, source_label, RelationType.WORKS_AT)
    if target_label in rule["sources"] and source_label in rule["targets"]:
        return (target_id, target_label, source_id, source_label, relation_type)
    if source_label == EntityType.Post:
        return (source_id, source_label, target_id, target_label, RelationType.MENTIONS)
    return (source_id, source_label, target_id, target_label, RelationType.RELATED_TO)


def _clean_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", name.lower(), flags=re.UNICODE)
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def is_author_entity(entity_name: str, author_title: str, author_handle: str) -> bool:
    def _clean(value: str) -> str:
        return re.sub(r"[^\w\s]", "", str(value or "")).strip().lower()

    clean_entity = _clean(entity_name)
    clean_title = _clean(author_title)
    clean_handle = _clean(author_handle)
    if not clean_entity or not clean_title:
        return False
    if clean_entity == clean_title:
        return True
    if clean_handle and (clean_entity == clean_handle or clean_handle in clean_entity):
        return True
    entity_tokens = [t for t in clean_entity.split() if len(t) >= 3]
    title_tokens = [t for t in clean_title.split() if len(t) >= 3]
    if not entity_tokens or not title_tokens:
        return False
    title_prefixes = {t[:3] for t in title_tokens}
    if not title_prefixes:
        return False
    matched = sum(1 for t in entity_tokens if t[:3] in title_prefixes)
    return matched / len(entity_tokens) >= 0.5 or matched / len(title_tokens) >= 0.5


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
    primary_language: str = "ru"
    primary_sentiment: SentimentType = SentimentType.neutral
    primary_tone: ToneType = ToneType.casual
    primary_hormone: HormoneType = HormoneType.dopamine
    secondary_hormone: HormoneType | None = None
    secondary_tone: ToneType | None = None

    @model_validator(mode="after")
    def _fill_name_lower(self) -> ActorNode:
        self.name_lower = _clean_name(self.name)
        return self


class PostNode(BaseModel):
    id: str
    account_id: int
    content_id: int
    published_at: int
    platform: PlatformType
    post_type: PostType = PostType.post
    language: str = "ru"
    sentiment: SentimentType | None = None
    tone: ToneType | None = None
    secondary_tone: ToneType | None = None
    primary_hormone: HormoneType | None = None
    secondary_hormone: HormoneType | None = None
    score_dopamine: float = 0.0
    score_oxytocin: float = 0.0
    score_serotonin: float = 0.0
    score_cortisol: float = 0.0
    score_adrenaline: float = 0.0
    score_endorphin: float = 0.0
    is_video: bool = False
    is_spam_or_gambling: bool = False


class EntityNode(BaseModel):
    id: str | None = None
    name: str
    name_lower: str | None = None
    entity_type: EntityCategory = EntityCategory.general
    mentions_count: int = 1

    @model_validator(mode="after")
    def _fill_name_lower(self) -> EntityNode:
        self.name_lower = _clean_name(self.name)
        return self


class OrganizationNode(BaseModel):
    id: str | None = None
    name: str
    name_lower: str | None = None
    org_type: OrgType = OrgType.company

    @model_validator(mode="after")
    def _fill_name_lower(self) -> OrganizationNode:
        self.name_lower = _clean_name(self.name)
        return self


class ProductNode(BaseModel):
    id: str | None = None
    name: str
    name_lower: str | None = None
    product_type: ProductType | None = None

    @model_validator(mode="after")
    def _fill_name_lower(self) -> ProductNode:
        self.name_lower = _clean_name(self.name)
        return self


class EventNode(BaseModel):
    id: str | None = None
    name: str
    name_lower: str | None = None
    event_type: EventType | None = None

    @model_validator(mode="after")
    def _fill_name_lower(self) -> EventNode:
        self.name_lower = _clean_name(self.name)
        return self


class MicroConceptNode(BaseModel):
    id: str | None = None
    name: str
    name_lower: str | None = None
    is_classified: bool = False

    @model_validator(mode="after")
    def _fill_name_lower(self) -> MicroConceptNode:
        self.name_lower = _clean_name(self.name)
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
        self.name_lower = _clean_name(self.name)
        return self


class HashtagNode(BaseModel):
    id: str
    name: str
    name_lower: str | None = None

    @model_validator(mode="after")
    def _fill_name_lower(self) -> HashtagNode:
        self.name_lower = _clean_name(self.name)
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
    confidence: float = 1.0
    embedding: list[float] | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _fill_name_lower(self) -> ExtractedEntity:
        self.name_lower = _clean_name(self.name)
        return self


class ExtractedRelation(BaseModel):
    source_id: str
    source_label: EntityType
    target_id: str
    target_label: EntityType
    relation_type: RelationType
    properties: dict[str, Any] = {}
    confidence: float = 1.0

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
            raise ValueError(f"Self-referencing relation rejected: {self.source_id} -> {self.target_id}")
        self.source_id, self.source_label, self.target_id, self.target_label, self.relation_type = repaired
        return self


class ExtractedPsychographics(BaseModel):
    language: str | None = None
    sentiment: SentimentType = SentimentType.neutral
    primary_tone: ToneType
    secondary_tones: list[ToneType] = []
    primary_hormone: HormoneType
    secondary_hormone: HormoneType | None = None
    scores: dict[str, float] = {}
    intent: str = ""


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
            for key in ("role", "platform", "author", "post"):
                e.properties.pop(key, None)
            clean_entities.append(e)
            if e.id:
                valid_ids.add(e.id)
        self.entities = clean_entities
        clean_relations: list[ExtractedRelation] = []
        for r in self.relations:
            if r.source_id not in valid_ids or r.target_id not in valid_ids:
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
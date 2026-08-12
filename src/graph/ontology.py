from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, model_validator


class EntityType(StrEnum):
    Actor = "Actor"
    Post = "Post"
    Entity = "Entity"
    Place = "Place"
    Organization = "Organization"
    Product = "Product"
    Concept = "Concept"
    Event = "Event"
    MicroConcept = "MicroConcept"
    Tone = "Tone"
    Language = "Language"
    Hashtag = "Hashtag"


class RelationType(StrEnum):
    PUBLISHED = "PUBLISHED"
    MENTIONS = "MENTIONS"
    ABOUT = "ABOUT"
    HAS_TONE = "HAS_TONE"
    BELONGS_TO = "BELONGS_TO"
    COVERS_TOPIC = "COVERS_TOPIC"
    COAUTHOR = "COAUTHOR"
    TAGGED_AT = "TAGGED_AT"
    HAS_CONTACT = "HAS_CONTACT"
    BASED_IN = "BASED_IN"
    WORKS_AT = "WORKS_AT"
    TAGGED_WITH = "TAGGED_WITH"
    MAPS_TO = "MAPS_TO"
    RELATED_TO = "RELATED_TO"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    USES_TECH = "USES_TECH"
    PRODUCES = "PRODUCES"
    TARGETS = "TARGETS"
    LOCATED_IN = "LOCATED_IN"
    HAS_LANGUAGE = "HAS_LANGUAGE"
    IS_A = "IS_A"


class PlatformType(StrEnum):
    instagram = "instagram"
    telegram = "telegram"
    youtube = "youtube"
    tiktok = "tiktok"
    threads = "threads"


class PostType(StrEnum):
    post = "post"
    reels = "reel"
    shorts = "short"
    tiktok = "tiktok"
    video = "video"


class ToneType(StrEnum):
    Analytical = "Analytical"
    Provocative = "Provocative"
    Expert = "Expert"
    Hype_Train = "Hype_Train"
    Sell_Courses = "Sell_Courses"
    Educational = "Educational"
    Entertainment = "Entertainment"
    Casual = "Casual"


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


class PlaceType(StrEnum):
    city = "city"
    country = "country"
    region = "region"
    venue = "venue"


def _clean_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", name.lower(), flags=re.UNICODE)
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


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
    primary_tone: ToneType = ToneType.Casual
    primary_hormone: HormoneType = HormoneType.dopamine
    secondary_hormone: HormoneType | None = None

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


class PlaceNode(BaseModel):
    id: str | None = None
    name: str
    name_lower: str | None = None
    city: str | None = None
    country: str | None = None
    country_code: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    place_type: PlaceType = PlaceType.venue
    region: str | None = None

    @model_validator(mode="after")
    def _fill_name_lower(self) -> PlaceNode:
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
    tier_1: str
    tier_2: str | None = None
    tier_3: str | None = None
    tier_4: str | None = None
    extension: str | None = None


class ToneNode(BaseModel):
    id: str
    name: ToneType


class LanguageNode(BaseModel):
    id: str
    code: str
    name: str


class HashtagNode(BaseModel):
    id: str
    name: str
    name_lower: str | None = None

    @model_validator(mode="after")
    def _fill_name_lower(self) -> HashtagNode:
        self.name_lower = _clean_name(self.name)
        return self


class ExtractedEntity(BaseModel):
    id: str | None = None
    name: str
    name_lower: str | None = None
    label: EntityType
    properties: dict[str, Any] = {}
    microconcept: str | None = None
    confidence: float = 1.0

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
    def _reject_self_reference(self) -> ExtractedRelation:
        if self.source_id == self.target_id:
            raise ValueError(f"Self-referencing relation rejected: {self.source_id} -> {self.target_id}")
        return self


class ExtractedPsychographics(BaseModel):
    language: str = "ru"
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

    def sanitize_and_validate(self, allowed_ids: set[str] | None = None) -> OpenSPGExtractionResult:
        phantom_names = frozenset({"<name>", "unknown", "null", "none", "n/a", ""})
        valid_ids = set(allowed_ids) if allowed_ids else set()
        clean_entities: list[ExtractedEntity] = []
        for e in self.entities:
            if e.name.lower() in phantom_names or len(e.name.strip()) < 2:
                continue
            clean_entities.append(e)
            if e.id:
                valid_ids.add(e.id)
        self.entities = clean_entities
        self.relations = [r for r in self.relations if r.source_id in valid_ids and r.target_id in valid_ids]
        return self
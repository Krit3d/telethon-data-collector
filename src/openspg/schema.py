from enum import StrEnum

from pydantic import BaseModel


class EntityType(StrEnum):
    PERSON = "PERSON"
    BRAND = "BRAND"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    TOPIC = "TOPIC"


class RelationType(StrEnum):
    MENTIONS = "MENTIONS"
    ABOUT = "ABOUT"
    BELONGS_TO = "BELONGS_TO"
    HAS_TONE = "HAS_TONE"
    HAS_VIBE = "HAS_VIBE"
    COVERS_TOPIC = "COVERS_TOPIC"
    COAUTHOR = "COAUTHOR"
    TAGGED_AT = "TAGGED_AT"
    HAS_CONTACT = "HAS_CONTACT"
    BASED_IN = "BASED_IN"
    WORKS_AT = "WORKS_AT"
    RELATED_TO = "RELATED_TO"


class ToneType(StrEnum):
    Analytical = "Analytical"
    Provocative = "Provocative"
    Expert = "Expert"
    Hype_Train = "Hype_Train"
    Sell_Courses = "Sell_Courses"
    Educational = "Educational"
    Entertainment = "Entertainment"
    Casual = "Casual"


class ExtractedEntity(BaseModel):
    name: str
    label: EntityType
    confidence: float = 1.0
    properties: dict[str, str | int | float | bool] = {}


class ExtractedRelation(BaseModel):
    source_name: str
    relation_type: RelationType
    target_name: str
    confidence: float = 1.0
    properties: dict[str, str | int | float | bool] = {}


class ExtractedPsychographics(BaseModel):
    primary_tone: ToneType
    secondary_tones: list[ToneType] = []
    intent: str


class OpenSPGExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    psychographics: ExtractedPsychographics
    raw_summary: str | None = None
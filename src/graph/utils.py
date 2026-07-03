import json
import logging
import re
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


_TRANSLIT_MAP = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e',
    'ё': 'yo', 'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'j', 'к': 'k',
    'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r',
    'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 'ts',
    'ч': 'ch', 'ш': 'sh', 'щ': 'shch', 'ъ': '', 'ы': 'y', 'ь': '',
    'э': 'e', 'ю': 'yu', 'я': 'ya',
}

_GARBAGE_EXACT: frozenset[str] = frozenset({
    "unknown", "null", "none", "undefined", "", "nan", "null_value",
    "n/a", "n_a", "na", "other", "id", "#", "topic", "category", "language",
    "place", "actor", "event", "entity", "location", "loc", "person",
    "instagram post", "instagram_post", "telegram post", "telegram_post",
    "social_post", "media_post", "publication", "post", "unnamed entity",
    "name", "instagram",
})

_GARBAGE_SUBSTRINGS: frozenset[str] = frozenset({
    "<name>", "name>", "<name",
})


def _find_key_array_start(text: str, key: str) -> int:
    idx = text.find(key)
    if idx == -1:
        return -1
    pos = idx + len(key)
    while pos < len(text) and text[pos] in " \t\n\r":
        pos += 1
    if pos < len(text) and text[pos] == ":":
        pos += 1
    else:
        return -1
    while pos < len(text) and text[pos] in " \t\n\r":
        pos += 1
    if pos < len(text) and text[pos] == "[":
        return pos + 1
    return -1


def _find_key_start(text: str, key: str) -> int:
    idx = text.find(key)
    if idx == -1:
        return -1
    return idx


def _convert_to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}

    if isinstance(obj, BaseModel):
        try:
            return obj.model_dump(exclude_none=True)
        except AttributeError:
            try:
                return obj.dict(exclude_none=True)
            except AttributeError:
                logger.warning("Failed to convert Pydantic model to dict")
                return {}

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, '__dict__'):
        try:
            return dict(obj.__dict__)
        except (TypeError, ValueError):
            pass

    logger.warning("Cannot convert object of type %s to dict", type(obj).__name__)
    return {}


def _merge_metadata_into_properties(
    base_props: dict[str, Any],
    metadata: dict[str, Any] | BaseModel | None,
) -> dict[str, Any]:
    if metadata is None:
        return base_props

    metadata_dict = _convert_to_dict(metadata)

    if not metadata_dict:
        return base_props

    merged = base_props.copy()
    for key, value in metadata_dict.items():
        sanitized_key = sanitize_key(key)
        if value is not None:
            merged[sanitized_key] = value
        elif sanitized_key not in merged:
            merged[sanitized_key] = None
    return merged


def _transliterate_cyrillic(text: str) -> str:
    text = text.lower()
    return ''.join(_TRANSLIT_MAP.get(ch, ch) for ch in text)


def repair_and_load_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()

    if not stripped:
        return {"entities": [], "relations": []}

    try:
        return json.loads(stripped, strict=False)
    except json.JSONDecodeError:
        pass

    in_string = False
    escape_next = False
    for ch in stripped:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string

    if in_string:
        stripped = stripped + '"'

    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False
    last_valid_pos = -1

    for i, ch in enumerate(stripped):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0 and bracket_count == 0:
                last_valid_pos = i
        elif ch == "[":
            bracket_count += 1
        elif ch == "]":
            bracket_count -= 1
            if brace_count == 0 and bracket_count == 0:
                last_valid_pos = i

    if brace_count > 0 or bracket_count > 0:
        if last_valid_pos > 0:
            truncated = stripped[: last_valid_pos + 1]
            truncated += "]" * bracket_count
            truncated += "}" * brace_count
            try:
                return json.loads(truncated, strict=False)
            except json.JSONDecodeError:
                pass

    entities_key_start = _find_key_array_start(stripped, '"entities"')
    relations_key_start = _find_key_array_start(stripped, '"relations"')

    if entities_key_start != -1 or relations_key_start != -1:
        truncate_point = last_valid_pos if last_valid_pos > 0 else len(stripped)

        last_complete_entity_end = -1
        if entities_key_start != -1:
            entities_array_start = entities_key_start
            search_limit = truncate_point if truncate_point < len(stripped) else len(stripped)
            pos = entities_array_start
            while pos < search_limit:
                brace_pos = stripped.find("}", pos, search_limit)
                if brace_pos == -1:
                    break
                depth = 0
                j = entities_array_start
                while j <= brace_pos:
                    c = stripped[j]
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            after = brace_pos + 1
                            while after < search_limit and stripped[after] in " \t\n\r":
                                after += 1
                            if after >= search_limit:
                                last_complete_entity_end = brace_pos
                                break
                            if stripped[after] in (",", "]"):
                                last_complete_entity_end = brace_pos
                                pos = brace_pos + 1
                                if stripped[after] == "]":
                                    break
                                continue
                    j += 1
                break

        last_complete_relation_end = -1
        if relations_key_start != -1:
            relations_array_start = relations_key_start
            search_limit = truncate_point if truncate_point < len(stripped) else len(stripped)
            pos = relations_array_start
            while pos < search_limit:
                brace_pos = stripped.find("}", pos, search_limit)
                if brace_pos == -1:
                    break
                depth = 0
                j = relations_array_start
                while j <= brace_pos:
                    c = stripped[j]
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            after = brace_pos + 1
                            while after < search_limit and stripped[after] in " \t\n\r":
                                after += 1
                            if after >= search_limit:
                                last_complete_relation_end = brace_pos
                                break
                            if stripped[after] in (",", "]"):
                                last_complete_relation_end = brace_pos
                                pos = brace_pos + 1
                                if stripped[after] == "]":
                                    break
                                continue
                    j += 1
                break

        if last_complete_entity_end > 0 or last_complete_relation_end > 0:
            entities_label_start = _find_key_start(stripped, '"entities"')
            relations_label_start = _find_key_start(stripped, '"relations"')

            repaired = "{"
            if entities_label_start != -1:
                repaired += stripped[entities_label_start:entities_array_start]
                if last_complete_entity_end > 0:
                    repaired += stripped[entities_array_start:last_complete_entity_end + 1]
                repaired += "]"
            if relations_label_start != -1:
                if entities_label_start != -1:
                    repaired += ","
                repaired += stripped[relations_label_start:relations_array_start]
                if last_complete_relation_end > 0:
                    repaired += stripped[relations_array_start:last_complete_relation_end + 1]
                repaired += "]"
            repaired += "}"

            try:
                parsed = json.loads(repaired, strict=False)
                if isinstance(parsed.get("entities"), list) and isinstance(parsed.get("relations"), list):
                    return parsed
            except json.JSONDecodeError:
                pass

    if last_valid_pos > 0:
        fallback = stripped[: last_valid_pos + 1]
        if brace_count > 0:
            fallback += "}" * brace_count
        if bracket_count > 0:
            fallback += "]" * bracket_count
        try:
            return json.loads(fallback, strict=False)
        except json.JSONDecodeError:
            pass

    return {"entities": [], "relations": []}


def sanitize_key(key: Any) -> str:
    if not isinstance(key, str):
        key = str(key)
    key = key.replace("-", "_").replace(" ", "_")
    key = re.sub(r'[^A-Za-z0-9_]', '', key)
    key = key.lower()
    return key if key else "unknown_property"


def sanitize_id(value: str) -> str:
    text = _transliterate_cyrillic(value)
    text = re.sub(r'[^a-zA-Z0-9]', '_', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_')


def is_garbage_value(val: str) -> bool:
    stripped = val.strip().lower()
    if len(stripped) < 2:
        return True
    if stripped in _GARBAGE_EXACT:
        return True
    for substr in _GARBAGE_SUBSTRINGS:
        if substr in stripped:
            return True
    return False

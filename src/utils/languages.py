import json
from pathlib import Path
from typing import Any

_ISO_LANGUAGES_PATH = Path(__file__).resolve().parent.parent / "config" / "iso_languages.json"
_ISO_LANGUAGES_DATA: dict[str, Any] | None = None
_ALIASES: dict[str, str] | None = None
_NAME_RU_MAP: dict[str, str] | None = None


def _load_iso_languages() -> dict[str, Any]:
    global _ISO_LANGUAGES_DATA
    data = _ISO_LANGUAGES_DATA
    if data is None:
        with _ISO_LANGUAGES_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        _ISO_LANGUAGES_DATA = data
    return data


def _get_aliases() -> dict[str, str]:
    global _ALIASES
    aliases = _ALIASES
    if aliases is None:
        data = _load_iso_languages()
        aliases = {str(k).strip().lower(): str(v).strip().lower() for k, v in data.get("aliases", {}).items()}
        _ALIASES = aliases
    return aliases


def _get_name_ru_map() -> dict[str, str]:
    global _NAME_RU_MAP
    name_ru_map = _NAME_RU_MAP
    if name_ru_map is None:
        data = _load_iso_languages()
        name_ru_map = {
            str(item.get("code", "")).strip().lower(): str(item.get("name_ru", ""))
            for item in data.get("languages", [])
            if item.get("code")
        }
        _NAME_RU_MAP = name_ru_map
    return name_ru_map


def canonicalize_language(code: str | None) -> str | None:
    if not code:
        return None
    cleaned = code.strip().lower()
    if not cleaned:
        return None
    return _get_aliases().get(cleaned, cleaned)


def canonicalize_languages(codes: list[str] | None) -> set[str]:
    if not codes:
        return set()
    result: set[str] = set()
    for code in codes:
        canonical = canonicalize_language(code)
        if canonical and canonical != "all":
            result.add(canonical)
    return result


def get_all_languages() -> dict[str, Any]:
    return _load_iso_languages()


def get_language_name_ru(code: str) -> str:
    canonical = canonicalize_language(code)
    if canonical:
        name = _get_name_ru_map().get(canonical)
        if name:
            return name
    return code.upper()
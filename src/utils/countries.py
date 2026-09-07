import json
from pathlib import Path
from typing import Any

_ISO_COUNTRIES_PATH = Path(__file__).resolve().parent.parent / "config" / "iso_countries.json"
_ISO_COUNTRIES_DATA: dict[str, Any] | None = None
_LOOKUP_MAP: dict[str, str] | None = None
_NAME_RU_MAP: dict[str, str] | None = None

_SPECIAL_VALUES = {"all", "все", "все страны", "*"}


def _load_iso_countries() -> dict[str, Any]:
    global _ISO_COUNTRIES_DATA
    data = _ISO_COUNTRIES_DATA
    if data is None:
        with _ISO_COUNTRIES_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        _ISO_COUNTRIES_DATA = data
    return data


def _get_lookup_map() -> dict[str, str]:
    global _LOOKUP_MAP
    lookup_map = _LOOKUP_MAP
    if lookup_map is None:
        data = _load_iso_countries()
        lookup_map = {}
        for item in data.get("countries", []):
            code = str(item.get("code", "")).strip().upper()
            if not code:
                continue
            lookup_map[str(item.get("code", "")).strip().lower()] = code
            lookup_map[str(item.get("name_ru", "")).strip().lower()] = code
            lookup_map[str(item.get("name_en", "")).strip().lower()] = code
        for alias, value in data.get("aliases", {}).items():
            lookup_map[str(alias).strip().lower()] = str(value).strip().upper()
        _LOOKUP_MAP = lookup_map
    return lookup_map


def _get_name_ru_map() -> dict[str, str]:
    global _NAME_RU_MAP
    name_ru_map = _NAME_RU_MAP
    if name_ru_map is None:
        data = _load_iso_countries()
        name_ru_map = {
            str(item.get("code", "")).strip().upper(): str(item.get("name_ru", ""))
            for item in data.get("countries", [])
            if item.get("code")
        }
        _NAME_RU_MAP = name_ru_map
    return name_ru_map


def canonicalize_country(code_or_name: str | None) -> str | None:
    if not code_or_name:
        return None
    cleaned = code_or_name.strip().lower()
    if not cleaned or cleaned in _SPECIAL_VALUES:
        return None
    return _get_lookup_map().get(cleaned)


def canonicalize_countries(items: list[str] | None) -> set[str]:
    if not items:
        return set()
    result: set[str] = set()
    for item in items:
        canonical = canonicalize_country(item)
        if canonical:
            result.add(canonical)
    return result


def get_country_name_ru(code: str) -> str:
    canonical = canonicalize_country(code)
    if canonical:
        name = _get_name_ru_map().get(canonical)
        if name:
            return name
    return code.strip().upper()


def get_all_countries() -> dict[str, Any]:
    return _load_iso_countries()
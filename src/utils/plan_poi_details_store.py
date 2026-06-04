from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.path_tool import get_abs_path


DEFAULT_PLAN_POI_DETAILS_PATH = Path(get_abs_path("data/plan_poi_details.json"))


def _resolve_path(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else DEFAULT_PLAN_POI_DETAILS_PATH


def load_plan_poi_details(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    target = _resolve_path(path)
    if not target.exists():
        return {}

    with target.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"plan POI details must be a JSON object: {target}")

    return {
        str(poi_id): detail
        for poi_id, detail in data.items()
        if isinstance(detail, dict)
    }


def save_plan_poi_details(
    details: dict[str, dict[str, Any]],
    path: str | Path | None = None,
) -> None:
    target = _resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)


def get_plan_poi_detail(
    poi_id: str,
    path: str | Path | None = None,
) -> dict[str, Any] | None:
    if not poi_id:
        return None
    return load_plan_poi_details(path).get(poi_id)


def upsert_plan_poi_detail(
    detail: dict[str, Any],
    path: str | Path | None = None,
) -> dict[str, Any]:
    poi_id = str(detail.get("poi_id") or detail.get("id") or "").strip()
    if not poi_id:
        raise ValueError("POI detail requires poi_id")

    details = load_plan_poi_details(path)
    details[poi_id] = detail
    save_plan_poi_details(details, path)
    return detail

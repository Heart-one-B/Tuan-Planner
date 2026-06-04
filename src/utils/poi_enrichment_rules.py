from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any


_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("KTV", ("ktv", "KTV", "唱歌", "量贩")),
    ("餐饮", ("餐饮", "火锅", "烤肉", "烧烤", "餐厅", "小吃")),
    ("咖啡", ("咖啡", "coffee", "Coffee", "COFFEE")),
    ("甜品饮品", ("奶茶", "甜品", "冰淇淋", "饮品", "DQ")),
    ("电影", ("电影院", "影城", "电影")),
    ("桌游", ("桌游", "剧本杀", "棋牌")),
    ("户外休闲", ("公园", "景区", "户外", "绿道")),
)

_AVG_PRICE_RANGES: dict[str, tuple[int, int]] = {
    "KTV": (45, 90),
    "餐饮": (55, 110),
    "咖啡": (25, 45),
    "甜品饮品": (15, 35),
    "电影": (35, 60),
    "桌游": (35, 70),
    "户外休闲": (0, 30),
    "通用": (30, 80),
}

_BASE_TAGS: dict[str, list[str]] = {
    "KTV": ["KTV", "唱歌", "朋友小聚"],
    "餐饮": ["用餐", "聚餐", "本地餐饮"],
    "咖啡": ["咖啡", "聊天", "休息"],
    "甜品饮品": ["甜品", "饮品", "轻松"],
    "电影": ["电影", "室内", "放松"],
    "桌游": ["桌游", "互动", "朋友小聚"],
    "户外休闲": ["户外", "散步", "轻松"],
    "通用": ["休闲", "本地生活", "轻松"],
}

_REVIEW_TEMPLATES: dict[str, list[str]] = {
    "KTV": ["包间私密性较好，适合朋友小聚。", "下午档安排比较轻松，唱歌聊天都合适。"],
    "餐饮": ["菜品选择比较丰富，适合作为行程里的正餐。", "整体用餐节奏轻松，适合边吃边聊。"],
    "咖啡": ["环境适合短暂休息，也方便聊天。", "饮品选择稳定，适合放在活动前后衔接。"],
    "甜品饮品": ["适合短暂停留，给行程增加一点轻松感。", "甜品饮品补充方便，不会占用太长时间。"],
    "电影": ["室内观影节奏稳定，适合作为放松安排。", "时间安排明确，比较容易和用餐衔接。"],
    "桌游": ["互动感比较强，适合朋友一起玩。", "停留时间灵活，可以按现场状态调整。"],
    "户外休闲": ["适合散步放松，整体节奏不会太赶。", "空间相对开阔，适合轻松活动。"],
    "通用": ["整体安排比较稳妥，适合放进行程中衔接。", "停留节奏灵活，可以根据现场状态调整。"],
}


def infer_category(poi: dict[str, Any]) -> str:
    text = " ".join(
        str(poi.get(key) or "")
        for key in ("name", "type", "keyword_source")
    )
    for category, keywords in _CATEGORY_RULES:
        if any(keyword in text for keyword in keywords):
            return category
    return "通用"


def _stable_number(seed: str, low: int, high: int) -> int:
    if high <= low:
        return low
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return low + int(digest[:8], 16) % (high - low + 1)


def _stable_pick(items: list[str], seed: str, count: int) -> list[str]:
    if not items:
        return []
    start = _stable_number(seed, 0, len(items) - 1)
    ordered = items[start:] + items[:start]
    return ordered[:count]


def _normalize_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "、".join(str(item) for item in value if item)
    return ""


def build_enriched_poi_detail(
    *,
    poi_id: str,
    poi_role: str,
    base_poi: dict[str, Any],
    timestamp: str | None = None,
) -> dict[str, Any]:
    now = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    category = infer_category(base_poi)
    low, high = _AVG_PRICE_RANGES.get(category, _AVG_PRICE_RANGES["通用"])
    avg_price = _stable_number(poi_id or base_poi.get("name", ""), low, high)
    business_area = _normalize_text(base_poi.get("business_area"))

    if business_area and category != "通用":
        rank_label = f"{business_area}{category}推荐"
    elif business_area:
        rank_label = f"{business_area}人气推荐"
    elif category != "通用":
        rank_label = f"{category}推荐"
    else:
        rank_label = "本地生活推荐"

    tags = list(_BASE_TAGS.get(category, _BASE_TAGS["通用"]))
    environment = base_poi.get("environment") or ""
    if environment == "indoor" and "室内" not in tags:
        tags.append("室内")
    elif environment == "outdoor" and "户外" not in tags:
        tags.append("户外")

    highlights = []
    if base_poi.get("distance"):
        highlights.append("距离信息明确，行程衔接更容易控制。")
    if business_area:
        highlights.append(f"位于{business_area}，适合与周边安排组合。")
    highlights.append(f"{category}类型适合放入本次休闲计划。")

    reviews = _stable_pick(_REVIEW_TEMPLATES.get(category, _REVIEW_TEMPLATES["通用"]), poi_id, 2)

    return {
        "poi_id": poi_id,
        "name": base_poi.get("name") or "",
        "poi_role": poi_role,
        "category": category,
        "address": base_poi.get("address") or "",
        "location": base_poi.get("location") or "",
        "distance": base_poi.get("distance") or "",
        "rating": base_poi.get("rating") or "",
        "tel": base_poi.get("tel") or "",
        "environment": base_poi.get("environment") or "",
        "business_area": business_area,
        "avg_price": avg_price,
        "rank_label": rank_label,
        "photos": _normalize_list(base_poi.get("photos")),
        "tags": tags,
        "highlights": highlights[:3],
        "reviews": reviews,
        "first_seen_at": now,
        "updated_at": now,
    }

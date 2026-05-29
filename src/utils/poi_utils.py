# src/utils/poi_utils.py
from src.utils.time_utils import _activity_matches_daypart, _restaurant_matches_daypart

def _normalize_text_list(value) -> list[str]:
    """格式化文本输入列表，去除无效和空字符串"""
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    out.append(text)
        return out
    return []


def _merge_text_lists(*values) -> list[str]:
    """合并多个文本输入列表并进行去重"""
    merged: list[str] = []
    for value in values:
        for item in _normalize_text_list(value):
            if item not in merged:
                merged.append(item)
    return merged


def _poi_text(item: dict) -> str:
    """整合并拼装 POI 元数据用于关键词语义检索"""
    tags = item.get("tags") or []
    tags_semantic = item.get("tags_semantic") or []
    parts = [
        item.get("name") or "",
        item.get("description") or "",
        item.get("location") or "",
    ]
    if isinstance(tags, list):
        parts.extend(str(tag) for tag in tags)
    if isinstance(tags_semantic, list):
        parts.extend(str(tag) for tag in tags_semantic)
    return " ".join(part for part in parts if isinstance(part, str)).lower()


def _activity_environment(item: dict) -> str:
    """提取活动的物理环境类型（室内/室外/混合）"""
    if not isinstance(item, dict):
        return "unknown"
    value = item.get("activity_environment")
    if isinstance(value, str) and value in {"indoor", "outdoor", "mixed", "unknown"}:
        return value
    legacy_type = item.get("type")
    if isinstance(legacy_type, str) and legacy_type in {"indoor", "outdoor", "mixed"}:
        return legacy_type
    return "unknown"


def _matches_keywords(item: dict, keywords: list[str], *, require_all: bool = False) -> bool:
    """检查 POI 属性是否匹配关键词列表中的关键词"""
    if not keywords:
        return True
    haystack = _poi_text(item)
    normalized = [k.strip().lower() for k in keywords if isinstance(k, str) and k.strip()]
    if not normalized:
        return True
    if require_all:
        return all(k in haystack for k in normalized)
    return any(k in haystack for k in normalized)


def _contains_excluded_keywords(item: dict, exclude_keywords: list[str]) -> bool:
    """排除命中黑名单关键词的 POI"""
    if not exclude_keywords:
        return False
    haystack = _poi_text(item)
    normalized = [k.strip().lower() for k in exclude_keywords if isinstance(k, str) and k.strip()]
    return any(k in haystack for k in normalized)


def _activity_bucket_key_from_name(name: str) -> str:
    """对活动进行职责大类归纳划分（桶分类法）"""
    text = (name or "").lower()
    if any(token in text for token in ("商场", "购物中心", "步行街", "广场", "mall")):
        return "shopping"
    if any(token in text for token in ("公园", "绿道", "步道")):
        return "park_walk"
    if any(token in text for token in ("展览", "美术馆", "博物馆", "艺术馆")):
        return "exhibition"
    if any(token in text for token in ("亲子", "儿童", "乐园")):
        return "parent_child"
    if any(token in text for token in ("ktv", "剧本杀", "桌游", "轰趴")):
        return "indoor_entertainment"
    return "other"


def _restaurant_bucket_key_from_name(name: str) -> str:
    """对餐厅进行风味类型职责归纳划分"""
    text = (name or "").lower()
    if any(token in text for token in ("火锅", "串串", "hotpot")):
        return "hotpot"
    if any(token in text for token in ("轻食", "沙拉", "健康")):
        return "light_meal"
    if any(token in text for token in ("咖啡", "cafe", "coffee")):
        return "cafe"
    if any(token in text for token in ("烧烤", "烤肉", "bbq")):
        return "bbq"
    if any(token in text for token in ("日料", "寿司", "居酒屋")):
        return "japanese"
    if any(token in text for token in ("西餐", "牛排", "意面")):
        return "western"
    if any(token in text for token in ("简餐", "餐厅", "饭")):
        return "simple_meal"
    return "other"


def _rating_value(item: dict) -> float:
    """评分字符串转换及默认坏值评估"""
    value = item.get("rating")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return -1.0
    return -1.0


def _dedupe_activities_by_identity(items: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip().lower()
        item_id = (item.get("id") or "").strip().lower()
        key = (name, item_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _dedupe_restaurants_by_identity(items: list[dict]) -> list[dict]:
    return _dedupe_activities_by_identity(items)


def _has_open_time_fields(item: dict) -> bool:
    open_time = item.get("open_time")
    opentime2 = item.get("opentime2")
    return bool((isinstance(open_time, str) and open_time.strip()) or (isinstance(opentime2, str) and opentime2.strip()))


def _shortlist_with_detail_gate(
    items: list[dict],
    *,
    bucket_key_fn,
    api,
    daypart: str,
    daypart_match_fn,
    per_bucket_limit: int = 2,
    max_scan_per_bucket: int = 10,
    bucket_order: list[str] | None = None,
    bucket_key_from_item_fn=None,
    allow_unverified_fallback: bool = False,
    total_limit: int | None = None,
    debug_label: str = "",
) -> list[dict]:
    """门禁漏斗精选器：桶级别的高精细度营业时间及详细信息验证过滤算法"""
    buckets = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if bucket_key_from_item_fn is not None:
            bucket = bucket_key_from_item_fn(item)
        else:
            bucket = bucket_key_fn(item.get("name", ""))
        buckets.setdefault(bucket, []).append(dict(item))

    if debug_label:
        print(f"[{debug_label}][DEBUG] candidates_by_bucket={ {k: len(v) for k, v in buckets.items()} }")

    shortlisted = []
    ordered_bucket_keys = []
    if bucket_order:
        for key in bucket_order:
            if key in buckets and key not in ordered_bucket_keys:
                ordered_bucket_keys.append(key)
    for key in buckets:
        if key not in ordered_bucket_keys:
            ordered_bucket_keys.append(key)

    accepted_by_bucket = {}
    for bucket_key in ordered_bucket_keys:
        bucket_items = buckets.get(bucket_key, [])
        ranked = sorted(
            bucket_items,
            key=lambda x: (_rating_value(x), len(x.get("name", ""))),
            reverse=True,
        )
        accepted = []
        scanned = 0
        for candidate in ranked:
            if scanned >= max_scan_per_bucket or len(accepted) >= per_bucket_limit:
                break
            scanned += 1
            enriched_batch = api.enrich_poi_details([candidate])
            enriched = enriched_batch[0] if isinstance(enriched_batch, list) and enriched_batch else candidate
            if _has_open_time_fields(enriched) and daypart_match_fn(enriched, daypart):
                enriched["time_validation_status"] = "verified_open_time"
                accepted.append(enriched)
        if not accepted and allow_unverified_fallback:
            for candidate in ranked[:per_bucket_limit]:
                fallback_item = dict(candidate)
                fallback_item["time_validation_status"] = "unverified_open_time"
                accepted.append(fallback_item)
        accepted_by_bucket[bucket_key] = accepted

    if debug_label:
        print(f"[{debug_label}][DEBUG] accepted_by_bucket={ {k: len(v) for k, v in accepted_by_bucket.items()} }")

    if total_limit is None:
        for bucket_key in ordered_bucket_keys:
            shortlisted.extend(accepted_by_bucket.get(bucket_key, []))
        return shortlisted

    for offset in range(per_bucket_limit):
        for bucket_key in ordered_bucket_keys:
            bucket_items = accepted_by_bucket.get(bucket_key, [])
            if offset < len(bucket_items):
                shortlisted.append(bucket_items[offset])
                if len(shortlisted) >= total_limit:
                    return shortlisted
    return shortlisted
# src/utils/weather_utils.py

_WEATHER_RISKY_ACTIVITY_KEYWORD_TOKENS = (
    "公园", "动物园", "植物园", "游乐园", "景区", "绿道", "步道", "露营", "营地", "农场", "森林", "湿地", "户外", "室外", "草坪", "江滩", "河滨",
)
_WEATHER_SAFE_ACTIVITY_FALLBACK_KEYWORDS = ("科技馆", "博物馆", "水族馆", "海洋馆", "商场", "儿童乐园")


def _weather_requires_indoor(weather_risk: str) -> bool:
    return weather_risk in {"High", "high"}


def _prune_weather_risky_activity_keywords(keywords: list[str], *, indoor_required: bool) -> tuple[list[str], list[str]]:
    """降雨/台风等恶劣天气下，过滤排除户外高风险行业的关键词检索（如公园、露营等）并自动兜底至安全室内活动"""
    if not indoor_required:
        return keywords, []
    kept = []
    removed = []
    for keyword in keywords:
        if not isinstance(keyword, str) or not keyword.strip():
            continue
        text = keyword.strip()
        if any(token in text for token in _WEATHER_RISKY_ACTIVITY_KEYWORD_TOKENS):
            removed.append(text)
        else:
            kept.append(text)
    if kept:
        return kept, removed
    # 如果户外关键词全被剔除导致空集，则自动启用室内安全性兜底
    fallback = [k for k in _WEATHER_SAFE_ACTIVITY_FALLBACK_KEYWORDS if k not in removed]
    return fallback, removed


def _derive_weather_scenario_key(constraints: dict) -> str:
    explicit = constraints.get("weather_scenario")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    date_label = constraints.get("date_label") or ""
    daypart = constraints.get("daypart") or ""
    is_weekend = any(token in date_label for token in ("周末", "周六", "周日", "周天"))
    if daypart == "晚上":
        return "storm"
    if daypart == "下午" and is_weekend:
        return "sunny"
    return "default"
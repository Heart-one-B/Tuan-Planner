# src/utils/time_utils.py
import re

def _parse_hhmm_to_minutes(text: str) -> int | None:
    """将 HH:MM 格式的文本解析为自零点起算的分钟数"""
    if not isinstance(text, str) or ":" not in text:
        return None
    try:
        hour_text, minute_text = text.strip().split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except (TypeError, ValueError):
        return None
    if hour < 0 or hour > 24 or minute < 0 or minute >= 60:
        return None
    return hour * 60 + minute


def _extract_time_ranges(text: str) -> list[tuple[int, int]]:
    """从营业时间文本中提取所有的分钟时间区间对"""
    if not isinstance(text, str) or not text.strip():
        return []
    ranges: list[tuple[int, int]] = []
    pattern = re.compile(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})")
    for start_text, end_text in pattern.findall(text):
        start_minutes = _parse_hhmm_to_minutes(start_text)
        end_minutes = _parse_hhmm_to_minutes(end_text)
        if start_minutes is None or end_minutes is None:
            continue
        if end_minutes <= start_minutes:
            end_minutes += 24 * 60  # 跨天处理
        ranges.append((start_minutes, end_minutes))
    return ranges


def _daypart_window(daypart: str) -> tuple[int, int] | None:
    """时段标签对应的分钟范围区间映射"""
    if daypart == "上午":
        return (9 * 60, 12 * 60)
    if daypart == "下午":
        return (12 * 60, 18 * 60)
    if daypart == "晚上":
        return (18 * 60, 22 * 60)
    if daypart == "全天":
        return (0, 24 * 60)
    return None


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """判断两个时间区间是否存在交集"""
    return max(a_start, b_start) < min(a_end, b_end)


def _matches_daypart_by_time_windows(item: dict, daypart: str) -> bool:
    """校验 POI 的开放/高峰时间是否与指定时段产生重合"""
    if daypart == "全天":
        return True

    window = _daypart_window(daypart)
    if window is None:
        return False
    window_start, window_end = window

    open_time = item.get("open_time")
    opentime2 = item.get("opentime2")
    open_window = f"{open_time or ''} {opentime2 or ''}".strip()
    time_ranges = _extract_time_ranges(open_window)
    for start_minutes, end_minutes in time_ranges:
        if _ranges_overlap(start_minutes, end_minutes, window_start, window_end):
            return True

    peak_hours = item.get("peak_hours")
    if not isinstance(peak_hours, list) or not peak_hours:
        return False
    for slot in peak_hours:
        for start_minutes, end_minutes in _extract_time_ranges(slot):
            if _ranges_overlap(start_minutes, end_minutes, window_start, window_end):
                return True
    return False


def _activity_matches_daypart(activity: dict, daypart: str) -> bool:
    return _matches_daypart_by_time_windows(activity, daypart)


def _restaurant_matches_daypart(restaurant: dict, daypart: str) -> bool:
    return _matches_daypart_by_time_windows(restaurant, daypart)


def _infer_queue_time_slot(time_window: str) -> str:
    """时间窗口转换为评估排队所需时段"""
    if isinstance(time_window, str) and time_window in {"today_evening", "weekend_evening"}:
        return "dinner"
    return "lunch"


def _infer_crowd_time_slot(time_window: str) -> str:
    """时间窗口转换为评估拥挤程度所需时段"""
    mapping = {
        "today_afternoon": "weekend_morning",
        "weekend_afternoon": "weekend_morning",
        "today_evening": "weekday_evening",
        "weekend_evening": "weekend_evening",
    }
    if isinstance(time_window, str):
        return mapping.get(time_window, "weekend_morning")
    return "weekend_morning"
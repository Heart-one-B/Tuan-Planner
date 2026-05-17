"""Retrieval Agent：闲时规划链路上的 Mock RAG 节点。

设计说明
--------
* 完全不依赖 LLM，纯查询：从 ``intent`` 字段中抽取关键词，调用 ``MockToolAPI.semantic_search``
  分别检索活动（``kind="activity"``）与餐厅（``kind="restaurant"``）。
* 任何异常都会被吞掉，返回空骨架 ``{"pois": [], "notes": []}``，避免阻塞主链路。
* 仅保留 ``id`` / ``name`` / ``score`` / ``matched_keywords`` 四个字段，避免污染状态。
"""

from __future__ import annotations

import re
from typing import Any

from src.tools.mock_api import MockToolAPI


# 用于切分 raw_query 的标点/空白集合（中英文常见标点 + 全/半角空格）
_RAW_QUERY_SPLIT_RE = re.compile(
    r"[\s\u3000,，.。!！?？、;；:：()（）\"'\[\]【】《》<>~`@#$%^&*+=|/\\-]+"
)

# 视为"空"的 diet_preference 取值
_EMPTY_DIET_VALUES = {"", "无", "none", "None"}

# 场景 → 关键词映射
_SCENARIO_KEYWORD = {
    "family": "亲子",
    "friends": "聚会",
}

# 仅保留这些字段返回给状态，避免巨大 payload
_KEEP_FIELDS = ("id", "name", "score", "matched_keywords")


def _extract_keywords(intent: dict[str, Any]) -> list[str]:
    """按文档约定的最简规则抽取关键词，去重保序。"""
    keywords: list[str] = []

    def _push(token: str) -> None:
        token = (token or "").strip()
        if not token:
            return
        if token in keywords:
            return
        keywords.append(token)

    # 1) diet_preference
    diet = intent.get("diet_preference")
    if isinstance(diet, str) and diet.strip() not in _EMPTY_DIET_VALUES:
        _push(diet.strip())

    # 2) scenario
    scenario = intent.get("scenario")
    if isinstance(scenario, str):
        mapped = _SCENARIO_KEYWORD.get(scenario.strip())
        if mapped:
            _push(mapped)

    # 3) raw_query 中的中文/词汇 token（最简分词）
    raw_query = intent.get("raw_query")
    if isinstance(raw_query, str) and raw_query.strip():
        for tok in _RAW_QUERY_SPLIT_RE.split(raw_query):
            _push(tok)

    return keywords


def _project(items: Any) -> list[dict[str, Any]]:
    """只保留 id/name/score/matched_keywords 四个字段。"""
    if not isinstance(items, list):
        return []
    projected: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        projected.append({k: item.get(k) for k in _KEEP_FIELDS})
    return projected


class RetrievalAgent:
    """Mock RAG：基于关键词在 mock_db 上做语义子串检索。"""

    def __init__(self) -> None:
        # 延迟到调用时实例化也可以，但 MockToolAPI 加载一次开销很小，提前持有更直观
        self._tools = MockToolAPI()

    def retrieve(self, intent: dict[str, Any]) -> dict[str, Any]:
        """根据 intent 抽取关键词并检索，返回 ``{"pois": [...], "notes": [...]}``。"""
        intent = intent if isinstance(intent, dict) else {}

        try:
            keywords = _extract_keywords(intent)
            print(
                f"[Retrieval Agent] 开始 mock RAG 检索，关键词={keywords}"
            )

            if not keywords:
                print("[Retrieval Agent] 关键词为空，跳过 semantic_search，返回空结果。")
                return {"pois": [], "notes": []}

            activities = self._tools.semantic_search(keywords, kind="activity")
            restaurants = self._tools.semantic_search(keywords, kind="restaurant")

            result = {
                "pois": _project(activities),
                "notes": _project(restaurants),
            }
            print(
                f"[Retrieval Agent] 检索完成：pois={len(result['pois'])} 条，"
                f"notes={len(result['notes'])} 条"
            )
            return result
        except Exception as exc:  # noqa: BLE001 - 演示需求：任意失败都不抛
            print(f"[Retrieval Agent][WARN] 检索失败，返回空结果：{exc}")
            return {"pois": [], "notes": []}

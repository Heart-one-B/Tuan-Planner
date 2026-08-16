# tests/fact_agent/test_cost.py
"""cost 解析的单元测试。无网络依赖，纯函数，秒级跑完。

两条不变量的可执行形式：
  ① 未知 → None，绝不是 0（否则价格断言静默通过）
  ② 顶层 cost 和 biz_ext.cost 两种形状都要认（MCP vs REST）

第 ② 条是真实踩过的坑：只按 REST 的 biz_ext 形状写，
在 MCP 路径下 50 家餐厅覆盖率 0%，看起来像"高德没数据"。
"""
from agents.fact.cost import biz_ext, extract_cost, parse_cost


# ── parse_cost：有效值 ────────────────────────────────────────────────

def test_parse_cost_real_forms():
    """高德真实返回的形态（实测 MCP 返回值）。"""
    assert parse_cost("120.00") == 120.0
    assert parse_cost("105.00") == 105.0
    assert parse_cost("66.00") == 66.0
    assert parse_cost("88") == 88.0
    assert parse_cost(" 120.00 ") == 120.0


def test_parse_cost_numeric_input():
    assert parse_cost(120) == 120.0
    assert parse_cost(88.5) == 88.5


# ── parse_cost：未知必须是 None，不能是 0 ──────────────────────────────

def test_parse_cost_missing_is_none_not_zero():
    """核心不变量。任何一条变成 0.0 都会让 `cost <= 100` 静默通过。"""
    assert parse_cost(None) is None
    assert parse_cost("") is None
    assert parse_cost("   ") is None
    assert parse_cost("暂无") is None
    assert parse_cost("人均100元") is None      # 不猜数字
    assert parse_cost([]) is None
    assert parse_cost({}) is None


def test_parse_cost_zero_is_unknown():
    assert parse_cost("0.00") is None
    assert parse_cost("0") is None
    assert parse_cost(0) is None
    assert parse_cost(-5) is None


def test_parse_cost_bool_is_not_one():
    """bool 是 int 子类，不挡住的话 True 会变成人均 1 元。"""
    assert parse_cost(True) is None
    assert parse_cost(False) is None


# ── extract_cost：两种字段位置 ────────────────────────────────────────

def test_extract_cost_mcp_shape():
    """高德 MCP：biz_ext 已被拍平，cost 在顶层。这是当前实际路径。"""
    detail = {
        "id": "B0FFJZ65CO",
        "name": "海底捞火锅(顺城购物中心店)",
        "cost": "120.00",
        "rating": "4.7",
        "type": "餐饮服务;中餐厅;火锅店",
    }
    assert extract_cost(detail) == 120.0
    assert "biz_ext" not in detail          # MCP 形状确实没有这一层


def test_extract_cost_rest_shape():
    """高德 REST 原始接口：cost 在 biz_ext 里。兼容兜底。"""
    detail = {"name": "某餐厅", "biz_ext": {"cost": "375.00", "rating": "4.5"}}
    assert extract_cost(detail) == 375.0


def test_extract_cost_top_level_wins():
    """两处都有时以顶层为准（MCP 是当前路径）。"""
    detail = {"cost": "120.00", "biz_ext": {"cost": "999.00"}}
    assert extract_cost(detail) == 120.0


def test_extract_cost_falls_through_to_biz_ext():
    """顶层是空字符串时，不能就此返回 None，要继续查 biz_ext。"""
    detail = {"cost": "", "biz_ext": {"cost": "88.00"}}
    assert extract_cost(detail) == 88.0


def test_extract_cost_missing_everywhere():
    assert extract_cost({"name": "某餐厅"}) is None
    assert extract_cost({"name": "某餐厅", "biz_ext": []}) is None
    assert extract_cost({"cost": ""}) is None
    assert extract_cost(None) is None
    assert extract_cost("not a dict") is None


# ── biz_ext 归一 ──────────────────────────────────────────────────────

def test_biz_ext_empty_list_form():
    """REST 无商业信息时 biz_ext 可能是 []，不归一会让下游 .get() 崩。"""
    assert biz_ext({"biz_ext": []}) == {}


def test_biz_ext_normal():
    assert biz_ext({"biz_ext": {"cost": "375.00"}}) == {"cost": "375.00"}


def test_biz_ext_defensive():
    assert biz_ext(None) == {}
    assert biz_ext({}) == {}
    assert biz_ext("not a dict") == {}
    assert biz_ext({"biz_ext": None}) == {}
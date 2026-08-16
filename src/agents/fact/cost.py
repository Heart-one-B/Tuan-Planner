# agents/fact/cost.py
from __future__ import annotations


def biz_ext(detail: dict | None) -> dict:
    """从 POI 详情里安全取出 biz_ext。

    高德 REST 原始接口在"这个 POI 没有商业信息"时，biz_ext 可能是
    {} 也可能是 []，不统一成 dict 的话下游一个 .get() 就会在 list
    上炸掉——偶发、只在特定 POI 上炸，比稳定崩溃更难查。

    注意：走高德 MCP 时通常**根本没有 biz_ext 这一层**（见
    extract_cost 的说明），这个函数因此主要是兼容兜底。
    """
    if not isinstance(detail, dict):
        return {}
    ext = detail.get("biz_ext")
    return ext if isinstance(ext, dict) else {}


def extract_cost(detail: dict | None) -> float | None:
    """从 POI 详情里取出人均消费，兼容两种字段位置。

    【实测确认的形状差异，这是一个踩过的坑】
    高德 **REST 原始接口** 把人均放在 biz_ext 里：
        {"name": "...", "biz_ext": {"cost": "120.00", "rating": "4.7"}}
    高德 **MCP 服务端** 已经把 biz_ext 拍平到顶层：
        {"name": "...", "cost": "120.00", "rating": "4.7", ...}

    最初只按 REST 形状写 biz_ext(detail).get("cost")，在 MCP 路径下
    恒为 None——50 家餐厅覆盖率 0%，看起来像"高德没数据"，实际是
    我们在错误的层级找字段。这类 bug 不报错、不抛异常，只是安静地
    全都是空值，所以必须靠探针脚本打印真实返回来发现，不能靠推断。

    rating 当初没出这个问题，纯属侥幸——原代码写的是
    `ext.get("rating") or detail.get("rating")`，那个顶层兜底救了它。
    这里两个位置都查，顺序是顶层优先（MCP 是当前实际路径），
    biz_ext 兜底（换 MCP 版本或改直连 REST 时仍然成立）。
    """
    if not isinstance(detail, dict):
        return None
    top = parse_cost(detail.get("cost"))
    if top is not None:
        return top
    return parse_cost(biz_ext(detail).get("cost"))


def parse_cost(raw) -> float | None:
    """把高德的人均消费字段解析成 float。

    返回 None 表示"不知道"，绝不是 0。

    这一条是价格断言正确性的地基：0 的语义是"免费"，缺失的语义是
    "高德没给数据"。把未知折成 0，`cost <= 100` 会**静默通过**，
    实验报告会说"约束满足"，而真实含义是"我们根本不知道价格"。
    与 StreamAccumulator 拒绝在没收到 usage 时伪造全零
    NormalizedUsage 是同一条纪律：伪造的确定性比诚实的未知更糟。

    已知形态（真实 MCP 观测）："120.00" / "105.00" / "66.00" / 缺字段。

    "0.00" 按未知处理——真实数据里没有观测到这个值，保守起见不把它
    当"免费"，因为"免费餐厅"这种正当解释在当前场景下不存在。

    bool 必须在 int 之前挡掉：Python 里 bool 是 int 的子类，
    True 会被 isinstance(raw, int) 放行、float(True) == 1.0，
    静默变成"人均 1 元"。
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None
    if not isinstance(raw, str):
        return None

    s = raw.strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        # "暂无"、"人均100元" 这类非纯数字文本：不做正则抠数字。
        # 抠出来的可能是"100元"里的 100，也可能是"消费50-100"里的 50，
        # 猜错的代价是一个看起来合理、实际错误的价格，比 None 更糟。
        return None
    return v if v > 0 else None
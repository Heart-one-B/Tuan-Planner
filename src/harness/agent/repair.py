# harness/agent/repair.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.snapshot.models import RunSnapshot

ORPHAN_MARKER = "[未执行]"


def _role(msg) -> str | None:
    return msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)


def _tool_call_id_of(msg) -> str | None:
    return msg.get("tool_call_id") if isinstance(msg, dict) else getattr(msg, "tool_call_id", None)


def _tool_call_ids(msg) -> list[str]:
    tcs = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
    ids = []
    for tc in (tcs or []):
        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
        if tc_id is not None:
            ids.append(tc_id)
    return ids


def repair_orphan_tool_calls(messages: list, reason: str) -> int:
    """扫描消息列表,给每个没有配对 tool 消息的 tool_call 合成一条
    tool 消息。CC 的原话是"先认错,再继续":宁可丑陋地塞一条假结果,
    也不能优雅地让整个会话因为 API 协议的死规矩而作废——几乎所有
    OpenAI 兼容 API 都要求"assistant 消息里的每个 tool_call 都必须
    有配对的 tool 结果",少一个就整个请求 400。

    原地修改 messages(往里插入合成消息),返回补了几条。

    通用扫描算法,不针对"调用方明确知道具体缺哪个"做特化——这个
    函数要同时服务两类调用方:①中断/异常场景,调用方其实知道大概
    是哪批调用出的问题,但仍然交给通用扫描处理,不自己维护一份
    "缺口清单"(维护两份账本、可能不一致的教训这个仓库已经付过很多
    次);②repaired_history() 场景,调用方压根不知道缺口在哪,只是
    拿到一份可能来自任意来源的历史,要把它修干净才能送出去。

    插入位置:每个 assistant 消息后面、紧跟的 tool 消息连续区间的
    末尾(不是紧贴 assistant 消息本身)——这样如果同一批 tool_calls
    里有些已经有真实结果、只是缺了其中一个,合成的占位不会插到
    真实结果前面打乱顺序观感(虽然大多数 provider 不要求 tool 消息
    之间严格保序,但没有理由制造不必要的凌乱)。
    """
    live_ids: set[str] = {
        _tool_call_id_of(m) for m in messages
        if _role(m) == "tool" and _tool_call_id_of(m) is not None
    }
    insertions: list[tuple[int, dict]] = []
    for i, m in enumerate(messages):
        if _role(m) != "assistant":
            continue
        missing = [tc_id for tc_id in _tool_call_ids(m) if tc_id not in live_ids]
        if not missing:
            continue
        j = i + 1
        while j < len(messages) and _role(messages[j]) == "tool":
            j += 1
        for tc_id in missing:
            insertions.append((j, {
                "role": "tool", "tool_call_id": tc_id,
                "content": f"{ORPHAN_MARKER} {reason}",
            }))
            live_ids.add(tc_id)

    for idx, msg in sorted(insertions, key=lambda x: x[0], reverse=True):
        messages.insert(idx, msg)
    return len(insertions)


def repaired_history(snapshot: "RunSnapshot") -> list[dict]:
    """resume_history() + 孤儿 tool_use 修复。

    【关键边界,写死在这里】宿主放弃 resume、改用 history 续跑(新开
    一次 run())时必须用这个,不能用 snapshot.resume_history()——两者
    唯一的区别就是要不要修复孤儿 tool_use,选错的后果完全不同:

      resume_history()   给 Agent.resume() 用。挂起状态下消息里
                         **必须**保留孤儿 tool_call(那正是
                         pending_tool_call_id 指向的那个),
                         resume() 靠扫描它来定位"从哪里继续"——
                         修复了反而会让 resume() 找不到挂起点。

      repaired_history()  给"放弃 resume、当作全新历史续跑"的场景
                         用。这条路上孤儿 tool_call 只会造成 API
                         400,没有任何人还需要靠它定位什么,必须修掉。

      混用的后果:用 resume_history() 去做续跑会 400;用
      repaired_history() 去 resume() 会让 resume() 报"找不到
      pending_tool_call_id 对应的消息"而失败——两个方向都是硬报错,
      不是那种会被忽略过去的软错误,但仍然值得在这里把边界说清楚,
      不要指望调用方靠试错发现。

    不作为 RunSnapshot 的方法实现,是刻意的架构选择:RunSnapshot
    (snapshot/models.py)对 harness.agent.* 保持零依赖,这个函数
    放在 agent/repair.py(agent 层依赖 snapshot 层是既有的、单向的
    依赖方向)里,只在类型标注时用 TYPE_CHECKING 引用 RunSnapshot,
    不引入 snapshot → agent 这个反方向的依赖。
    """
    history = snapshot.resume_history()
    repair_orphan_tool_calls(history, reason="审批未完成")
    return history
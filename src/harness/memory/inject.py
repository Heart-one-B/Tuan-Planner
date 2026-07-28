# harness/memory/inject.py
from __future__ import annotations

from harness.memory.instructions import MEMORY_INDEX_MARKER, MemoryConfig
from harness.memory.store import MemoryStore


def build_index_message(store: MemoryStore, config: MemoryConfig) -> dict | None:
    """从记忆文件的 frontmatter 现场生成索引消息(user 角色,独立第二条
    消息注入,不进 system prompt——保护可全局缓存的前缀,用户数据每人
    不同,混进 system prompt 缓存就废了)。

    索引是派生视图,不落盘(刻意偏离 CC 的可维护 MEMORY.md 文件):
    手工维护的索引必然漂移(写了记忆忘更新索引/删了文件没删条目),
    CC 靠 AutoDream 定期修复漂移,我们第一期没有修复机制,就不该引入
    一个已知会烂的账本。派生视图结构上不可能撒谎。代价是 agent 不能
    在索引里写自由格式的组织性注记,等 Phase 7 Dream 落地再评估。

    条目按 updated_at 降序(store.list_all 已排好):双截断砍掉的是
    最旧的条目。时间列用绝对日期——相对时间('3天前')每天自己变,
    打穿 prompt cache;绝对日期只在内容真变时才变。

    没有任何记忆时返回 None:不注入空索引,零记忆用户零感知。
    """
    records = store.list_all()
    if not records:
        return None

    lines = [
        f"{MEMORY_INDEX_MARKER} 以下是你的跨会话记忆索引"
        f"(共 {len(records)} 条;读取全文用 memory_read):",
    ]
    for r in records:
        date = r.updated_at.strftime("%Y-%m-%d")
        desc = r.description or "(无描述)"
        lines.append(f"- {r.name} [{r.type}] ({date}): {desc}")

    # 双截断:行数先行,字节兜底(单一限制不够,CC 实测过 200 行塞
    # 197KB 的病态案例——行数限不住"每行超长")
    truncated = False
    if len(lines) > config.index_max_lines:
        lines = lines[:config.index_max_lines]
        truncated = True
    text = "\n".join(lines)
    if len(text.encode("utf-8")) > config.index_max_bytes:
        raw = text.encode("utf-8")[:config.index_max_bytes]
        text = raw.decode("utf-8", errors="ignore")
        truncated = True
    if truncated:
        text += "\n(索引超长已截断,未显示的均为更早的记忆)"

    return {"role": "user", "content": text}


def strip_old_index_messages(history: list) -> list:
    """从续跑 history 里剥掉旧的索引消息。

    为什么必须剥:快照的 resume_history() 只剥开头的 system 消息,
    索引消息是 user 角色,会原样留在历史里;续跑时 Agent 又注入一条
    新索引,历史里就有两条(其中一条还是过期的)。索引是派生配置不是
    对话数据("指令是配置、对话是数据"——resume_history 剥 system、
    CC compact 后重新加载 CLAUDE.md,同一条原则的第三次应用),
    每次 run 都应该只有当次生成的最新一条。

    只认 MEMORY_INDEX_MARKER 开头的 user 消息,其余(包括恰好在正文
    中间提到这个标记串的消息)一律不动。
    """
    cleaned = []
    for m in history:
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content") or ""
            if isinstance(content, str) and content.startswith(MEMORY_INDEX_MARKER):
                continue
        cleaned.append(m)
    return cleaned
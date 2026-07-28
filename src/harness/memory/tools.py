# harness/memory/tools.py
from __future__ import annotations

from datetime import datetime

from harness.memory.instructions import MemoryConfig
from harness.memory.models import MemoryRecord, VALID_TYPES
from harness.memory.store import MemoryStore
from harness.tools.tool_definition import ToolDefinition

MEMORY_WRITE_TOOL = "memory_write"
MEMORY_READ_TOOL = "memory_read"
MEMORY_DELETE_TOOL = "memory_delete"


def build_memory_tools(
    store: MemoryStore, config: MemoryConfig, include_delete: bool = True,
) -> list[ToolDefinition]:
    """include_delete=False:第二期新增,提取 Agent 装配时用——提取者
    只负责"学到新东西就记下来",没有"发现记忆过时该删"的正当性,
    那是主 Agent 在实际使用记忆时才具备的判断力。这是 Phase 6 权限
    门控做完之前,靠"装配时少注册"实现的权限收窄。"""
    async def memory_write(name: str, type: str, description: str, content: str) -> str:
        if type not in VALID_TYPES:
            return (f"【参数错误,请调整后重试】type 必须是 {list(VALID_TYPES)} 之一,"
                    f"收到: {type}")
        if not name.strip():
            return "【参数错误,请调整后重试】name 不能为空"
        record = MemoryRecord(name=name.strip(), type=type,
                              description=description.strip(), content=content)
        store.write(record)
        return f"已写入记忆 '{record.name}' (type={type})"

    async def memory_read(name: str) -> str:
        record = store.read(name)
        age_days = (datetime.now() - record.updated_at).days
        warning = ""
        if age_days >= config.staleness_days:
            warning = (
                f"[过期警告] 该记忆最后更新于 "
                f"{record.updated_at.strftime('%Y-%m-%d')}(约 {age_days} 天前),"
                f"其中的信息可能已经过时,采纳前请先验证是否仍然准确。\n\n"
            )
        return (f"{warning}--- 记忆 '{record.name}' (type={record.type}) ---\n"
                f"{record.content}")

    async def memory_delete(name: str) -> str:
        deleted = store.delete(name)
        return f"已删除记忆 '{name}'" if deleted else f"记忆 '{name}' 不存在,无需删除"

    tools = [
        ToolDefinition(
            name=MEMORY_WRITE_TOOL,
            description=(
                "写入或更新一条跨会话记忆(同名即覆盖更新)。适用于:用户的"
                "长期偏好(user)、被纠正过的协作方式(feedback)、当前任务的"
                "关键决策与进度(task)、外部信息的位置(reference)。"
                "feedback/task 类的 content 必须包含 Why 和 How to apply。"
                "日期一律写绝对日期。不要写入能从代码/文件直接读到的内容。"
            ),
            parameters={
                "name": {"type": "string",
                         "description": "记忆名(主键,建议下划线风格,如 user_tech_background)"},
                "type": {"type": "string", "enum": list(VALID_TYPES),
                         "description": "user=用户是谁 feedback=怎么协作 "
                                        "task=当前任务动态 reference=外部信息位置"},
                "description": {"type": "string",
                                "description": "一句话描述——它出现在索引里,是未来"
                                               "唤起这条记忆的唯一线索,要写得能唤起回忆"},
                "content": {"type": "string", "description": "记忆正文"},
            },
            required=["name", "type", "description", "content"],
            func=memory_write,
        ),
        ToolDefinition(
            name=MEMORY_READ_TOOL,
            description=(
                "读取一条记忆的完整内容。当对话开头的[记忆索引]里有和当前"
                "任务相关的条目、且索引里的一句话描述不足以支撑判断时调用。"
            ),
            parameters={
                "name": {"type": "string", "description": "索引中列出的记忆名"},
            },
            required=["name"],
            func=memory_read,
        ),
    ]
    if include_delete:
        tools.append(ToolDefinition(
            name=MEMORY_DELETE_TOOL,
            description=(
                "删除一条记忆。当发现某条记忆与现实冲突、或用户明确要求"
                "忘记某事时调用。"
            ),
            parameters={
                "name": {"type": "string", "description": "要删除的记忆名"},
            },
            required=["name"],
            func=memory_delete,
        ))
    return tools
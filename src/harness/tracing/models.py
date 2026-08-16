# harness/tracing/models.py
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolEvent:
    event_id: str
    trace_id: str
    tool_name: str
    args: str
    result: str
    status: str
    duration_ms: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LLMCall:
    event_id: str
    trace_id: str
    prompt_tokens: int
    completion_tokens: int
    token_source: str        # "api_usage" | "estimated" —— 标注数字可信度
    output: str
    has_tool_calls: bool
    duration_ms: int
    reasoning: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Trace:
    trace_id: str
    session_id: str
    user_input: str
    final_reply: str
    total_duration_ms: int
    tool_call_count: int
    llm_call_count: int
    status: str
    parent_trace_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    tool_events: list[ToolEvent] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)


@dataclass
class TraceNode:
    """Span 树的一个节点:一条 trace + 它的全部子 trace。

    【Phase 4 新增】写入端(parent_trace_id)从一开始就完备,读取端此前
    完全不存在——代码里没有任何一处查询过 parent_trace_id。这个类型
    是读取端的载体。

    为什么需要它:提取子 Agent、召回、压缩这些嵌套调用各自是独立的
    trace,靠 parent_trace_id 串成一棵树。没有树视图时,想回答
    "这次 run 一共花了多少 token""那次提取到底看到了什么"这类问题,
    只能手工按 trace_id 一条条捞,实践中等于做不到。
    """
    trace: Trace
    children: list["TraceNode"] = field(default_factory=list)

    def walk(self):
        """深度优先遍历整棵树,先自己再孩子。返回 TraceNode 而不是
        Trace——调用方经常需要知道深度/父子关系,只给 Trace 会丢掉
        树结构信息。"""
        yield self
        for child in self.children:
            yield from child.walk()

    def flatten(self) -> list[Trace]:
        return [node.trace for node in self.walk()]

    def total_tokens(self) -> tuple[int, int]:
        """整棵树的 (prompt_tokens, completion_tokens) 合计。

        这是"一次用户请求到底花了多少钱"的答案——注意它必须是**整棵
        树**的合计,只看顶层 trace 会漏掉提取子 Agent、召回选择器、
        压缩摘要器这些嵌套调用的开销,而那些恰恰是容易被低估的部分。

        token_source 混合了 api_usage 和 estimated 时不做区分,如实
        相加——调用方要区分可以自己走 walk() 逐条看。这里不替它做
        "要不要信任估算值"这个判断。
        """
        prompt = completion = 0
        for node in self.walk():
            for call in node.trace.llm_calls:
                prompt += call.prompt_tokens or 0
                completion += call.completion_tokens or 0
        return prompt, completion

    def all_tool_events(self) -> list[ToolEvent]:
        """整棵树里的全部工具调用,按 trace 的层级顺序。
        审计场景的主力查询:"这次运行到底动了哪些工具、参数是什么、
        结果是什么"。"""
        events = []
        for node in self.walk():
            events.extend(node.trace.tool_events)
        return events

    def find(self, trace_id: str) -> "TraceNode | None":
        for node in self.walk():
            if node.trace.trace_id == trace_id:
                return node
        return None

    def depth(self) -> int:
        """树的最大深度(单节点为 1)。用于快速判断嵌套是否异常——
        正常一次 run 的深度在 1-2 层,深度突然变大通常意味着有意外的
        递归嵌套。"""
        return 1 + max((c.depth() for c in self.children), default=0)
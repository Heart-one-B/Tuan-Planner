# harness/agent/dag_scheduler.py
import asyncio
import logging
import time
from typing import Callable, Protocol

from harness.agent.result import AgentResult

logger = logging.getLogger(__name__)


# ── 失败策略(gate)────────────────────────────────────────────────────────
# 现在只有一个,故合并在调度器文件里。等策略长成一族(strict/lenient/per_node)
# 再抽到独立的 dag_policy.py。
def default_gate(upstream_results: list[AgentResult]) -> bool:
    """上游结果够不够让下游放行执行?

    默认最严格:所有上游都必须 ok。partial/empty/error 一律视为不放行,
    下游会被跳过(并连锁跳过更下游)。
    以后想接受 partial,只改这一个函数,调度器主循环不动。
    """
    return all(r.status == "ok" for r in upstream_results)


# ── 执行器接口(鸭子类型)───────────────────────────────────────────────────
class StepExecutor(Protocol):
    """任何长成这个形状的东西都能当节点执行器。
    调度器只认这个接口,不认具体实现(函数 / StructuredAgent / 远程调用皆可)。
    """
    async def execute(self, task: str, context: dict[int, AgentResult]) -> AgentResult:
        ...


# ── 调度器 ────────────────────────────────────────────────────────────────
class DagScheduler:
    """DAG 调度器:Plan-and-Execute 的发动机。

    职责仅三件,且对"任务是什么"一无所知:
      1. 读 DAG,按 depends_on 拓扑分波次
      2. 波次内并行(gather),波次间串行
      3. 维护结果表;按 gate 决定下游放行/跳过;把上游结果透传给下游

    不解读任何 data,不假设步骤类型,不预设并行/串行——并行性纯由 depends_on 推导。
    全独立的图 → 一个波次全并行;全串行的图 → 每波一个节点;混合 → 自然分波。
    """

    def __init__(
        self,
        executor: StepExecutor,
        gate: Callable[[list[AgentResult]], bool] = default_gate,
        tracer=None,
    ):
        self.executor = executor
        self.gate = gate           # 失败策略注入,默认"只认全 ok"
        self.tracer = tracer

    async def run(self, steps: list[dict], trace_id: str = None) -> dict[int, AgentResult]:
        """执行整张 DAG,返回 {step_id: AgentResult} 结果表。"""
        nodes = {s["id"]: s for s in steps}
        results: dict[int, AgentResult] = {}
        done: set[int] = set()

        wave = 0
        while len(done) < len(nodes):
            # ① 算当前波次:依赖全完成、自己还没做的节点
            ready = [
                node for nid, node in nodes.items()
                if nid not in done and all(dep in done for dep in node["depends_on"])
            ]

            # 死锁护栏:没节点能跑但又没做完 → 环依赖或非法依赖 id
            if not ready:
                unfinished = [nid for nid in nodes if nid not in done]
                logger.error(f"[DagScheduler] 死锁,未完成={unfinished}(疑似循环依赖)")
                for nid in unfinished:
                    results[nid] = AgentResult(
                        status="error",
                        summary="依赖无法满足(循环依赖或非法依赖id)",
                        data={},
                    )
                    done.add(nid)
                break

            wave += 1

            # ② 用 gate 决定每个 ready 节点:真执行,还是因上游不达标而跳过
            to_run = []
            for node in ready:
                upstream = [results[dep] for dep in node["depends_on"]]
                if node["depends_on"] and not self.gate(upstream):
                    bad = [dep for dep in node["depends_on"] if results[dep].status != "ok"]
                    results[node["id"]] = AgentResult(
                        status="error",
                        summary=f"上游步骤 {bad} 未达放行标准,本步跳过",
                        data={},
                    )
                    done.add(node["id"])
                    logger.warning(f"[DagScheduler] 节点{node['id']} 跳过(上游 {bad} 不达标)")
                else:
                    to_run.append(node)

            logger.info(f"[DagScheduler] 波次{wave}:并行 {[n['id'] for n in to_run]}")
            if not to_run:
                continue

            # ③ 波次内并行执行
            async def run_one(node):
                ctx = {dep: results[dep] for dep in node["depends_on"]}
                start = time.time()
                try:
                    res = await self.executor.execute(node["task"], ctx)
                except Exception as e:
                    # 执行器理应自己兜住异常;这里是最后防线,绝不让一个节点炸全图
                    res = AgentResult(status="error", summary=f"执行器异常: {e}", data={})
                    logger.error(f"[DagScheduler] 节点{node['id']} 执行器抛异常: {e}", exc_info=True)

                duration = int((time.time() - start) * 1000)
                if self.tracer and trace_id:
                    self.tracer.record_tool_event(
                        trace_id=trace_id,
                        tool_name=f"dag_step_{node['id']}",
                        args={"task": node["task"]},
                        result=res.summary,
                        duration_ms=duration,
                        status=res.status,
                    )
                return node["id"], res

            wave_results = await asyncio.gather(*[run_one(n) for n in to_run])

            # ④ 收果,标记完成 → 下一轮 while 自动解锁下游
            for nid, res in wave_results:
                results[nid] = res
                done.add(nid)

        return results
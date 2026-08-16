# tests/real_interrupt_2_long_task_kill_me.py
"""
持久化中断恢复测试 —— 第 2 步:故意跑一个较长的真实任务,你需要在它
跑完之前手动杀掉这个进程(Ctrl+C,或从任务管理器/活动监视器强制结束)。

【设计说明,务必先读】
按当前 harness 的设计(拍板一:快照只在 run() 正常结束时自动落盘,
不是每轮都存),杀掉本脚本这个进程后,**这次运行产生的任何进度都不会
被保存**——这不是 bug,是设计取舍(每轮都拍需要额外配置,目前没做,
Phase 7 才会按需回来加)。

所以本测试真正验证的不是"恢复被中断的进度",而是:杀掉一个"正在运行、
且这次运行从未成功保存过快照"的进程,**不会损坏第 1 步已经保存下来的
那份旧快照**。这正是原子写(os.replace)机制存在的意义——之前只在代码
里用 monkeypatch 模拟过"写到一半崩溃",这是第一次用真实的操作系统级
kill 信号去验证它。

跑法:
  1. 先确认已经跑过 real_interrupt_1_baseline.py
  2. 运行本脚本
  3. 看到"第 N 轮查询"的输出后,任意时刻按 Ctrl+C(不要等它自然跑完)
  4. 运行第 3 步: python -m tests.real_interrupt_3_verify_resume

运行:python -m tests.real_interrupt_2_long_task_kill_me
"""
import asyncio
from pathlib import Path

from harness.agent.agent import Agent
from harness.agent.termination import AnswerTermination
from harness.snapshot import FileSnapshotStore
from harness.tools.tool_definition import ToolDefinition
from harness.tools.tool_executor import ToolExecutor

from tests.real_smoke_llm import get_real_llm_client

SNAPSHOT_DIR = Path("data/real_interrupt_test_snapshots")
SESSION_ID = "real-interrupt-test"   # 故意和第1步用同一个 session_id


async def slow_lookup(q: str) -> str:
    print(f"  [工具执行] 正在查询 {q} ……")
    return f"关于'{q}'的查询结果:数据点{q}已记录。"


async def main() -> None:
    llm = get_real_llm_client()
    if llm is None:
        return

    store = FileSnapshotStore(SNAPSHOT_DIR)
    executor = ToolExecutor()
    executor.register(ToolDefinition(
        name="slow_lookup", description="查询一个数据点",
        parameters={"q": {"type": "string"}}, required=["q"], func=slow_lookup,
    ))
    agent = Agent(llm, executor, "你是助手,严格按步骤执行,每次只调用一个工具、"
                  "等结果返回后再调用下一个。",
                  AnswerTermination(), snapshot_store=store)
    task = (
        "请依次调用 slow_lookup 查询以下8个数据点,每次只查一个:"
        "A、B、C、D、E、F、G、H。全部完成后回复'完成'。"
    )
    print("开始运行较长任务,现在可以随时按 Ctrl+C 中断……\n")
    outcome = await agent.run(task, session_id=SESSION_ID)

    print("\n(如果看到这一行,说明任务在你中断之前就跑完了——这次没能测到"
         "'中断中的进程'这个场景,可以再跑一次、更早按 Ctrl+C。)")
    print("状态:", outcome.status)


if __name__ == "__main__":
    asyncio.run(main())
# tests/real_interrupt_1_baseline.py
"""
持久化中断恢复测试 —— 第 1 步:跑一个正常完成的基线任务,建立一份好快照。

三步流程说明(为什么要拆成三个脚本,不能塞进一个):
  第1步(本脚本)  跑一个短任务,正常结束,自动落一份好快照。
  第2步          跑一个较长的任务,你需要在它跑完之前手动杀掉进程
                 (Ctrl+C 或任务管理器强制结束)。
  第3步          检查快照文件有没有被中断损坏,并用它做一次真实续跑。

先运行本脚本,确认看到"baseline 快照已保存",再进行第 2 步。
配置复用 real_smoke_llm.py 里已经填好的 API_KEY/BASE_URL/MODEL_NAME,
不用再填一遍。

运行:python -m tests.real_interrupt_1_baseline
"""
import asyncio
from pathlib import Path

from harness.agent.agent import Agent
from harness.agent.termination import AnswerTermination
from harness.snapshot import FileSnapshotStore
from harness.tools.tool_executor import ToolExecutor

from tests.real_smoke_llm import get_real_llm_client

SNAPSHOT_DIR = Path("data/real_interrupt_test_snapshots")
SESSION_ID = "real-interrupt-test"


async def main() -> None:
    llm = get_real_llm_client()
    if llm is None:
        return

    store = FileSnapshotStore(SNAPSHOT_DIR)
    agent = Agent(llm, ToolExecutor(), "你是助手,回答简洁。",
                  AnswerTermination(), snapshot_store=store)
    outcome = await agent.run("用一句话介绍一下你自己。", session_id=SESSION_ID)
    print("baseline 任务完成,状态:", outcome.status)
    print("回复:", outcome.final_text)

    snap = store.load_latest(SESSION_ID)
    print(f"\n✅ baseline 快照已保存: task={snap.task!r} 消息数={len(snap.messages)}")
    print(f"   文件位置: {SNAPSHOT_DIR / SESSION_ID / 'latest.json'}")
    print("\n现在运行第 2 步: python -m tests.real_interrupt_2_long_task_kill_me")
    print("并在它运行途中手动杀掉进程(Ctrl+C 或任务管理器强制结束)。")


if __name__ == "__main__":
    asyncio.run(main())
# tests/real_interrupt_3_verify_resume.py
"""
持久化中断恢复测试 —— 第 3 步:验证被强制中断后,快照文件是否完好,
并用它做一次真实续跑。

先决条件:已经跑过第 1 步(留下一份好快照),并在第 2 步跑到一半时
被你强制杀掉了进程。

运行:python -m tests.real_interrupt_3_verify_resume
"""
import asyncio
import json
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

    latest_path = SNAPSHOT_DIR / SESSION_ID / "latest.json"
    print(f"检查文件: {latest_path}")
    if not latest_path.is_file():
        print("❌ 文件不存在——请先跑第 1 步(real_interrupt_1_baseline.py)。")
        return

    raw = latest_path.read_text(encoding="utf-8")
    try:
        json.loads(raw)
        print(f"✅ 文件是合法 JSON,共 {len(raw)} 字节,没有被中断损坏成半成品。")
    except json.JSONDecodeError as e:
        print(f"❌ 文件已损坏,不是合法 JSON: {e}")
        print("   这如果真的发生,说明原子写机制在你的操作系统/文件系统上")
        print("   没有起到作用,需要认真排查——这是不该发生的情况,请把这个")
        print("   结果和你的操作系统/文件系统信息一起报告。")
        return

    store = FileSnapshotStore(SNAPSHOT_DIR)
    snap = store.load_latest(SESSION_ID)
    print(f"\ntask 字段: {snap.task!r}")

    if "介绍一下你自己" in snap.task:
        print("✅ 符合预期:这是第 1 步(baseline)留下的快照。说明第 2 步被")
        print("   杀掉后,从未成功写入过 latest.json——旧的好快照完好保留,")
        print("   原子写在真实操作系统级 kill 下生效了。")
    elif "slow_lookup" in snap.task:
        print("⚠️  这是第 2 步自己留下的快照——说明第 2 步在你按 Ctrl+C 之前")
        print("   已经跑完了一整轮(run 结束才会自动保存),没有真的测到")
        print("   '中断中的进程'这个场景。可以再跑一次第 2 步,更早按 Ctrl+C。")
    else:
        print(f"⚠️  task 字段既不匹配第1步也不匹配第2步的任务文本,请人工核对。")

    # 无论是哪种情况，拿到的都是一份完整可用的快照，做一次真实续跑验证可用性
    agent2 = Agent(llm, ToolExecutor(), "你是助手,回答简洁。", AnswerTermination())
    outcome = await agent2.run(
        "简单确认一下:你还记得我们之前聊到哪了吗?",
        history=snap.resume_history(), session_id=SESSION_ID + "-verify",
    )
    print("\n续跑状态:", outcome.status)
    print("续跑回复:", outcome.final_text)
    if outcome.status == "completed":
        print("\n✅ 快照在真实中断后依然可用,续跑成功。")
    else:
        print("\n❌ 续跑失败,见上方报错。")


if __name__ == "__main__":
    asyncio.run(main())
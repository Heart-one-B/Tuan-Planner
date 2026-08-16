# evals/merge.py
"""把分层跑出的多份结果合并成一份 baseline。

【为什么需要它】
`--layer all` 一次跑完最干净，但 68 条要半小时。分层跑（先 L1 便宜
的探路、再 L2、最后 L3）在迭代期更实际——代价是结果散在三个文件里，
而 diff_baseline 只读一份。

【合并的前提：三份必须可比】
同一个模型、同一个 runner 版本、同一份 cases.yaml。任何一项不同，
合出来的基线就是三个不同实验的拼盘，以后 diff 出变化时无法归因
——你分不清是代码变了还是当初的基线本来就不一致。

所以本脚本**强制校验 cases_version**，不一致直接拒绝合并而不是
警告后继续。模型和 runner 版本没有记录在结果里（这是个缺口，
见下方 TODO），只能靠人确认——脚本会把每份的时间戳打出来提醒。

【重复 id 的处理】
后指定的文件覆盖先指定的。理由：迭代时你会重跑某一层修完的用例，
命令行里把新的放后面，语义直观。覆盖时会打印出来，不静默。

用法：
    python evals/merge.py \\
        data/experiments/eval_A.json \\
        data/experiments/eval_B.json \\
        data/experiments/eval_C.json \\
        -o data/experiments/baseline.json

TODO：结果 JSON 里应该记模型名和 runner 版本。现在没有，合并的
可比性只能靠人确认——这是一个真实的缺口，不是"以后再说"的优化。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)


def stat(rows: list[dict]) -> dict:
    return {"n": len(rows),
            "passed": sum(1 for r in rows if r["verdict"] == "passed"),
            "failed": sum(1 for r in rows if r["verdict"] == "failed"),
            "error": sum(1 for r in rows if r["verdict"] == "error")}


def summarize(records: list[dict]) -> dict:
    out = {"overall": stat(records), "by_layer": {}, "by_origin": {}}
    for layer in sorted({r["layer"] for r in records}):
        out["by_layer"][layer] = stat([r for r in records if r["layer"] == layer])
    for origin in sorted({r.get("origin", "constructed") for r in records}):
        out["by_origin"][origin] = stat(
            [r for r in records if r.get("origin", "constructed") == origin])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="要合并的结果 JSON，后面的覆盖前面的")
    ap.add_argument("-o", "--out", default="data/experiments/baseline.json")
    ap.add_argument("--force", action="store_true",
                    help="cases_version 不一致时仍然合并（不推荐）")
    args = ap.parse_args()

    loaded = []
    for f in args.files:
        p = Path(f)
        if not p.is_file():
            print(f"❌ 找不到 {p}")
            return 1
        loaded.append((p, json.loads(p.read_text(encoding="utf-8"))))

    print("待合并：")
    versions = set()
    for p, d in loaded:
        n = len(d.get("records") or [])
        versions.add(d.get("cases_version"))
        print(f"  {p.name}  layers={d.get('layers')}  记录={n}  "
              f"cases_v={d.get('cases_version')}  memory={d.get('memory')}  "
              f"time={d.get('timestamp', '')[:19]}")

    if len(versions) > 1 and not args.force:
        print(f"\n❌ cases_version 不一致：{versions}")
        print("   合并不同版本用例的结果，得到的是三个不同实验的拼盘——")
        print("   以后 diff 出变化时无法归因（是代码变了，还是基线本来就不齐）。")
        print("   请用同一版用例重跑，或显式 --force 并在 changelog 里记一笔。")
        return 1

    memories = {d.get("memory") for _p, d in loaded}
    if len(memories) > 1:
        print(f"\n⚠️ memory 开关不一致：{memories}")
        print("   记忆会让同一输入的结果依赖此前跑过什么，混合统计的分数含义不明。")

    merged: dict[str, dict] = {}
    for p, d in loaded:
        for r in d.get("records") or []:
            cid = r["id"]
            if cid in merged:
                print(f"  ↻ {cid}: {merged[cid]['verdict']} → {r['verdict']}"
                      f"（被 {p.name} 覆盖）")
            merged[cid] = r

    records = sorted(merged.values(), key=lambda r: r["id"])
    s = summarize(records)
    o = s["overall"]

    print("\n" + "=" * 70)
    print("合并结果")
    print("=" * 70)
    print(f"  总计 {o['n']}  ✅{o['passed']}  ❌{o['failed']}  💥{o['error']}")
    for name, st in s["by_layer"].items():
        print(f"    {name:<14} {st['passed']}/{st['n']}")
    print("\n  按来源：")
    for name, st in s["by_origin"].items():
        print(f"    {name:<14} {st['passed']}/{st['n']}")

    if o["failed"]:
        print("\n  失败的用例（基线里带着已知失败是正常的——"
              "哨兵就该是失败的）：")
        for r in records:
            if r["verdict"] == "failed":
                keys = [c["key"] for c in (r.get("checks") or [])
                        if not c["passed"]]
                print(f"    ❌ {r['id']}  {keys}")
    if o["error"]:
        print("\n  ⚠️ 含 error 的用例——基线里不该有 error"
              "（那意味着这几条从来没被真正测过）：")
        for r in records:
            if r["verdict"] == "error":
                print(f"    💥 {r['id']}  {str(r.get('error'))[:100]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "merged_from": [str(p) for p, _d in loaded],
        "cases_version": next(iter(versions)),
        "layers": sorted({r["layer"] for r in records}),
        "memory": next(iter(memories)),
        "summary": s, "records": records,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n  → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
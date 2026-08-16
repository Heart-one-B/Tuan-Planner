# tests/conftest.py
"""【Phase 4 之后重写】

改造前这个 fixture 是个**补丁**:它伸手进 tracer 的模块级私有字典
(`tracer._active.clear()` / `tracer._storage = None`)手工清场,因为
除此之外没有别的办法隔离测试。当时的注释里我自己写着"这正是进度表
Phase 4 记的账:tracer 去全局单例...测试靠 fixture 手动清场打补丁"。

现在那笔账还掉了,补丁也就不需要了:每个测试拿一个全新的 Tracer 实例,
通过 use_tracer() 作用域生效,测试之间天然不共享任何状态。不再有
"某个测试忘了清场污染下一个"这种可能——因为压根没有需要清的共享物。

保留 default_tracer().reset() 是给"没有经过 use_tracer 的代码路径"
兜底(比如某些直接调模块级函数、又不在 fixture 作用域内的边界情况),
属于纵深防御,不是主要机制。
"""
import pytest

from harness.tracing import Tracer, default_tracer, use_tracer


@pytest.fixture(autouse=True)
def isolated_tracer():
    tracer = Tracer()
    with use_tracer(tracer):
        yield tracer
    default_tracer().reset()

# tests/conftest.py
"""tracer 是模块级全局单例(harness/tracing/tracer.py 的 _active/_storage),
没有暴露 reset() —— 这正是进度表 Phase 4 记的账:"tracer 去全局单例...
测试靠 fixture 手动清场打补丁"。这里就是那个补丁,不改 tracer.py 本身。"""
import pytest

from harness.tracing import tracer


@pytest.fixture(autouse=True)
def _reset_tracer_state():
    tracer._active.clear()
    tracer._start_times.clear()
    tracer._storage = None
    yield
    tracer._active.clear()
    tracer._start_times.clear()
    tracer._storage = None

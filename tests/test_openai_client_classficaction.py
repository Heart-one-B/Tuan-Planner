# tests/test_openai_client_classification.py
"""
验证 OpenAIClient._is_retryable 的错误分类,用真实的 openai SDK 异常类
构造(不是猜的签名,是先在沙箱里查过 openai==2.44.0 的真实构造函数签名)。

不发真实网络请求,只测试"给定这类异常,该不该重试"这一个纯函数判断。
"""
from __future__ import annotations

import sys

import httpx
import openai

from harness.llm.openai_client import _is_retryable


def _make_status_error(cls, status_code: int):
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request, json={"error": {"message": "boom"}})
    return cls("boom", response=response, body=None)


def _make_connection_error():
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    return openai.APIConnectionError(request=request)


def _make_timeout_error():
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    return openai.APITimeoutError(request=request)


def test_transient_errors_are_retryable():
    cases = [
        _make_connection_error(),
        _make_timeout_error(),
        _make_status_error(openai.RateLimitError, 429),
        _make_status_error(openai.InternalServerError, 500),
        _make_status_error(openai.InternalServerError, 503),
    ]
    for exc in cases:
        assert _is_retryable(exc), f"{type(exc).__name__} 应判定为可重试,但结果是不可重试"
    print("✅ test_transient_errors_are_retryable 通过"
          f"(覆盖 {len(cases)} 类瞬态错误)")


def test_permanent_errors_are_not_retryable():
    cases = [
        _make_status_error(openai.BadRequestError, 400),
        _make_status_error(openai.AuthenticationError, 401),
        _make_status_error(openai.PermissionDeniedError, 403),
        _make_status_error(openai.NotFoundError, 404),
    ]
    for exc in cases:
        assert not _is_retryable(exc), f"{type(exc).__name__} 应判定为不可重试(重试无用),但结果是可重试"
    print("✅ test_permanent_errors_are_not_retryable 通过"
          f"(覆盖 {len(cases)} 类永久性错误,验证不会浪费重试预算)")


def test_status_code_529_overloaded_is_retryable():
    # 529 不是 openai SDK 原生定义的异常类,但部分兼容端点(如某些
    # Anthropic-compatible 网关)会返回这个状态码,走 status_code 判断分支
    exc = _make_status_error(openai.APIStatusError, 529)
    assert _is_retryable(exc), "529(overloaded)应判定为可重试"
    print("✅ test_status_code_529_overloaded_is_retryable 通过")


def test_unrecognized_generic_error_not_retryable():
    # 普通异常(不是任何 openai 异常类型、也没有 status_code 属性)
    # 应该保守地判定为不可重试,而不是默认重试
    assert not _is_retryable(ValueError("some unrelated bug")), \
        "未知异常类型应保守判定为不可重试,而不是默认重试(默认重试会掩盖真实 bug)"
    print("✅ test_unrecognized_generic_error_not_retryable 通过")


def main():
    tests = [
        test_transient_errors_are_retryable,
        test_permanent_errors_are_not_retryable,
        test_status_code_529_overloaded_is_retryable,
        test_unrecognized_generic_error_not_retryable,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"❌ {t.__name__} 失败: {e}")
        except Exception as e:
            failed += 1
            print(f"💥 {t.__name__} 抛出意外异常: {type(e).__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"总计 {len(tests)} 项, 失败 {failed} 项")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
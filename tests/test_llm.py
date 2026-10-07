"""
Tests for llm.py: error classification, API-key stripping, retry rules, structured output.
No real CLI call is made (subprocess is mocked).   Run:  python tests/test_llm.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import llm  # noqa: E402


class FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def _patch(run_fn):
    llm.find_cli = lambda: "claude"
    llm.RETRY_DELAYS_S = (0, 0, 0)
    subprocess_run = subprocess.run
    subprocess.run = run_fn
    return subprocess_run


def _ok(structured=None, text="hello"):
    body = {"type": "result", "subtype": "success", "is_error": False, "result": text}
    if structured is not None:
        body["structured_output"] = structured
    return FakeProc(json.dumps(body))


def test_classification():
    assert llm.classify("Claude AI usage limit reached|1791000000").status_code == 402
    assert llm.classify("Not logged in · Please run /login").status_code == 401
    assert llm.classify("API Error: 529 overloaded").status_code == 529
    assert llm.classify("something odd").status_code == 502
    assert llm.classify("usage limit reached").fatal and llm.classify("not logged in").fatal
    assert not llm.classify("overloaded").fatal


def test_api_key_is_stripped_and_prompt_goes_via_stdin():
    seen = {}

    def run(cmd, **kw):
        seen["env"], seen["input"], seen["cmd"], seen["cwd"] = kw["env"], kw["input"], cmd, kw["cwd"]
        return _ok({"x": 1})
    orig = _patch(run)
    os.environ["ANTHROPIC_API_KEY"] = "sk-should-not-be-sent"
    try:
        result, by = llm.call_model("score", "PROMPT", schema={"type": "object"}, system="SYS")
    finally:
        subprocess.run = orig
        del os.environ["ANTHROPIC_API_KEY"]
    assert result == {"x": 1} and by.startswith("claude_cli/")
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert seen["input"] == "PROMPT" and "--bare" not in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--system-prompt") + 1] == "SYS"
    assert Path(seen["cwd"]) != ROOT, "runs in an empty temp dir, not the repo"


def test_usage_limit_is_fatal_and_not_retried():
    calls = []

    def run(cmd, **kw):
        calls.append(1)
        return FakeProc(json.dumps({"is_error": True, "result": "Claude AI usage limit reached|1791000000"}), "", 1)
    orig = _patch(run)
    try:
        try:
            llm.call_model("write", "p")
            raise AssertionError("should raise")
        except llm.CliError as e:
            assert e.status_code == 402 and e.fatal
    finally:
        subprocess.run = orig
    assert len(calls) == 1


def test_transient_errors_are_retried_then_raised():
    calls = []

    def run(cmd, **kw):
        calls.append(1)
        return FakeProc("", "Error: overloaded", 1)
    orig = _patch(run)
    try:
        try:
            llm.call_model("write", "p")
            raise AssertionError("should raise")
        except llm.CliError as e:
            assert e.status_code == 529 and not e.fatal
    finally:
        subprocess.run = orig
    assert len(calls) == llm.TRANSIENT_ATTEMPTS


def test_recovers_after_one_transient_failure():
    calls = []

    def run(cmd, **kw):
        calls.append(1)
        return FakeProc("", "overloaded", 1) if len(calls) == 1 else _ok(text="fine")
    orig = _patch(run)
    try:
        text, _ = llm.call_model("write", "p")
    finally:
        subprocess.run = orig
    assert text == "fine" and len(calls) == 2


def test_missing_structured_output_is_a_retryable_error():
    orig = _patch(lambda cmd, **kw: _ok(structured=None))
    try:
        try:
            llm.call_model("score", "p", schema={"type": "object"})
            raise AssertionError("should raise")
        except llm.CliError as e:
            assert e.status_code == 502
    finally:
        subprocess.run = orig


def test_timeout_is_transient():
    def run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)
    orig = _patch(run)
    try:
        try:
            llm.call_model("write", "p")
            raise AssertionError("should raise")
        except llm.CliError as e:
            assert e.status_code == 504 and not e.fatal
    finally:
        subprocess.run = orig


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:           # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)

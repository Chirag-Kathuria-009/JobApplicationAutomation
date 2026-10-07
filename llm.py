"""
job_pipeline/llm.py
===================
The ONLY place the pipeline talks to a model. Every call goes through the Claude
Code CLI in headless mode (`claude -p`) on the owner's Claude subscription, so no
Anthropic API key, SDK or API billing is involved.

    call_model("score", prompt, schema=..., system=...) -> (parsed_json, "claude_cli/sonnet")
    call_model("write", prompt, system=...)             -> (text,        "claude_cli/sonnet")

Which model each task uses lives in models.json. A second provider can be added
later as another function + a "provider" value; nothing else in the repo imports
a provider SDK.

Billing safety: ANTHROPIC_API_KEY is removed from the environment of the
subprocess, because with a key present Claude Code bills the API instead of the
subscription. `--bare` is never used (it ignores the subscription login).

Errors are raised as CliError(status_code, message):
    401  login problem          - fatal   (run `claude` once and log in again)
    402  usage limit reached    - fatal   (resets later; the pipeline stops and retries next run)
    404  CLI not found          - fatal
    502 / 504 / 529             - transient: retried here, then raised
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
MODELS_PATH = BASE_DIR / "models.json"
DEFAULT_MODELS = {
    "score": {"provider": "claude_cli", "model": "sonnet"},
    "write": {"provider": "claude_cli", "model": "sonnet"},
}
DEFAULT_TIMEOUT_S = 300
TRANSIENT_ATTEMPTS = 3
RETRY_DELAYS_S = (0, 8, 20)
GENERIC_SYSTEM_PROMPT = "You are a precise assistant. Follow the instructions in the user message exactly."

LIMIT_RE = re.compile(r"usage limit|limit reached|rate limit|quota|resets? (at|in)|too many requests", re.I)
LOGIN_RE = re.compile(r"not logged in|please run /login|/login|authentication|oauth|unauthori[sz]ed|"
                      r"invalid api key|expired|api error: 401", re.I)
OVERLOAD_RE = re.compile(r"overloaded|503|529|temporarily|try again", re.I)


class CliError(Exception):
    """A failed model call. `fatal` means retrying cannot help until a human acts."""

    def __init__(self, status_code: int, message: str, fatal: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.fatal = fatal


def classify(text: str, returncode: int | None = None) -> CliError:
    text = (text or "").strip()
    if LIMIT_RE.search(text):
        return CliError(402, f"usage limit: {text[:200]}", fatal=True)
    if LOGIN_RE.search(text):
        return CliError(401, f"login problem (run `claude` and log in again): {text[:200]}", fatal=True)
    if OVERLOAD_RE.search(text):
        return CliError(529, text[:200])
    return CliError(502, f"CLI failed (exit {returncode}): {text[:200]}")


# --------------------------------------------------------------------------- #
# Locating the CLI
# --------------------------------------------------------------------------- #
def find_cli() -> str:
    """CLAUDE_CLI env var, then PATH, then ~/.local/bin, then the newest VS Code extension copy."""
    explicit = os.getenv("CLAUDE_CLI")
    if explicit:
        if Path(explicit).exists():
            return explicit
        raise CliError(404, f"CLAUDE_CLI points to a missing file: {explicit}", fatal=True)
    for name in ("claude.exe", "claude"):
        found = shutil.which(name)
        if found:
            return found
    local = Path.home() / ".local" / "bin" / "claude.exe"
    if local.exists():
        return str(local)
    ext = sorted((Path.home() / ".vscode" / "extensions").glob(
        "anthropic.claude-code-*/resources/native-binary/claude.exe"),
        key=lambda p: [int(x) if x.isdigit() else 0 for x in re.split(r"[.-]", p.parent.parent.parent.name)])
    if ext:
        return str(ext[-1])      # works, but the folder name changes with every extension update
    raise CliError(404, "Claude Code CLI not found. Install it (https://claude.ai/install.ps1), run `claude` "
                        "once to log in, or set CLAUDE_CLI in .env to the full path of claude.exe.", fatal=True)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
def _task_config(task: str) -> dict:
    cfg = dict(DEFAULT_MODELS)
    if MODELS_PATH.exists():
        cfg.update(json.loads(MODELS_PATH.read_text(encoding="utf-8")))
    if task not in cfg:
        raise KeyError(f"models.json has no entry for task '{task}'")
    return cfg[task]


def _claude_cli(prompt: str, schema: dict | None, system: str | None, model: str, timeout: int):
    """One `claude -p` call. Returns the parsed structured_output (if a schema was given) or the text."""
    cmd = [find_cli(), "-p", "--output-format", "json", "--model", model,
           "--system-prompt", system or GENERIC_SYSTEM_PROMPT,   # replaces the ~7k-token default prompt
           "--no-session-persistence", "--disable-slash-commands", "--strict-mcp-config",
           "--setting-sources", "", "--tools", ""]
    if schema is not None:
        cmd += ["--json-schema", json.dumps(schema)]
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}     # never bill the API
    try:
        with tempfile.TemporaryDirectory() as empty_dir:      # no CLAUDE.md / repo files loaded as context
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                  cwd=empty_dir, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise CliError(504, f"timeout after {timeout}s")

    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise classify(proc.stderr or proc.stdout, proc.returncode)
    if proc.returncode != 0 or out.get("is_error"):
        raise classify(f"{out.get('result', '')} {proc.stderr} {out.get('api_error_status', '')}", proc.returncode)
    if schema is not None:
        structured = out.get("structured_output")
        if not isinstance(structured, dict):
            raise CliError(502, "reply had no valid structured_output")
        return structured
    text = (out.get("result") or "").strip()
    if not text:
        raise CliError(502, "empty reply")
    return text


_PROVIDERS = {"claude_cli": _claude_cli}


def call_model(task: str, prompt: str, schema: dict | None = None, system: str | None = None):
    """Run `prompt` for `task` ("score" | "write"). Returns (result, "provider/model").

    Transient failures (overload, timeout, bad output) are retried a few times;
    login and usage-limit problems are raised immediately as fatal CliError.
    """
    cfg = _task_config(task)
    provider = _PROVIDERS[cfg["provider"]]
    last: CliError | None = None
    for attempt in range(TRANSIENT_ATTEMPTS):
        time.sleep(RETRY_DELAYS_S[attempt])
        try:
            result = provider(prompt, schema, system, cfg["model"], cfg.get("timeout", DEFAULT_TIMEOUT_S))
            return result, f"{cfg['provider']}/{cfg['model']}"
        except CliError as e:
            if e.fatal:
                raise
            last = e
    raise last


def preflight() -> None:
    """Cheap check before a run: the CLI exists, is logged in, and the plan has capacity.
    Raises CliError (fatal) so the run can stop before touching any job."""
    call_model("score", "Reply with ok.", schema={"type": "object", "properties": {"ok": {"type": "boolean"}},
                                                   "required": ["ok"]})


def warn_if_api_key_set() -> bool:
    """True if ANTHROPIC_API_KEY is in the parent environment (it is stripped for every call)."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))

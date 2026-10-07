"""
Failure-recovery and duplicate-detection tests (stubbed model calls, temp directories).

Run:  python tests/test_recovery.py        (or: python -m pytest tests, if pytest is installed)
Needs BASE_RESUME/base.json for the resume steps; no API key or network.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cover_letter  # noqa: E402
import dedupe  # noqa: E402
import llm  # noqa: E402
import matcher  # noqa: E402
import resume  # noqa: E402
import run_pipeline as rp  # noqa: E402
import scraper  # noqa: E402

LONG_JD = ("Your Role:\n" + "Build and operate data pipelines with Python and SQL for our analytics platform. " * 12
           + "\nYour Profile:\nSolid SQL and Python skills and experience with orchestration. " * 3)


class Env:
    """Isolated pipeline environment with counters on every model/file step."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "profile_de.md").write_text("de")
        rp.PROFILE_PATHS = {"DE": self.tmp / "profile_de.md"}
        rp.PROFILE_PATH = self.tmp / "none.md"
        rp.LOG_PATH = self.tmp / "tracker.json"
        rp.LOCK_PATH = self.tmp / ".lock"
        rp.LAST_RUN_PATH = self.tmp / "last_run.json"
        rp.OUTPUT_DIR = self.tmp / "applications"
        rp.NEAR_THRESHOLD_LOG = self.tmp / "near.jsonl"
        self.jobs = []
        self.calls = {"score": 0, "cover": 0, "resume": 0}
        self.score = 0.9
        self.score_failed = False
        self.score_error = None
        self.cover_error = None
        self.resume_error = None
        self.crash_in_resume = False
        llm.preflight = lambda: None
        scraper.load_startup_sources = lambda f=None: [{"channel": "s", "name": "Board"}]
        scraper.scrape_startup_source = lambda s: list(self.jobs)
        matcher.score_job_multi = self._score
        cover_letter.generate_cover_letter = self._cover
        resume.generate_tailored_resume = self._resume

    def _score(self, profiles, title, desc):
        self.calls["score"] += 1
        if self.score_error:
            raise RuntimeError(self.score_error)
        if self.score_failed:
            return {"scores": {"DE": 0.0}, "best_profile": "DE", "best_score": 0.0, "reasoning": "", "failed": True}
        return {"scores": {"DE": self.score}, "best_profile": "DE", "best_score": self.score, "reasoning": ""}

    def _cover(self, *a, **k):
        self.calls["cover"] += 1
        if self.cover_error:
            raise llm.CliError(402, self.cover_error, fatal=True) if "limit" in self.cover_error else RuntimeError(self.cover_error)
        return "cover letter text"

    def _resume(self, base, profile, company, title, jd, out_dir, **k):
        self.calls["resume"] += 1
        if self.crash_in_resume:
            raise KeyboardInterrupt("simulated crash")          # not an Exception: bypasses error handling
        if self.resume_error:
            raise resume.ResumeError(self.resume_error)
        pdf, audit = resume.resume_paths(base, company, title, out_dir)
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF-fake")
        audit.write_text("{}")
        return {"pdf_path": str(pdf), "audit_path": str(audit)}

    def job(self, title="Data Engineer", company="Acme GmbH", url="https://boards.greenhouse.io/acme/jobs/1001", desc=LONG_JD):
        return {"title": title, "company": company, "url": url, "description": desc, "source_channel": "s"}

    def run(self, *argv) -> int:
        sys.argv = ["run_pipeline.py", "--tier", "D", *argv]
        try:
            rp.main()
            return 0
        except SystemExit as e:
            return int(e.code or 0)

    def tracker(self) -> list[dict]:
        if not rp.LOG_PATH.exists():
            return []
        return json.loads(rp.LOG_PATH.read_text(encoding="utf-8"))["applications"]


def test_clean_run_completes_and_second_run_is_free():
    env = Env(); env.jobs = [env.job()]
    assert env.run() == 0
    (e,) = env.tracker()
    assert e["status"] == "pending_review" and e["stage"] == "done" and "jd_text" not in e
    assert env.calls == {"score": 1, "cover": 1, "resume": 1}
    env.run()
    assert env.calls == {"score": 1, "cover": 1, "resume": 1}, "nothing may be repeated"


def test_crash_mid_job_resumes_without_rescoring_or_new_cover_letter():
    env = Env(); env.jobs = [env.job()]
    env.crash_in_resume = True
    try:
        env.run()
    except BaseException:
        pass
    rp.LOCK_PATH.unlink(missing_ok=True)     # a crashed process leaves its lock behind (stale after 3h)
    (e,) = env.tracker()
    assert e["status"] == "in_progress" and e["stage"] == "cover_letter_done"
    assert e["cover_attempts"] == 1 and e["resume_attempts"] == 1, "attempt counted before the call"
    assert e["jd_text"], "job text kept so the job can be finished without the posting"
    env.crash_in_resume = False
    env.jobs = []                            # the posting has even disappeared from the board
    assert env.run() == 0
    (e,) = env.tracker()
    assert e["status"] == "pending_review"
    assert env.calls["score"] == 1 and env.calls["cover"] == 1, "no re-score, no second cover letter"


def test_failed_step_is_retried_next_run_then_gives_up_after_three():
    env = Env(); env.jobs = [env.job()]
    env.resume_error = "validation failed"
    for expected_attempts in (1, 2):
        env.run()
        (e,) = env.tracker()
        assert e["status"] == "in_progress" and e["resume_attempts"] == expected_attempts
    env.run()
    (e,) = env.tracker()
    assert e["status"] == "resume_failed" and e["cover_letter_path"], "keeps its cover letter"
    assert env.calls["resume"] == 3 and env.calls["cover"] == 1 and env.calls["score"] == 1
    env.run()
    assert env.calls["resume"] == 3, "no more attempts after the limit"


def test_fatal_api_error_stops_run_and_leaves_jobs_pending_with_attempt_refunded():
    env = Env(); env.jobs = [env.job(), env.job("ML Engineer", url="https://x.example/2", desc=LONG_JD.replace("Python", "Rust"))]
    env.cover_error = "usage limit reached"
    assert env.run() == 2
    entries = env.tracker()
    assert len(entries) == 1, "run stopped at the first job"
    assert entries[0]["status"] == "in_progress" and entries[0]["cover_attempts"] == 0
    assert json.loads(rp.LAST_RUN_PATH.read_text())["stopped_early"] is True
    env.cover_error = None
    assert env.run() == 0
    assert {e["status"] for e in env.tracker()} == {"pending_review"} and len(env.tracker()) == 2


def test_three_consecutive_scoring_failures_stop_the_run():
    env = Env(); env.jobs = [env.job(f"Data Engineer {i}", url=f"https://x.example/{i}", desc=LONG_JD + str(i) * 50) for i in range(5)]
    env.score_error = "connection reset"
    assert env.run() == 2
    assert env.calls["score"] == 3 and env.tracker() == [], "failed jobs are not recorded"


def test_unparseable_score_is_not_recorded_as_below_threshold():
    env = Env(); env.jobs = [env.job()]
    env.score_failed = True
    env.run()
    assert env.tracker() == []
    env.score_failed = False
    env.run()
    assert len(env.tracker()) == 1


def test_finished_files_on_disk_are_reused_when_tracker_is_lost():
    env = Env(); env.jobs = [env.job()]
    env.run()
    rp.LOG_PATH.unlink(); rp.LOG_PATH.with_name("tracker.json.bak").unlink()
    env.run()
    assert env.calls["cover"] == 1 and env.calls["resume"] == 1, "cover letter and resume reused from disk"
    assert env.tracker()[0]["status"] == "pending_review"


def test_corrupt_tracker_restores_from_backup():
    env = Env(); env.jobs = [env.job()]
    env.run()
    rp.LOG_PATH.write_text("{ not json")
    env.run()
    assert len(env.tracker()) == 1 and env.calls["score"] == 1


def test_lock_blocks_a_second_run():
    env = Env(); env.jobs = [env.job()]
    assert rp.acquire_lock()
    assert env.run() == 3
    rp.release_lock()


def test_duplicates_are_skipped_by_every_rule():
    env = Env()
    env.jobs = [env.job()]
    env.run()
    n = env.calls["score"]
    variants = {
        "tracking params": env.job(url="https://www.boards.greenhouse.io/acme/jobs/1001/?utm_source=x&gh_src=y"),
        "same ATS id, other board": env.job(url="https://job-boards.greenhouse.io/acme/jobs/1001/apply"),
        "company+title variants": env.job(title="Data Engineer (m/w/d)", company="ACME", url="https://other.example/p/9"),
        "same text, new title/url": env.job(title="Senior Data Platform Engineer", url="https://other.example/p/10"),
    }
    for name, job in variants.items():
        env.jobs = [job]
        env.run()
        assert env.calls["score"] == n, f"'{name}' was scored again"
    assert len(env.tracker()) == 1
    assert "https://other.example/p/9" in env.tracker()[0].get("alt_urls", [])
    env.jobs = [env.job(title="Data Analyst", url="https://other.example/p/11", desc=LONG_JD.replace("Python", "R"))]
    env.run()
    assert len(env.tracker()) == 2, "a genuinely different job is still processed"


def test_dry_run_writes_nothing_and_dedupes_within_the_run():
    env = Env()
    env.jobs = [env.job(), env.job(url="https://other.example/dup")]    # same company+title
    assert env.run("--dry-run") == 0
    assert env.calls["score"] == 1
    assert not rp.LOG_PATH.exists() and not rp.LOCK_PATH.exists()


def test_normalisers():
    assert dedupe.normalize_company("N26 GmbH") == dedupe.normalize_company("n26")
    assert dedupe.normalize_title("Data Engineer (f/m/x)") == dedupe.normalize_title("Data Engineer  [m/w/d]")
    assert dedupe.normalize_title("Sr. Data Engineer") == "senior data engineer"
    assert dedupe.ats_key("https://jobs.lever.co/acme/123e4567-e89b-12d3-a456-426614174000") == "lever:123e4567-e89b-12d3-a456-426614174000"
    assert dedupe.jd_hash("short") == ""


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

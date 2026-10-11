"""
job_pipeline/run_pipeline.py
============================
Main entry point. Run this daily (manually or via Task Scheduler).
It scans all companies, matches roles, generates cover letters and tailored
resume PDFs for strong matches, and logs results.

Failure handling (checkpointing):
  * A job is written to tracker.json the moment it is scored, and again after
    every finished step (cover letter, resume). Each step is saved before the
    next one starts, and every save is atomic.
  * A job whose steps are not all done has status "in_progress". The next run
    finishes those FIRST, from the stored job text, without re-scoring and
    without redoing a finished step - even if the posting has disappeared.
  * A step that keeps failing is retried on later runs, at most
    MAX_STAGE_ATTEMPTS times (the attempt is counted before the call, so a crash
    mid-call also counts). Then the job ends as resume_failed / needs_attention.
  * If Claude is unusable (not logged in, usage limit reached) or fails repeatedly, the run
    stops cleanly with exit code 2 and the remaining jobs stay pending.
  * Finished files on disk are reused, never regenerated.
Duplicate detection (dedupe.py) stops the same application being scored or
generated twice, even when it appears under a different URL or board.

Usage:
    python run_pipeline.py                  # Full run (Tiers A-C companies + Tier D startup boards)
    python run_pipeline.py --dry-run        # Scrape + score only, writes nothing
    python run_pipeline.py --pending-only   # Only finish unfinished jobs from earlier runs
    python run_pipeline.py --tier A         # Only Tier A companies
    python run_pipeline.py --tier D         # Only Tier D startup boards (startup_sources.json)
    python run_pipeline.py --source startupjobs   # One startup-board channel
    python run_pipeline.py --company N26    # Single company test

Exit codes: 0 ok | 2 stopped early (Claude CLI login/usage-limit problem; jobs left pending) | 3 another run is active

Requirements:
    pip install -r requirements.txt
    playwright install chromium
    Claude Code CLI installed and logged in (model calls use your subscription, no API key)
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

# --- Paths ---
BASE_DIR = Path(__file__).parent
PROFILE_PATH = BASE_DIR / "profile.md"          # full profile (used for cover letters)
# One short positioning summary per resume variant, used for multi-profile matching.
# Dict order is the tie-break order when two profiles score identically.
PROFILE_PATHS = {
    "DE": BASE_DIR / "profile_de.md",   # Data Engineer
    "DS": BASE_DIR / "profile_ds.md",   # Data Science
    "AI": BASE_DIR / "profile_ai.md",   # AI Engineer
}
COMPANIES_PATH = BASE_DIR / "companies.json"
OUTPUT_DIR = BASE_DIR / "applications"
LOG_PATH = BASE_DIR / "tracker.json"
LOCK_PATH = BASE_DIR / ".pipeline.lock"
LAST_RUN_PATH = BASE_DIR / "last_run.json"
NEAR_THRESHOLD_LOG = BASE_DIR / "near_threshold_log.jsonl"

# Minimum best-profile fit score (0.0-1.0) to generate artefacts. 0.60 is a
# starting point, not a verified number — see NEAR_THRESHOLD_BAND.
DEFAULT_MIN_SCORE = 0.60
# Jobs scoring in this band are appended to NEAR_THRESHOLD_LOG (the log never
# rejects anything itself) so the cutoff can be calibrated against real output.
NEAR_THRESHOLD_BAND = (0.50, 0.65)

# Tailor the resume for strong matches: any tier, best-profile score >= 70%.
RESUME_MIN_SCORE = 0.70

MAX_STAGE_ATTEMPTS = 3            # tries per step (cover letter / resume) across runs
MAX_CONSECUTIVE_API_FAILURES = 3  # then the run stops and leaves jobs pending
LOCK_STALE_HOURS = 3              # a lock older than this is treated as left over from a crash
JD_STORE_CHARS = 4000             # job text kept on unfinished entries so they can be resumed

console = Console()


class StopRun(Exception):
    """The API is unusable for now: stop cleanly and leave the remaining jobs pending."""


# --------------------------------------------------------------------------- #
# Run lock, atomic tracker I/O
# --------------------------------------------------------------------------- #
def acquire_lock() -> bool:
    """Create the lock file; False if another run holds a fresh one."""
    if LOCK_PATH.exists():
        age_h = (time.time() - LOCK_PATH.stat().st_mtime) / 3600
        if age_h < LOCK_STALE_HOURS:
            return False
        console.print(f"[yellow]Ignoring stale lock ({age_h:.1f}h old) from an earlier crashed run.[/yellow]")
        LOCK_PATH.unlink(missing_ok=True)
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(f"{os.getpid()} {datetime.now().isoformat()}\n")
    return True


def release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def write_text_atomic(path: Path, text: str) -> None:
    """Write via a temp file + rename so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_tracker() -> dict:
    """Load the tracker. A corrupt file falls back to the last good backup; it is never silently reset."""
    bak = LOG_PATH.with_name(LOG_PATH.name + ".bak")
    for candidate in (LOG_PATH, bak):
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if candidate is bak:
                    console.print("[yellow]tracker.json was unreadable — restored from tracker.json.bak[/yellow]")
                return data
            except json.JSONDecodeError:
                continue
    if LOG_PATH.exists():
        raise SystemExit("tracker.json (and its .bak) are corrupt — fix or move them; refusing to start "
                         "from an empty tracker, which would re-process everything.")
    return {"applications": []}


def save_tracker(tracker: dict) -> None:
    """Atomic save, keeping the previous version as tracker.json.bak."""
    if LOG_PATH.exists():
        shutil.copyfile(LOG_PATH, LOG_PATH.with_name(LOG_PATH.name + ".bak"))
    write_text_atomic(LOG_PATH, json.dumps(tracker, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# Profiles / companies
# --------------------------------------------------------------------------- #
def load_profiles() -> tuple[dict[str, str], str]:
    """Return ({profile_key: summary_text}, full_profile_text).

    Variant files that don't exist are skipped. If NONE exist, profile.md is
    used as a single "DE" profile so the pipeline still runs (no multi-profile
    comparison). full_profile_text is profile.md ("" if absent) and is what
    cover letters are written from.
    """
    variants = {}
    for key, path in PROFILE_PATHS.items():
        if path.exists():
            variants[key] = path.read_text(encoding="utf-8")
    full = PROFILE_PATH.read_text(encoding="utf-8") if PROFILE_PATH.exists() else ""

    if not variants:
        if not full:
            raise FileNotFoundError(
                "No profile found: create profile_de.md / profile_ds.md / profile_ai.md "
                "(see the *.example.md templates) or profile.md.")
        console.print("[yellow]No profile_de/ds/ai.md found — falling back to profile.md "
                      "as a single profile (no multi-profile comparison).[/yellow]")
        variants = {"DE": full}
    return variants, full


def load_companies(tier_filter: str = None, company_filter: str = None) -> list[dict]:
    """Load and optionally filter companies.

    Companies marked "skip": true are excluded (no working scraper backend —
    see their notes), UNLESS explicitly requested by name via --company.
    """
    with open(COMPANIES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    companies = data["companies"]

    if company_filter:
        # Explicit single-company request: honour it even if marked skip.
        return [c for c in companies if company_filter.lower() in c["name"].lower()]

    companies = [c for c in companies if not c.get("skip", False)]
    if tier_filter:
        companies = [c for c in companies if c["fit_tier"] == tier_filter.upper()]

    return companies


# --------------------------------------------------------------------------- #
# Tracker entries
# --------------------------------------------------------------------------- #
def new_entry(company: str, role_title: str, role_url: str, fit_score: float, fingerprints: dict,
              status: str, date_posted: str = "", spoc_email: str = "", tier: str = "",
              source_channel: str = "", scores: dict = None, best_profile: str = "",
              jd_text: str = "", language: str = "") -> dict:
    """Build a tracker entry.

    `first_seen` is stamped once, when the job is first written; the dashboard
    sorts on it (newest first). `fit_score` is the best-profile score, `scores`
    holds every profile's score. Unfinished entries (status "in_progress") keep
    `jd_text` so they can be completed later without the posting still being online.
    """
    now = datetime.now().isoformat()
    entry = {
        "company": company,
        "role_title": role_title,
        "role_url": role_url,
        "fit_score": fit_score,
        "scores": scores or {},
        "best_profile": best_profile,
        "tier": tier,
        "source_channel": source_channel,
        "status": status,
        "stage": "scored",
        "date_posted": date_posted,
        "spoc_email": spoc_email,
        "first_seen": now,
        "applied_date": now,
        "cover_letter_path": None,
        "resume_path": None,             # tailored one-page PDF
        "cover_attempts": 0,
        "resume_attempts": 0,
        "cover_error": "",
        "resume_error": "",              # non-empty = last tailoring attempt failed
        "language": language,            # why the posting passed the language rule
        "fingerprints": fingerprints,
        "notes": "",
    }
    if jd_text:
        entry["jd_text"] = jd_text
    return entry


def log_near_threshold(entry: dict):
    """Append a job scoring inside NEAR_THRESHOLD_BAND to the calibration log."""
    with open(NEAR_THRESHOLD_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def safe_name(text: str) -> str:
    """Filesystem-safe name: spaces -> '_', '/' -> '-', other Windows-illegal chars dropped."""
    return re.sub(r'[<>:"\\|?*]', "", text.replace("/", "-").replace(" ", "_"))


def file_ok(path) -> bool:
    return bool(path) and Path(path).exists() and Path(path).stat().st_size > 0


def classify_error(e: Exception) -> str:
    """'fatal' = the API is unusable until a human acts (bad key, no credit); else 'transient'."""
    if getattr(e, "fatal", False) or getattr(e, "status_code", None) in (401, 402):
        return "fatal"          # llm.CliError: login problem, usage limit, CLI missing
    msg = str(e).lower()
    fatal_markers = ("usage limit", "limit reached", "not logged in", "please run /login",
                     "credit balance", "billing", "insufficient")
    return "fatal" if any(m in msg for m in fatal_markers) else "transient"


def print_summary(results: list[dict]):
    """Print a summary table of the run."""
    table = Table(title="Pipeline Run Summary", show_header=True, header_style="bold cyan")
    table.add_column("Company", style="white")
    table.add_column("Role", style="white")
    table.add_column("Fit Score", style="green")
    table.add_column("Profile", style="magenta")
    table.add_column("Status", style="yellow")

    for r in results:
        score_str = f"{r['fit_score']:.0%}" if r.get("fit_score") else "—"
        table.add_row(r["company"], r.get("role_title", "—"), score_str,
                      r.get("best_profile", "—"), r.get("status", "—"))

    console.print(table)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run(args) -> int:
    # Lazy imports — only loaded when needed
    import scraper
    from scraper import scrape_company, scrape_startup_source, load_startup_sources, assess_language
    from matcher import score_job_multi
    from cover_letter import generate_cover_letter
    from resume import generate_tailored_resume, existing_resume, load_base
    from jd_cleaner import clean_job_description
    from dedupe import DedupeIndex, fingerprints

    import llm

    # All model calls go through the Claude Code CLI on the subscription. An API key in the
    # environment is stripped for every call (otherwise Claude Code would bill the API).
    if llm.warn_if_api_key_set():
        console.print("[yellow]ANTHROPIC_API_KEY is set in your environment. It is ignored by this pipeline "
                      "(removed from every CLI call); consider unsetting it.[/yellow]")
    try:
        llm.preflight()      # CLI present, logged in, plan not exhausted - before any job is touched
    except llm.CliError as e:
        console.print(f"[bold red]Claude CLI not usable: {e}[/bold red]")
        console.print("[red]No job was touched; everything stays pending for the next run.[/red]")
        return 2

    profiles, full_profile = load_profiles()
    console.print(f"[dim]Profiles scored per job: {', '.join(profiles)}[/dim]")

    # Which sources run: Tiers A-C company scrapers and/or Tier D startup boards.
    tier_d_only = bool(args.source) or (args.tier or "").upper() == "D"
    run_tier_d = tier_d_only or not (args.tier or args.company)
    companies = [] if (tier_d_only or args.pending_only) else load_companies(
        tier_filter=args.tier, company_filter=args.company)
    sources = load_startup_sources(args.source) if (run_tier_d and not args.pending_only) else []
    tracker = load_tracker()
    index = DedupeIndex(tracker["applications"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Structured base resume (BASE_RESUME/base.json); tailored PDFs for strong (>= 70%) matches in any tier.
    base_resume = load_base()
    if base_resume is None:
        console.print("[yellow]No BASE_RESUME/base.json found — resume tailoring disabled.[/yellow]")

    results = []
    stats = {"scraped": 0, "matched": 0, "applied": 0, "finished_pending": 0}
    duplicates: dict[str, int] = {}
    api = {"consecutive_failures": 0}

    def save():
        save_tracker(tracker)

    def api_ok():
        api["consecutive_failures"] = 0

    def api_failed(e: Exception):
        """Count an API failure; raise StopRun for a fatal error or repeated failures."""
        if classify_error(e) == "fatal":
            raise StopRun(f"API unusable: {str(e)[:160]}")
        api["consecutive_failures"] += 1
        if api["consecutive_failures"] >= MAX_CONSECUTIVE_API_FAILURES:
            raise StopRun(f"{api['consecutive_failures']} consecutive API failures: {str(e)[:160]}")

    def attempt(entry: dict, key: str, produce) -> bool:
        """Run one checkpointed step ('cover' or 'resume'). True on success.

        The attempt is counted and saved BEFORE the call, so a crash mid-call
        still uses one (a job that always crashes cannot loop forever). A fatal
        API error refunds the attempt, because it is not the job's fault.
        """
        counter = f"{key}_attempts"
        if entry.get(counter, 0) >= MAX_STAGE_ATTEMPTS:
            return False
        entry[counter] = entry.get(counter, 0) + 1
        entry[f"{key}_error"] = ""
        save()
        try:
            produce()
        except Exception as e:
            entry[f"{key}_error"] = str(e)[:300]
            console.print(f"    [red]{key} failed (attempt {entry[counter]}/{MAX_STAGE_ATTEMPTS}): {e}[/red]")
            # Only API-type failures count toward "the API is down"; a resume that fails
            # validation (ResumeError) is the job's problem, not the provider's.
            is_api = e.__class__.__module__.split(".")[0] in ("llm", "httpx") or isinstance(e, RuntimeError)
            try:
                if is_api:
                    api_failed(e)
            except StopRun:
                entry[counter] -= 1          # refund: not the job's fault
                save()
                raise
            save()
            return False
        api_ok()
        save()
        return True

    def finish_entry(entry: dict) -> None:
        """Do whatever is still missing for this application, checkpointing after each step."""
        company, title, best = entry["company"], entry["role_title"], entry.get("best_profile", "")
        jd = entry.get("jd_text", "")
        company_dir = OUTPUT_DIR / safe_name(company)

        # Step 1: cover letter (reuse a finished file, e.g. left by an interrupted run)
        cl_path = company_dir / f"cover_letter_{safe_name(title)}.md"
        if not file_ok(entry.get("cover_letter_path")):
            if file_ok(cl_path):
                entry["cover_letter_path"] = str(cl_path)
                console.print(f"    [dim]Cover letter already on disk — reusing {cl_path.name}[/dim]")
                entry["stage"] = "cover_letter_done"
                save()
            else:
                def make_cover():
                    text = generate_cover_letter(full_profile or profiles.get(best, ""), company, title, jd)
                    write_text_atomic(cl_path, text)
                    entry["cover_letter_path"] = str(cl_path)
                    entry["stage"] = "cover_letter_done"
                if attempt(entry, "cover", make_cover):
                    console.print(f"    [green]Cover letter saved: {cl_path.name}[/green]")

        # Step 2: tailored resume PDF for strong matches
        wants_resume = base_resume is not None and entry["fit_score"] >= RESUME_MIN_SCORE
        if wants_resume and not file_ok(entry.get("resume_path")):
            found = existing_resume(base_resume, company, title, company_dir)
            if found:
                entry["resume_path"] = found["pdf_path"]
                entry["resume_error"] = ""
                entry["stage"] = "resume_done"
                save()
                console.print(f"    [dim]Resume already on disk — reusing {Path(found['pdf_path']).name}[/dim]")
            else:
                def make_resume():
                    out = generate_tailored_resume(base_resume, best, company, title, jd, company_dir)
                    entry["resume_path"] = out["pdf_path"]
                    entry["stage"] = "resume_done"
                if attempt(entry, "resume", make_resume):
                    console.print(f"    [green]Tailored resume PDF ({best} profile): "
                                  f"{Path(entry['resume_path']).name}[/green]")

        # Step 3: decide whether anything is still open
        cover_ok, resume_ok = file_ok(entry.get("cover_letter_path")), file_ok(entry.get("resume_path"))
        cover_open = not cover_ok and entry.get("cover_attempts", 0) < MAX_STAGE_ATTEMPTS
        resume_open = wants_resume and not resume_ok and entry.get("resume_attempts", 0) < MAX_STAGE_ATTEMPTS
        if cover_open or resume_open:
            entry["status"] = "in_progress"          # picked up again first thing next run
            console.print("    [yellow]Left in_progress — will be retried next run[/yellow]")
            save()
            return
        if not cover_ok:
            entry["status"] = "needs_attention"
        elif wants_resume and not resume_ok:
            entry["status"] = "resume_failed"
        else:
            entry["status"] = "pending_review" if resume_ok else "cover_letter_ready"
        entry["stage"] = "done"
        entry.pop("jd_text", None)
        save()
        stats["applied"] += 1
        console.print(f"    [cyan]Done: {entry['status']} ✓[/cyan]")

    def finish_pending():
        pending = [a for a in tracker["applications"] if a.get("status") == "in_progress"]
        if args.company:
            pending = [a for a in pending if args.company.lower() in a["company"].lower()]
        if not pending:
            return
        console.print(f"\n[bold]→ Finishing {len(pending)} unfinished job(s) from earlier runs[/bold]")
        for entry in pending:
            console.print(f"  ↻ {entry['company']} — {entry['role_title']} "
                          f"[dim](last step: {entry.get('stage', '?')})[/dim]")
            if not entry.get("jd_text"):
                entry["status"], entry["stage"] = "needs_attention", "done"
                entry["cover_error"] = "no stored job text to resume from"
                save()
                console.print("    [red]No stored job text — marked needs_attention[/red]")
                continue
            finish_entry(entry)
            stats["finished_pending"] += 1
            results.append({"company": entry["company"], "role_title": entry["role_title"],
                            "fit_score": entry["fit_score"], "best_profile": entry.get("best_profile", ""),
                            "status": entry["status"]})

    def process_job(job: dict, company_name: str, tier: str, source_channel: str = ""):
        """Dedupe -> score (all profiles) -> checkpoint -> cover letter -> resume."""
        url = job.get("url", "")
        # Role + requirements + company context only (benefits etc. dropped).
        jd_store = clean_job_description(job["description"], JD_STORE_CHARS)
        jd_scoring = jd_store[:3000]

        fp = fingerprints(company_name, job["title"], url, jd_store)
        dup, reason = index.find(fp)
        if dup is not None:
            duplicates[reason] = duplicates.get(reason, 0) + 1
            console.print(f"  [dim]Duplicate ({reason}) of {dup.get('company')} — {dup.get('role_title')} "
                          f"[{dup.get('status', 'seen this run')}]: {job['title']} — skipping[/dim]")
            if url and not args.dry_run and "status" in dup and url != dup.get("role_url") \
                    and url not in dup.setdefault("alt_urls", []):
                dup["alt_urls"].append(url)          # remember this URL so it matches instantly next time
                index.add({"alt": [(fp["ats"], fp["url"])]}, dup)
                save()
            return

        # Score against every profile in one call (only reached for genuinely new jobs)
        try:
            match = score_job_multi(profiles, job["title"], jd_scoring)
        except Exception as e:
            console.print(f"  [red]Scoring failed for {job['title']}: {e}[/red]")
            api_failed(e)       # may raise StopRun
            return              # not recorded: it is scored again next run
        if match.get("failed"):
            console.print(f"  [yellow]Could not parse the scoring reply for {job['title']} — "
                          f"not recorded, will be retried next run[/yellow]")
            return
        api_ok()
        score, best, scores = match["best_score"], match["best_profile"], match["scores"]

        score_emoji = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🔴"
        per_profile = " ".join(f"{k}:{v:.0%}" for k, v in scores.items())
        console.print(f"  {score_emoji} [{score:.0%} {best}] {job['title']}  [dim]({per_profile})[/dim]")

        result = {"company": company_name, "role_title": job["title"],
                  "fit_score": score, "best_profile": best}

        if NEAR_THRESHOLD_BAND[0] <= score <= NEAR_THRESHOLD_BAND[1]:
            log_near_threshold({
                "logged": datetime.now().isoformat(), "company": company_name,
                "role_title": job["title"], "url": url, "tier": tier,
                "best_profile": best, "scores": scores, "min_score": args.min_score,
                "passed": score >= args.min_score, "dry_run": args.dry_run,
                "reasoning": match.get("reasoning", ""),
            })

        below = score < args.min_score
        if not below:
            stats["matched"] += 1

        if args.dry_run:
            index.add(fp, {"company": company_name, "role_title": job["title"]})   # catch in-run duplicates
            results.append({**result, "status": "below_threshold" if below else "dry_run_match"})
            if below:
                console.print("    [dim]Below threshold[/dim]")
            return

        # CHECKPOINT 1: record the score immediately. Below-threshold jobs are recorded too so they
        # are never scored again; strong ones keep their job text so they can be finished later.
        entry = new_entry(company_name, job["title"], url, score, fp,
                          "below_threshold" if below else "in_progress",
                          date_posted=job.get("date_posted", ""), spoc_email=job.get("spoc_email", ""),
                          tier=tier, source_channel=source_channel, scores=scores, best_profile=best,
                          jd_text="" if below else jd_store,
                          language=job.get("language_note") or assess_language(job["description"])["note"])
        if below:
            entry["stage"] = "done"
        tracker["applications"].append(entry)
        index.add(fp, entry)
        save()
        if below:
            console.print("    [dim]Below threshold — recorded & skipping[/dim]")
            results.append({**result, "status": "below_threshold"})
            return

        finish_entry(entry)
        results.append({**result, "status": entry["status"]})

    stopped = None
    try:
        # Unfinished work from earlier runs goes first.
        if not args.dry_run:
            finish_pending()

        # --- Tiers A-C: fixed company list ---
        for company in companies:
            console.print(f"\n[bold]→ {company['name']}[/bold] ({company['category']}, Tier {company['fit_tier']})")
            try:
                jobs = scrape_company(company)
                stats["scraped"] += len(jobs)
                console.print(f"  [dim]Found {len(jobs)} job(s)[/dim]")
            except Exception as e:
                console.print(f"  [red]Scrape failed: {e}[/red]")
                continue
            for job in jobs:
                process_job(job, company["name"], company["fit_tier"])

        # --- Tier D: startup job boards (re-scraped every run) ---
        for source in sources:
            console.print(f"\n[bold]→ {source['name']}[/bold] (startup board, Tier D)")
            try:
                jobs = scrape_startup_source(source)
                stats["scraped"] += len(jobs)
                console.print(f"  [dim]Found {len(jobs)} job(s)[/dim]")
            except Exception as e:
                console.print(f"  [red]Scrape failed: {e}[/red]")
                continue
            for job in jobs:
                process_job(job, job.get("company", "Unknown startup"), "D",
                            source_channel=job.get("source_channel", source["channel"]))
    except StopRun as e:
        stopped = str(e)

    # Summary
    still_pending = sum(1 for a in tracker["applications"] if a.get("status") == "in_progress")
    console.rule()
    console.print(f"\n[bold]Run complete[/bold]" if not stopped else "\n[bold red]Run STOPPED EARLY[/bold red]")
    console.print(f"  Sources checked:   {len(companies)} companies, {len(sources)} startup boards")
    console.print(f"  Jobs scraped:      {stats['scraped']}")
    console.print(f"  Jobs matched:      {stats['matched']}")
    console.print(f"  Applications done: {stats['applied']}  (of which finished from earlier runs: {stats['finished_pending']})")
    if duplicates:
        console.print("  Duplicates skipped: " + ", ".join(f"{n} by {k}" for k, n in sorted(duplicates.items())))
    console.print(f"  Still pending:     {still_pending}")
    if scraper.STALE_SKIPPED:
        console.print(f"  [yellow]Dropped as older than {scraper.MAX_AGE_DAYS} days "
                      f"(safety net): {scraper.STALE_SKIPPED}[/yellow]")
    if stopped:
        console.print(f"\n  [red]Reason: {stopped}[/red]")
        console.print("  [red]Unfinished jobs are saved and will be completed on the next run.[/red]")
    console.print(f"\n  Tracker: [cyan]{LOG_PATH}[/cyan]")
    console.print(f"  Output:  [cyan]{OUTPUT_DIR}[/cyan]")
    if NEAR_THRESHOLD_LOG.exists():
        console.print(f"  Near-threshold calibration log: [cyan]{NEAR_THRESHOLD_LOG}[/cyan]\n")

    if not args.dry_run:
        write_text_atomic(LAST_RUN_PATH, json.dumps({
            "finished": datetime.now().isoformat(), "stopped_early": bool(stopped), "reason": stopped or "",
            "pending": still_pending, "duplicates_skipped": duplicates, **stats}, indent=2))

    if results:
        print_summary([r for r in results if r.get("fit_score", 0) >= args.min_score])
    return 2 if stopped else 0


def main():
    parser = argparse.ArgumentParser(description="Chirag's Job Application Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Scrape and score only — writes nothing")
    parser.add_argument("--pending-only", action="store_true",
                        help="Only finish unfinished jobs from earlier runs (no scraping)")
    parser.add_argument("--tier", type=str, help="Filter to specific tier: A, B, C, or D (startup boards)")
    parser.add_argument("--company", type=str, help="Run for a single company by name (Tiers A-C)")
    parser.add_argument("--source", type=str, help="Run a single Tier D startup-board channel (e.g. startupjobs)")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help=f"Minimum best-profile fit score, 0.0-1.0 (default {DEFAULT_MIN_SCORE})")
    args = parser.parse_args()

    console.rule("[bold cyan]Chirag's Job Pipeline[/bold cyan]")
    console.print(f"[dim]Run started: {datetime.now().strftime('%Y-%m-%d %H:%M')}[/dim]")
    console.print(f"[dim]Dry run: {args.dry_run} | Tier filter: {args.tier or 'all'} | Min score: {args.min_score:.0%}[/dim]\n")

    if not args.dry_run and not acquire_lock():
        console.print("[red]Another pipeline run is active (lock file present). Exiting.[/red]")
        sys.exit(3)
    try:
        code = run(args)
    finally:
        if not args.dry_run:
            release_lock()
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()

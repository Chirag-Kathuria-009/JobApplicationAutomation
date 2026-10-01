"""
job_pipeline/run_pipeline.py
============================
Main entry point. Run this daily (manually or via Task Scheduler).
It scans all companies, matches roles, generates cover letters, and logs results.

Usage:
    python run_pipeline.py                  # Full run (Tiers A-C companies + Tier D startup boards)
    python run_pipeline.py --dry-run        # Scrape + score only, no cover letter generation
    python run_pipeline.py --tier A         # Only Tier A companies
    python run_pipeline.py --tier D         # Only Tier D startup boards (startup_sources.json)
    python run_pipeline.py --source startupjobs   # One startup-board channel
    python run_pipeline.py --company N26    # Single company test

Requirements:
    pip install anthropic httpx playwright beautifulsoup4 rich
    playwright install chromium
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import print as rprint

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
NEAR_THRESHOLD_LOG = BASE_DIR / "near_threshold_log.jsonl"

# Minimum best-profile fit score (0.0-1.0) to generate artefacts. 0.60 is a
# starting point, not a verified number — see NEAR_THRESHOLD_BAND.
DEFAULT_MIN_SCORE = 0.60
# Jobs scoring in this band are appended to NEAR_THRESHOLD_LOG (the log never
# rejects anything itself) so the cutoff can be calibrated against real output.
NEAR_THRESHOLD_BAND = (0.50, 0.65)

# Tailor the resume for strong matches: any tier, best-profile score >= 70%.
RESUME_MIN_SCORE = 0.70

console = Console()


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


def load_tracker() -> dict:
    """Load existing application tracker."""
    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"applications": []}


def save_tracker(tracker: dict):
    """Save tracker to disk."""
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(tracker, f, indent=2, ensure_ascii=False)


def find_application(tracker: dict, company: str, role_title: str, role_url: str = "") -> dict | None:
    """Return the existing tracker record for this job, or None if it's new.

    Matches on role_url first (most stable), then company + role_title. Used to
    skip jobs we've already processed so scoring / cover letter / resume aren't
    redone — saving API calls. Any job that was scored before (including
    below-threshold ones) is recorded, so it's recognised here next run. This
    also de-dupes Tier D board postings against Tier A-C company scrapes.
    """
    url = (role_url or "").strip()
    for app in tracker["applications"]:
        if url and (app.get("role_url") or "").strip() == url:
            return app
        if (app.get("company", "").lower() == company.lower()
                and app.get("role_title", "").lower() == role_title.lower()):
            return app
    return None


def log_application(tracker: dict, company: str, role_title: str, role_url: str,
                     fit_score: float, status: str, cover_letter_path: str = None,
                     date_posted: str = "", spoc_email: str = "", resume_path: str = None,
                     tier: str = "", source_channel: str = "",
                     scores: dict = None, best_profile: str = ""):
    """Log a new job to the tracker.

    `first_seen` is stamped here, once, when the job is first written. The
    dashboard sorts on it (newest first) — freshness comes from this, not from
    a hard age cutoff. `fit_score` is the best-profile score; `scores` holds
    every profile's score.
    """
    now = datetime.now().isoformat()
    tracker["applications"].append({
        "company": company,
        "role_title": role_title,
        "role_url": role_url,
        "fit_score": fit_score,
        "scores": scores or {},
        "best_profile": best_profile,
        "tier": tier,
        "source_channel": source_channel,
        "status": status,
        "date_posted": date_posted,
        "spoc_email": spoc_email,
        "first_seen": now,
        "applied_date": now,
        "cover_letter_path": cover_letter_path,
        "resume_path": resume_path,
        "notes": ""
    })
    save_tracker(tracker)


def log_near_threshold(entry: dict):
    """Append a job scoring inside NEAR_THRESHOLD_BAND to the calibration log."""
    with open(NEAR_THRESHOLD_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def safe_name(text: str) -> str:
    """Filesystem-safe name: spaces -> '_', '/' -> '-', other Windows-illegal chars dropped."""
    return re.sub(r'[<>:"\\|?*]', "", text.replace("/", "-").replace(" ", "_"))


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


def main():
    parser = argparse.ArgumentParser(description="Chirag's Job Application Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Scrape and score only — no cover letter or apply")
    parser.add_argument("--tier", type=str, help="Filter to specific tier: A, B, C, or D (startup boards)")
    parser.add_argument("--company", type=str, help="Run for a single company by name (Tiers A-C)")
    parser.add_argument("--source", type=str, help="Run a single Tier D startup-board channel (e.g. startupjobs)")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help=f"Minimum best-profile fit score, 0.0-1.0 (default {DEFAULT_MIN_SCORE})")
    args = parser.parse_args()

    console.rule("[bold cyan]Chirag's Job Pipeline[/bold cyan]")
    console.print(f"[dim]Run started: {datetime.now().strftime('%Y-%m-%d %H:%M')}[/dim]")
    console.print(f"[dim]Dry run: {args.dry_run} | Tier filter: {args.tier or 'all'} | Min score: {args.min_score:.0%}[/dim]\n")

    # Lazy imports — only loaded when needed
    import scraper
    from scraper import scrape_company, scrape_startup_source, load_startup_sources
    from matcher import score_job_multi
    from cover_letter import generate_cover_letter
    from resume import generate_tailored_resume, load_base_resumes

    profiles, full_profile = load_profiles()
    console.print(f"[dim]Profiles scored per job: {', '.join(profiles)}[/dim]")

    # Which sources run: Tiers A-C company scrapers and/or Tier D startup boards.
    tier_d_only = bool(args.source) or (args.tier or "").upper() == "D"
    run_tier_d = tier_d_only or not (args.tier or args.company)
    companies = [] if tier_d_only else load_companies(tier_filter=args.tier, company_filter=args.company)
    sources = load_startup_sources(args.source) if run_tier_d else []
    tracker = load_tracker()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # One base resume per profile (DE / DS / AI), tailored for strong (>= 70%) matches in any tier.
    base_resumes = load_base_resumes(profiles)
    if not base_resumes:
        console.print("[yellow]No base resume found in BASE_RESUME/ — resume tailoring disabled.[/yellow]")
    else:
        missing = [k for k in profiles if k not in base_resumes]
        if missing:
            console.print(f"[yellow]No base resume for profile(s) {', '.join(missing)} — "
                          f"tailoring disabled for jobs that match them best.[/yellow]")

    results = []
    stats = {"scraped": 0, "matched": 0, "applied": 0}
    seen_this_run: set[str] = set()   # in-run de-dupe (also covers --dry-run, which writes no tracker)

    def process_job(job: dict, company_name: str, tier: str, source_channel: str = ""):
        """Dedupe -> score (all profiles) -> threshold -> cover letter -> resume -> log."""
        url = job.get("url", "")
        run_key = url.strip() or f"{company_name}|{job['title']}".lower()
        if run_key in seen_this_run:
            console.print(f"  [dim]Duplicate within this run: {job['title']} — skipping[/dim]")
            return
        seen_this_run.add(run_key)

        # Skip jobs we've already processed in a previous run (matched by
        # URL/company+title). This avoids re-scoring and re-generating
        # artifacts — the expensive, API-billed steps.
        if find_application(tracker, company_name, job["title"], url) is not None:
            console.print(f"  [dim]Already processed: {job['title']} — skipping[/dim]")
            return

        # Score against every profile in one call (only reached for genuinely new jobs)
        try:
            match = score_job_multi(profiles, job["title"], job["description"])
        except Exception as e:
            console.print(f"  [red]Scoring failed for {job['title']}: {e}[/red]")
            return
        score, best, scores = match["best_score"], match["best_profile"], match["scores"]

        score_emoji = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🔴"
        per_profile = " ".join(f"{k}:{v:.0%}" for k, v in scores.items())
        console.print(f"  {score_emoji} [{score:.0%} {best}] {job['title']}  [dim]({per_profile})[/dim]")

        meta = dict(tier=tier, source_channel=source_channel, scores=scores, best_profile=best)
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

        if score < args.min_score:
            # Record below-threshold jobs too (no cover letter / resume) so we
            # don't waste a scoring call on them again next run.
            if not args.dry_run:
                log_application(tracker, company_name, job["title"], url,
                                score, "below_threshold",
                                date_posted=job.get("date_posted", ""),
                                spoc_email=job.get("spoc_email", ""), **meta)
            console.print(f"    [dim]Below threshold — recorded & skipping[/dim]")
            results.append({**result, "status": "below_threshold"})
            return

        stats["matched"] += 1

        if args.dry_run:
            results.append({**result, "status": "dry_run_match"})
            return

        company_dir = OUTPUT_DIR / safe_name(company_name)
        safe_title = safe_name(job["title"])

        # Step 3: Generate cover letter (from the full profile when available)
        try:
            cover_letter_text = generate_cover_letter(
                full_profile or profiles[best], company_name, job["title"], job["description"])
            company_dir.mkdir(parents=True, exist_ok=True)
            cl_path = company_dir / f"cover_letter_{safe_title}.md"
            with open(cl_path, "w", encoding="utf-8") as f:
                f.write(cover_letter_text)
            console.print(f"    [green]Cover letter saved: {cl_path.name}[/green]")
        except Exception as e:
            console.print(f"    [red]Cover letter failed: {e}[/red]")
            cl_path = None

        # Step 3b: Tailor the best-profile base resume for strong matches (score >= 70%, any tier)
        resume_path = None
        if score >= RESUME_MIN_SCORE:
            if best not in base_resumes:
                console.print(f"    [yellow]No base resume for profile {best} — skipping tailoring[/yellow]")
            else:
                try:
                    resume_text = generate_tailored_resume(
                        base_resumes[best], company_name, job["title"], job["description"])
                    company_dir.mkdir(parents=True, exist_ok=True)
                    rz_path = company_dir / f"resume_{safe_title}.md"
                    with open(rz_path, "w", encoding="utf-8") as f:
                        f.write(resume_text)
                    resume_path = str(rz_path)
                    console.print(f"    [green]Tailored resume ({best} base) saved: {rz_path.name}[/green]")
                except Exception as e:
                    console.print(f"    [red]Resume tailoring failed: {e}[/red]")

        # Step 4: Log result
        log_application(tracker, company_name, job["title"], url,
                        score, "cover_letter_ready", str(cl_path) if cl_path else None,
                        date_posted=job.get("date_posted", ""),
                        spoc_email=job.get("spoc_email", ""),
                        resume_path=resume_path, **meta)
        stats["applied"] += 1
        results.append({**result, "status": "cover_letter_ready"})
        console.print(f"    [cyan]Logged to tracker ✓[/cyan]")

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

    # Summary
    console.rule()
    console.print(f"\n[bold]Run complete[/bold]")
    console.print(f"  Sources checked:   {len(companies)} companies, {len(sources)} startup boards")
    console.print(f"  Jobs scraped:      {stats['scraped']}")
    console.print(f"  Jobs matched:      {stats['matched']}")
    console.print(f"  Applications ready:{stats['applied']}")
    if scraper.STALE_SKIPPED:
        console.print(f"  [yellow]Dropped as older than {scraper.MAX_AGE_DAYS} days "
                      f"(safety net): {scraper.STALE_SKIPPED}[/yellow]")
    console.print(f"\n  Tracker: [cyan]{LOG_PATH}[/cyan]")
    console.print(f"  Output:  [cyan]{OUTPUT_DIR}[/cyan]")
    if NEAR_THRESHOLD_LOG.exists():
        console.print(f"  Near-threshold calibration log: [cyan]{NEAR_THRESHOLD_LOG}[/cyan]\n")

    if results:
        print_summary([r for r in results if r.get("fit_score", 0) >= args.min_score])


if __name__ == "__main__":
    main()

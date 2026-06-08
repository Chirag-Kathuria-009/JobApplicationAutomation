"""
job_pipeline/run_pipeline.py
============================
Main entry point. Run this daily (manually or via Task Scheduler).
It scans all companies, matches roles, generates cover letters, and logs results.

Usage:
    python run_pipeline.py                  # Full run
    python run_pipeline.py --dry-run        # Scrape + score only, no cover letter generation
    python run_pipeline.py --tier A         # Only Tier A companies
    python run_pipeline.py --company N26    # Single company test

Requirements:
    pip install anthropic httpx playwright beautifulsoup4 rich
    playwright install chromium
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import print as rprint

# --- Paths ---
BASE_DIR = Path(__file__).parent
PROFILE_PATH = BASE_DIR / "profile.md"
COMPANIES_PATH = BASE_DIR / "companies.json"
OUTPUT_DIR = BASE_DIR / "applications"
LOG_PATH = BASE_DIR / "tracker.json"

# Tailor the resume only for strong matches: Tier A/B roles scoring >= 70%.
RESUME_TIERS = ("A", "B")
RESUME_MIN_SCORE = 0.70

console = Console()


def load_profile() -> str:
    """Load the profile.md file."""
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        return f.read()


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
    below-threshold ones) is recorded, so it's recognised here next run.
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
                     date_posted: str = "", spoc_email: str = "", resume_path: str = None):
    """Log a new application to the tracker."""
    tracker["applications"].append({
        "company": company,
        "role_title": role_title,
        "role_url": role_url,
        "fit_score": fit_score,
        "status": status,
        "date_posted": date_posted,
        "spoc_email": spoc_email,
        "applied_date": datetime.now().isoformat(),
        "cover_letter_path": cover_letter_path,
        "resume_path": resume_path,
        "notes": ""
    })
    save_tracker(tracker)


def print_summary(results: list[dict]):
    """Print a summary table of the run."""
    table = Table(title="Pipeline Run Summary", show_header=True, header_style="bold cyan")
    table.add_column("Company", style="white")
    table.add_column("Role", style="white")
    table.add_column("Fit Score", style="green")
    table.add_column("Status", style="yellow")

    for r in results:
        score_str = f"{r['fit_score']:.0%}" if r.get("fit_score") else "—"
        table.add_row(r["company"], r.get("role_title", "—"), score_str, r.get("status", "—"))

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Chirag's Job Application Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Scrape and score only — no cover letter or apply")
    parser.add_argument("--tier", type=str, help="Filter to specific tier: A, B, or C")
    parser.add_argument("--company", type=str, help="Run for a single company by name")
    parser.add_argument("--min-score", type=float, default=0.5, help="Minimum fit score to process (default 0.5)")
    args = parser.parse_args()

    console.rule("[bold cyan]Chirag's Job Pipeline[/bold cyan]")
    console.print(f"[dim]Run started: {datetime.now().strftime('%Y-%m-%d %H:%M')}[/dim]")
    console.print(f"[dim]Dry run: {args.dry_run} | Tier filter: {args.tier or 'all'} | Min score: {args.min_score:.0%}[/dim]\n")

    # Lazy imports — only loaded when needed
    from scraper import scrape_company
    from matcher import score_job
    from cover_letter import generate_cover_letter
    from resume import generate_tailored_resume, load_base_resume

    profile = load_profile()
    companies = load_companies(tier_filter=args.tier, company_filter=args.company)
    tracker = load_tracker()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Base resume is tailored only for strong Tier A/B matches (see RESUME_* below).
    base_resume = load_base_resume()
    if not base_resume:
        console.print("[yellow]No base resume found in BASE_RESUME/ — resume tailoring disabled.[/yellow]")

    results = []
    total_scraped = 0
    total_matched = 0
    total_applied = 0

    for company in companies:
        console.print(f"\n[bold]→ {company['name']}[/bold] ({company['category']}, Tier {company['fit_tier']})")

        # Step 1: Scrape
        try:
            jobs = scrape_company(company)
            total_scraped += len(jobs)
            console.print(f"  [dim]Found {len(jobs)} job(s)[/dim]")
        except Exception as e:
            console.print(f"  [red]Scrape failed: {e}[/red]")
            continue

        for job in jobs:
            # Skip jobs we've already processed in a previous run (matched by
            # URL/company+title). This avoids re-scoring and re-generating
            # artifacts — the expensive, API-billed steps.
            if find_application(tracker, company["name"], job["title"], job.get("url", "")) is not None:
                console.print(f"  [dim]Already processed: {job['title']} — skipping[/dim]")
                continue

            # Step 2: Score (only reached for genuinely new jobs)
            try:
                score, reasoning = score_job(profile, job["title"], job["description"])
            except Exception as e:
                console.print(f"  [red]Scoring failed for {job['title']}: {e}[/red]")
                continue

            score_emoji = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🔴"
            console.print(f"  {score_emoji} [{score:.0%}] {job['title']}")

            if score < args.min_score:
                # Record below-threshold jobs too (no cover letter / resume) so we
                # don't waste a scoring call on them again next run.
                if not args.dry_run:
                    log_application(tracker, company["name"], job["title"], job.get("url", ""),
                                    score, "below_threshold",
                                    date_posted=job.get("date_posted", ""),
                                    spoc_email=job.get("spoc_email", ""))
                console.print(f"    [dim]Below threshold — recorded & skipping[/dim]")
                results.append({"company": company["name"], "role_title": job["title"],
                                 "fit_score": score, "status": "below_threshold"})
                continue

            total_matched += 1

            if args.dry_run:
                results.append({"company": company["name"], "role_title": job["title"],
                                 "fit_score": score, "status": "dry_run_match"})
                continue

            # Step 3: Generate cover letter
            try:
                cover_letter_text = generate_cover_letter(profile, company["name"], job["title"], job["description"])
                company_dir = OUTPUT_DIR / company["name"].replace(" ", "_")
                company_dir.mkdir(parents=True, exist_ok=True)
                safe_title = job["title"].replace("/", "-").replace(" ", "_")
                cl_path = company_dir / f"cover_letter_{safe_title}.md"
                with open(cl_path, "w", encoding="utf-8") as f:
                    f.write(cover_letter_text)
                console.print(f"    [green]Cover letter saved: {cl_path.name}[/green]")
            except Exception as e:
                console.print(f"    [red]Cover letter failed: {e}[/red]")
                cl_path = None

            # Step 3b: Tailor resume for strong Tier A/B matches (score >= 70%)
            resume_path = None
            if (base_resume and score >= RESUME_MIN_SCORE
                    and company.get("fit_tier") in RESUME_TIERS):
                try:
                    resume_text = generate_tailored_resume(
                        base_resume, company["name"], job["title"], job["description"])
                    company_dir = OUTPUT_DIR / company["name"].replace(" ", "_")
                    company_dir.mkdir(parents=True, exist_ok=True)
                    safe_title = job["title"].replace("/", "-").replace(" ", "_")
                    rz_path = company_dir / f"resume_{safe_title}.md"
                    with open(rz_path, "w", encoding="utf-8") as f:
                        f.write(resume_text)
                    resume_path = str(rz_path)
                    console.print(f"    [green]Tailored resume saved: {rz_path.name}[/green]")
                except Exception as e:
                    console.print(f"    [red]Resume tailoring failed: {e}[/red]")

            # Step 4: Log result
            log_application(tracker, company["name"], job["title"], job.get("url", ""),
                            score, "cover_letter_ready", str(cl_path) if cl_path else None,
                            date_posted=job.get("date_posted", ""),
                            spoc_email=job.get("spoc_email", ""),
                            resume_path=resume_path)
            total_applied += 1
            results.append({"company": company["name"], "role_title": job["title"],
                             "fit_score": score, "status": "cover_letter_ready"})
            console.print(f"    [cyan]Logged to tracker ✓[/cyan]")

    # Summary
    console.rule()
    console.print(f"\n[bold]Run complete[/bold]")
    console.print(f"  Companies checked: {len(companies)}")
    console.print(f"  Jobs scraped:      {total_scraped}")
    console.print(f"  Jobs matched:      {total_matched}")
    console.print(f"  Applications ready:{total_applied}")
    console.print(f"\n  Tracker: [cyan]{LOG_PATH}[/cyan]")
    console.print(f"  Output:  [cyan]{OUTPUT_DIR}[/cyan]\n")

    if results:
        print_summary([r for r in results if r.get("fit_score", 0) >= args.min_score])


if __name__ == "__main__":
    main()

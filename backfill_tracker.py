"""
backfill_tracker.py
===================
One-off (idempotent) utility: fills in `date_posted` (Greenhouse `first_published`,
the original post date) and a best-effort `spoc_email` for every entry already in
tracker.json. Re-running only fills blanks, so it's safe to run again.

    python backfill_tracker.py
"""

import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import httpx
from bs4 import BeautifulSoup

from scraper import extract_spoc_email

BASE_DIR = Path(__file__).parent
TRACKER_PATH = BASE_DIR / "tracker.json"


def parse_greenhouse_ref(role_url: str, company: str) -> tuple[str, str]:
    """Return (board_token, job_id) for a Greenhouse job URL.

    Handles both job-boards.greenhouse.io/{board}/jobs/{id} and custom domains
    (e.g. sumup.com/careers/positions/{id}?gh_jid={id}) where the board token is
    derived from the company name.
    """
    parsed = urlparse(role_url)
    parts = [p for p in parsed.path.split("/") if p]
    job_id = (parse_qs(parsed.query).get("gh_jid") or [""])[0]

    if not job_id and "jobs" in parts:
        idx = parts.index("jobs")
        if idx + 1 < len(parts):
            job_id = parts[idx + 1]
    if not job_id and parts and parts[-1].isdigit():
        job_id = parts[-1]

    if "greenhouse.io" in parsed.netloc:
        board = parts[0] if parts else ""
    else:
        board = company.lower().replace(" ", "").replace("-", "").replace("_", "")

    return board, job_id


def fetch_greenhouse_details(board: str, job_id: str) -> tuple[str, str]:
    """Return (first_published, spoc_email) for a single Greenhouse job."""
    if not board or not job_id:
        return "", ""
    url = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}"
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code} for {board}/{job_id}")
            return "", ""
        data = resp.json()
        date_posted = data.get("first_published", "")
        description = BeautifulSoup(
            data.get("content", ""), "html.parser"
        ).get_text(separator=" ")
        return date_posted, extract_spoc_email(description)
    except Exception as e:
        print(f"    Error for {board}/{job_id}: {e}")
        return "", ""


def main():
    tracker = json.loads(TRACKER_PATH.read_text(encoding="utf-8"))
    updated = 0

    for app in tracker.get("applications", []):
        # Skip entries that already have a posted date filled in.
        if app.get("date_posted"):
            continue

        company = app.get("company", "")
        role_url = app.get("role_url", "")
        board, job_id = parse_greenhouse_ref(role_url, company)
        print(f"- {company} | {app.get('role_title','')} ({board}/{job_id})")

        date_posted, spoc_email = fetch_greenhouse_details(board, job_id)
        app["date_posted"] = date_posted
        app["spoc_email"] = spoc_email
        print(f"    date_posted={date_posted or '-'}  spoc_email={spoc_email or '-'}")
        updated += 1

    TRACKER_PATH.write_text(
        json.dumps(tracker, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nBackfilled {updated} ent(ies). Saved {TRACKER_PATH.name}.")


if __name__ == "__main__":
    main()

# Auto Job Applier — Claude-Powered Job Application Pipeline

An end-to-end pipeline that **finds, scores, and drafts applications** for data
roles across European company career portals. It scrapes jobs from many Applicant
Tracking Systems (ATS), uses the **Claude API** to score how well each role fits
your profile, and auto-generates tailored **cover letters** and **resumes** for the
strong matches. Everything is logged to a tracker and viewable in a local web
dashboard.

> Built for a Data Engineer / Data Analyst / Data Scientist targeting Germany &
> nearby EU countries, but the profile, company list, and filters are fully
> configurable for any role or region.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [How it works (pipeline flow)](#how-it-works-pipeline-flow)
3. [Project structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Setup](#setup)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [The dashboard](#the-dashboard)
9. [Supported ATS / scrapers](#supported-ats--scrapers)
10. [Module reference](#module-reference)
11. [Cost](#cost)
12. [Privacy, secrets & what is git-ignored](#privacy-secrets--what-is-git-ignored)
13. [Troubleshooting](#troubleshooting)
14. [Disclaimer](#disclaimer)

---

## What it does

- **Scrapes** open roles from 50+ company career portals across 8 different ATS
  backends (Greenhouse, Lever, Ashby, SmartRecruiters, Recruitee, Workday,
  Radancy, plus Amazon's proprietary portal).
- **Filters** jobs by role keywords, target location (Germany + bordering EU
  countries + remote), posting age, and English-language requirement — before
  spending any API tokens.
- **Scores** every new job 0.0–1.0 against your `profile.md` using Claude, with a
  short reasoning string and skill match/gap analysis.
- **Generates** a tailored **cover letter** for any match above the threshold, and
  a tailored **one-page resume** for strong Tier A/B matches (score ≥ 70%).
- **Tracks** every job processed in `tracker.json` (and a SQLite mirror,
  `tracker.db`) so it never re-scores or re-bills a job it has already seen.
- **Visualises** everything in a zero-dependency local web **dashboard** with
  editable "Applied/Pending" status, notes, and analytics.

---

## How it works (pipeline flow)

```
                       companies.json (your target companies + ATS config)
                                  │
                                  ▼
        ┌──────────────────────────────────────────────────┐
        │  scraper.py   →  fetch open roles per company      │
        │  • routes by ATS (Greenhouse, Lever, Workday, …)   │
        │  • pre-filters: role keywords, location, age, lang │
        └──────────────────────────────────────────────────┘
                                  │  list of {title, description, url, …}
                                  ▼
        ┌──────────────────────────────────────────────────┐
        │  run_pipeline.py  (orchestrator)                   │
        │  • skips jobs already in tracker.json (no re-bill) │
        └──────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┼──────────────────────────┐
                ▼                 ▼                          ▼
        matcher.py         cover_letter.py              resume.py
      score 0.0–1.0      tailored cover letter      tailored 1-page resume
      vs profile.md      (Claude, per role)         (Tier A/B & score ≥ 0.70)
                │                 │                          │
                └─────────────────┼──────────────────────────┘
                                  ▼
                       applications/<Company>/*.md      (generated artefacts)
                       tracker.json + tracker.db        (history / status)
                                  │
                                  ▼
                          dashboard.py  →  http://127.0.0.1:8000
```

The Claude API is only ever called for **new** jobs that pass the cheap pre-filters
— already-seen jobs are skipped via the tracker, and below-threshold jobs are
recorded so they are not re-scored on the next run.

---

## Project structure

```
claude_implementation/
├── run_pipeline.py        # ▶ Main entry point — scrape, score, generate, log
├── scraper.py             # Fetches + filters jobs from all ATS backends
├── matcher.py             # Scores a job vs your profile with Claude → 0.0–1.0
├── cover_letter.py        # Generates tailored cover letters (+ cold emails)
├── resume.py              # Generates a tailored 1-page resume from your base CV
├── dashboard.py           # Local web dashboard over tracker.json / tracker.db
├── backfill_tracker.py    # One-off: fills date_posted / recruiter email blanks
│
├── companies.json         # Your target companies + ATS handles + fit tiers
├── profile.md             # YOUR profile (git-ignored — PII)        ← you create
├── profile.example.md     # Sanitised template for profile.md
├── requirements.txt       # Python dependencies
├── .env                   # ANTHROPIC_API_KEY (git-ignored)          ← you create
├── .env.example           # Template for .env
├── SETUP_GUIDE.md         # Step-by-step Windows setup walkthrough
│
├── BASE_RESUME/           # Drop your base CV here (.docx/.md/.txt)  (git-ignored)
├── applications/          # Generated cover letters & resumes        (git-ignored)
│   └── <Company>/cover_letter_<Role>.md
├── tracker.json           # Application history / status             (git-ignored)
└── tracker.db             # SQLite mirror used by the dashboard       (git-ignored)
```

---

## Prerequisites

- **Python 3.10+** (uses modern type hints like `list[dict]` and `str | None`)
- An **Anthropic API key** — get one at <https://console.anthropic.com>
- For Workday/proprietary portals: **Playwright** + a headless Chromium browser

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install the Playwright browser (needed for Workday / JS-heavy portals)
playwright install chromium

# 3. Add your API key
#    Copy the template and paste your real key into .env
cp .env.example .env
#    then edit .env  →  ANTHROPIC_API_KEY=sk-ant-api03-...

# 4. Create your profile
cp profile.example.md profile.md
#    then fill profile.md with your real details

# 5. (Optional) Drop your base CV into BASE_RESUME/ as .docx, .md, or .txt
```

> **Windows:** see [`SETUP_GUIDE.md`](SETUP_GUIDE.md) for a click-by-click
> walkthrough including Task Scheduler automation. You can use `copy` instead of
> `cp`. The API key can also be set as a system environment variable
> (`setx ANTHROPIC_API_KEY "..."`) instead of using `.env`.

---

## Configuration

| File | What you set |
|------|--------------|
| `.env` | Your `ANTHROPIC_API_KEY`. |
| `profile.md` | Your CV-in-markdown: skills, experience, projects, salary, work authorisation, standard form answers. Read on **every** run and injected into scoring/cover-letter/resume prompts. |
| `companies.json` | The list of target companies. Each entry has a `name`, `category`, `careers_url`, `ats` (which scraper to use), an `ats_handle`/board token, a `fit_tier` (A/B/C), and optional `"skip": true` to disable a broken portal. |
| `BASE_RESUME/` | Your base CV file used as the source for tailored resumes. |

Scraper behaviour (role keywords, target locations, max posting age, exclude
keywords) is configured at the top of [`scraper.py`](scraper.py).

---

## Usage

```bash
# Full run — all (non-skipped) companies
python run_pipeline.py

# Scrape + score only, no cover letters / resumes (cheap, no generation cost)
python run_pipeline.py --dry-run

# Only your best-fit (Tier A) companies
python run_pipeline.py --tier A

# Test a single company (runs even if marked "skip")
python run_pipeline.py --company "Snowflake"

# Raise the match threshold (only process 70%+ fits)
python run_pipeline.py --min-score 0.7
```

| Flag | Effect |
|------|--------|
| `--dry-run` | Scrape + score only; skip cover-letter & resume generation. |
| `--tier A\|B\|C` | Restrict to one fit tier. |
| `--company NAME` | Run a single company by (partial) name; honoured even if `skip`. |
| `--min-score 0.5` | Minimum fit score to generate artefacts (default `0.5`). |

Generated cover letters and resumes land in `applications/<Company>/`, and every
processed job is appended to `tracker.json`.

---

## The dashboard

A single-page web UI over your tracker — **standard library only**, no extra deps.

```bash
python dashboard.py                 # imports tracker.json → tracker.db, opens browser
python dashboard.py --port 9000     # custom port
python dashboard.py --no-browser    # don't auto-open the browser
```

It serves <http://127.0.0.1:8000> showing, per job: posted date, fit score,
pipeline status, application link, an editable **Applied/Pending** toggle, and
editable **notes** — plus analytics (totals, applied vs pending, score
distribution, top companies). Re-importing is safe: scraped fields refresh while
your own edits (status, notes) are preserved.

---

## Supported ATS / scrapers

`scrape_company()` in [`scraper.py`](scraper.py) routes each company to the right
backend based on its `ats` field:

| ATS | Function | Notes |
|-----|----------|-------|
| Greenhouse | `scrape_greenhouse` | JSON board API; also recovers original post date. |
| Lever | `scrape_lever` | Public postings API. |
| Ashby | `scrape_ashby` | GraphQL job board. |
| SmartRecruiters | `scrape_smartrecruiters` | Public postings API. |
| Recruitee | `scrape_recruitee` | Public offers API. |
| Workday | `scrape_workday` / `scrape_workday_cxs` | CxS JSON API + Playwright fallback. |
| Radancy / TalentBrew | `scrape_radancy` | Used by some large enterprises. |
| Amazon | `scrape_amazon` | Amazon's proprietary jobs API. |
| _anything else_ | `scrape_proprietary` | Generic Playwright/HTML fallback. |

Shared helpers filter by relevant role title (`is_relevant_role`), target location
(`_in_target_location`), posting age (`is_posted_within_days`), and English-language
requirement (`requires_english`), and extract a best-effort recruiter contact email
(`extract_spoc_email`).

---

## Module reference

- **`run_pipeline.py`** — Orchestrator and CLI. Loads profile + companies, walks
  each company → scrape → de-dupe against tracker → score → (cover letter +
  resume) → log. Prints a rich summary table.
- **`scraper.py`** — All scraping + filtering logic. Returns a normalised
  `{title, description, url, location, date_posted, ats}` list per company.
- **`matcher.py`** — `score_job()` calls Claude (`claude-sonnet-4-6`) with a strict
  JSON-only scoring prompt and returns `(score, reasoning)`.
- **`cover_letter.py`** — `generate_cover_letter()` writes a ≤350-word letter with
  company-appropriate tone; `generate_cold_email()` writes a short recruiter email.
- **`resume.py`** — Extracts text from your base `.docx` (via `python-docx`, with a
  stdlib XML fallback) and `generate_tailored_resume()` rewrites it for the role
  under strict one-page / no-fabrication constraints.
- **`dashboard.py`** — Imports the tracker into SQLite and serves the web UI.
- **`backfill_tracker.py`** — Idempotent one-off to fill missing `date_posted` /
  recruiter-email fields on existing tracker entries.

---

## Cost

Scoring uses `claude-sonnet-4-6` at roughly **$0.003 per job scored**. A full run
of ~57 companies × ~3 jobs ≈ **$0.50**, plus ~$0.10 for ~10 cover letters — about
**$0.60 per daily run**. Because the tracker skips already-seen jobs, steady-state
daily runs cost far less (only genuinely new postings are billed).

---

## Privacy, secrets & what is git-ignored

This repo is configured so **no secrets or personal data are ever committed**. The
[`.gitignore`](.gitignore) excludes:

- **`.env`** — your live `ANTHROPIC_API_KEY`.
- **`profile.md`** — your real PII (address, phone, visa, salary). Commit the
  sanitised `profile.example.md` instead.
- **`tracker.json` / `tracker.db`** — your application history & recruiter emails.
- **`applications/`** — every generated cover letter and tailored resume.
- **`BASE_RESUME/`** — your personal CV source files.
- `__pycache__/`, virtualenvs, Playwright output, and editor/OS noise.

When you clone fresh, copy the two `*.example` files, fill them in, and you're
ready — your private data stays local.

> **Security note:** if a real API key was ever committed or shared, rotate it at
> <https://console.anthropic.com> → API Keys. Keys are billed against your account.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ANTHROPIC_API_KEY not found` | Ensure `.env` exists with the key, or set the env var; reopen the terminal. |
| `playwright install chromium` fails | Run the terminal as Administrator. |
| A company returns 0 jobs | Its portal may have changed; open the URL manually. Proprietary portals are least reliable. |
| Cover letter sounds generic | Add more specific detail to `profile.md`'s summary/experience. |
| API rate-limit errors | Add a small `time.sleep()` between calls in `matcher.py`; free-tier keys have lower limits. |

See [`SETUP_GUIDE.md`](SETUP_GUIDE.md) for more.

---

## Disclaimer

This tool **drafts** applications — it does not auto-submit them. Always review
generated cover letters and resumes before sending. Respect each site's terms of
service and `robots.txt` when scraping, and keep request volumes polite (the
scraper already adds delays between requests).

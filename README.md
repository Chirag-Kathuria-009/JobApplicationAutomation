# Auto Job Applier — Claude-Powered Job Application Pipeline

An end-to-end pipeline that **finds, scores, and drafts applications** for data
roles across European company career portals. It scrapes jobs from many Applicant
Tracking Systems (ATS), uses **Claude (through the Claude Code CLI, no API key)** to score how well each role fits
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
- **Scores** every new job 0.0–1.0 against three resume-variant profiles
  (`profile_de.md` Data Engineer, `profile_ds.md` Data Science, `profile_ai.md`
  AI Engineer) in a single Claude call, keeps all three scores and the best one
  (`best_profile`), which also decides which base resume gets tailored.
- **Tier D (startups)** is sourced from job boards listed in
  `startup_sources.json` rather than a fixed company list. Only Startupjobs.de is
  currently scrapable; the other boards are recorded there as disabled with the
  reason (bot protection / login wall).
- **Generates** a tailored **cover letter** for any match above the threshold, and
  a tailored **one-page resume** for strong matches in any tier (score ≥ 70%), rendered as a one-page PDF.
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
      vs profile.md      (Claude, per role)         (score ≥ 0.70, any tier)
                │                 │                          │
                └─────────────────┼──────────────────────────┘
                                  ▼
                       applications/<Company>/*.md      (generated artefacts)
                       tracker.json + tracker.db        (history / status)
                                  │
                                  ▼
                          dashboard.py  →  http://127.0.0.1:8000
```

Claude is only ever called for **new** jobs that pass the cheap pre-filters
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
├── profile_{de,ds,ai}.md  # Per-resume positioning summaries (git-ignored) ← you create
├── profile_*.example.md   # Templates for the three summaries
├── startup_sources.json   # Tier D job-board sources (re-scraped every run)
├── requirements.txt       # Python dependencies
├── llm.py                 # The only place models are called (Claude Code CLI)
├── models.json            # Which model each task uses (score / write)
├── .env                   # optional: CLAUDE_CLI path (git-ignored)
├── .env.example           # Template for .env
├── SETUP_GUIDE.md         # Step-by-step Windows setup walkthrough
│
├── BASE_RESUME/base.json  # Structured base resume (bullet ids)           (git-ignored)
├── resume_render.py       # Fixed-layout one-page PDF renderer (Playwright/Chromium)
├── applications/          # Generated cover letters & resumes        (git-ignored)
│   └── <Company>/cover_letter_<Role>.md
├── tracker.json           # Application history / status             (git-ignored)
└── tracker.db             # SQLite mirror used by the dashboard       (git-ignored)
```

---

## Prerequisites

- **Python 3.10+** (uses modern type hints like `list[dict]` and `str | None`)
- The **Claude Code CLI**, installed and logged in with your Claude subscription (run `claude` once and log in). All model calls go through `claude -p`: **no Anthropic API key and no API billing**. Install on Windows (PowerShell): `irm https://claude.ai/install.ps1 | iex`.
- For Workday/proprietary portals: **Playwright** + a headless Chromium browser

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install the Playwright browser (needed for Workday / JS-heavy portals)
playwright install chromium

# 3. Make sure the Claude Code CLI works from your terminal
claude --version          # installed?  then run `claude` once and log in
#    Optional: if `claude` is not on PATH, copy .env.example to .env and set CLAUDE_CLI=<full path to claude.exe>

# 4. Create your profile
cp profile.example.md profile.md
#    then fill profile.md with your real details

# 5. (Optional) Drop your base CV into BASE_RESUME/ as .docx, .md, or .txt
```

> **Windows:** see [`SETUP_GUIDE.md`](SETUP_GUIDE.md) for a click-by-click
> walkthrough including Task Scheduler automation. You can use `copy` instead of
> `cp`. Do not set `ANTHROPIC_API_KEY`: with a key present, Claude Code would bill
> the API instead of your subscription (the pipeline removes it from every call
> and warns if it finds one).

---

## Configuration

| File | What you set |
|------|--------------|
| `.env` | Optional: `CLAUDE_CLI` (full path to `claude.exe`) if it is not on PATH. |
| `models.json` | Model per task: `score` (profile matching) and `write` (resume + cover letter), e.g. `"model": "sonnet"` or `"haiku"`. Changing the scoring model shifts scores, so re-check the 0.60 cutoff. |
| `profile.md` | Your CV-in-markdown: skills, experience, projects, salary, work authorisation, standard form answers. Read on **every** run and injected into scoring/cover-letter/resume prompts. |
| `companies.json` | The list of target companies. Each entry has a `name`, `category`, `careers_url`, `ats` (which scraper to use), an `ats_handle`/board token, a `fit_tier` (A/B/C), and optional `"skip": true` to disable a broken portal. |
| `profile_de.md` / `profile_ds.md` / `profile_ai.md` | Short positioning summary per resume variant; every job is scored against all three. Missing files are skipped; if none exist, `profile.md` is used as a single profile. |
| `startup_sources.json` | Tier D startup job boards (feeds, enabled flag, notes on why a board is disabled). |
| `BASE_RESUME/base.json` | Your resume as structured JSON (git-ignored): every bullet has an id, plus summaries, projects, skill groups and per-profile hints. **Review it by hand once** — it is the only source of resume text. Add a tool to `skills`/`extra_allowed_terms` only if you can back it with experience. |

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

# Only Tier D startup boards, or one board
python run_pipeline.py --tier D
python run_pipeline.py --source startupjobs

# Raise the match threshold (only process 70%+ fits)
python run_pipeline.py --min-score 0.7
```

| Flag | Effect |
|------|--------|
| `--dry-run` | Scrape + score only; skip cover-letter & resume generation. |
| `--tier A\|B\|C\|D` | Restrict to one fit tier (`D` = startup boards). |
| `--source NAME` | Run one Tier D board channel (e.g. `startupjobs`). |
| `--company NAME` | Run a single company by (partial) name; honoured even if `skip`. |
| `--min-score 0.6` | Minimum **best-profile** fit score to generate artefacts (default `0.6`). Jobs scoring 0.50–0.65 are also appended to `near_threshold_log.jsonl` so you can spot-check the cutoff — it is a starting point, not a calibrated value. |

**Freshness:** "new to me" comes from `tracker.json` de-duplication, not an age
cutoff. Each tracker entry gets a `first_seen` timestamp and the dashboard sorts
newest-first on it. The only age filter left is the loose `MAX_AGE_DAYS` (7)
safety net in `scraper.py`; the run summary prints how many postings it dropped.

Generated cover letters and resumes land in `applications/<Company>/`, and every
processed job is appended to `tracker.json`.

**Failure recovery and duplicates.** A job is saved to `tracker.json` the moment
it is scored and again after each finished step (cover letter, resume); saves are
atomic and keep a `.bak`. A job with unfinished steps has status `in_progress`
and keeps its job text, so the next run finishes it first, without re-scoring and
without redoing a finished step, even if the posting is gone. Each step is tried
at most 3 times across runs, then the job ends as `resume_failed` or
`needs_attention`. If Claude is not logged in, your usage limit is reached, or 3 calls in a row
fail, the run stops with exit code 2 and the rest stays pending (`last_run.json`
records why). Files already on disk are reused. A lock file stops two runs
overlapping (exit code 3). `python run_pipeline.py --pending-only` finishes
unfinished jobs without scraping. Duplicates (`dedupe.py`) are matched by ATS
posting id, URL without tracking parameters, company + title (ignoring GmbH/AG
and (m/w/d)), or identical job text under the same company; the same role in
several cities counts as one application. Tests: `python tests/test_recovery.py`.

**Sitemap careers sites (BMW).** Sites whose search is drawn by JavaScript but which publish a sitemap (BMW Group, on SAP SuccessFactors) use `"ats": "Sitemap"`: the scraper reads the sitemap and the public job pages only (never the `/services/` API, which BMW's robots.txt disallows), keeps data-role titles at German sites, and applies the language rule below locally (no model tokens).

**Language rule (all companies and boards).** A posting is kept when English is what is needed, or the posting is simply written in English, and German is at most optional ("a plus", "nice to have", "von Vorteil", "not required"). It is dropped when German is required ("fluent German", "verhandlungssicher", "B2") or is mentioned without being called optional. A posting written in German is kept only if it says English is needed and German is optional. Text too short to judge (placeholder descriptions) passes. The decision and its reason are saved on the tracker entry as `language`. This is plain text rules, so unusual phrasings can be misjudged either way; `tests/test_language.py` lists the covered cases.

**Tailored resume PDFs.** The instructions the model follows are your own resume prompt, kept in `resume_prompt.md` (edit it there; no code change needed), plus STAR-bullet and no-invention rules. The PDF sections are always Summary, Skills, Experience, Projects, Education, on one page; the audit file also lists the job's technologies that are NOT on your resume, any skills cut for space and any bullets dropped to fit. The model rewrites your base resume for the job
(summary, skill order, STAR-style bullets, the 3 most relevant projects) and
returns JSON; fixed facts (titles, companies, dates, education) come from
`base.json`. Code then rejects the rewrite (one retry, then `resume_failed`) if a
number isn't in the original entry, a technology from `tech_vocabulary.json` isn't
already in your base resume, a role is missing, or the project count is wrong;
skills your base resume can't back up are removed. The result is rendered to a
fixed one-page layout, with an `.audit.json` beside the PDF showing original vs
rewritten bullets. Jobs with a PDF get status `pending_review`; if tailoring
fails the job keeps its cover letter and `resume_error` is set. The model sees
only the role, requirements and company sections of the posting (`jd_cleaner.py`
drops benefits and legal text). The check cannot catch a rewrite that merely
sounds stronger than the facts, so read the PDF before sending. One-time setup:
`playwright install chromium`. Test on saved job descriptions:
`python resume.py --test jd1.txt jd2.txt --profile DE`.

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

## Cost and usage limits

There is no per-call cost: every model call runs through the Claude Code CLI on your
Claude subscription, so what matters is your plan's **usage limit**, not money. Per
new job the pipeline makes 1 scoring call, and for matches (score >= 0.60) 1 cover
letter call and (score >= 0.70) 1 resume call; each takes roughly 3-13 seconds. The
tracker and duplicate check mean only genuinely new jobs use any of it. If the limit
is reached mid-run, the run stops cleanly and the next run continues where it left
off. Schedule it early in the day, before you use Claude for other work.

---

## Privacy, secrets & what is git-ignored

This repo is configured so **no secrets or personal data are ever committed**. The
[`.gitignore`](.gitignore) excludes:

- **`.env`** — local settings such as `CLAUDE_CLI`.
- **`profile.md`** — your real PII (address, phone, visa, salary). Commit the
  sanitised `profile.example.md` instead.
- **`tracker.json` / `tracker.db`** — your application history & recruiter emails.
- **`applications/`** — every generated cover letter and tailored resume.
- **`BASE_RESUME/`** — your personal CV source files.
- `__pycache__/`, virtualenvs, Playwright output, and editor/OS noise.

When you clone fresh, copy the two `*.example` files, fill them in, and you're
ready — your private data stays local.

> **Security note:** the pipeline needs no API key. If an old Anthropic API key was ever
> committed or shared, rotate or delete it at <https://console.anthropic.com> → API Keys.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Claude CLI not usable: login problem` | Run `claude` in a terminal and log in again (same Windows user as the scheduled task). |
| `Claude CLI not usable: usage limit` | Your plan's limit is used up; the run stops, nothing is lost, and the next run continues. Schedule the run before you use Claude for other work. |
| `Claude Code CLI not found` | Install it, or set `CLAUDE_CLI` in `.env` to the full path of `claude.exe`. |
| `playwright install chromium` fails | Run the terminal as Administrator. |
| A company returns 0 jobs | Its portal may have changed; open the URL manually. Proprietary portals are least reliable. |
| Cover letter sounds generic | Add more specific detail to `profile.md`'s summary/experience. |
| Runs are slow | Each model call takes about 3-13 s through the CLI: 1 call to score, then for matches 1 cover letter and 1 resume. |

See [`SETUP_GUIDE.md`](SETUP_GUIDE.md) for more.

---

## Disclaimer

This tool **drafts** applications — it does not auto-submit them. Always review
generated cover letters and resumes before sending. Respect each site's terms of
service and `robots.txt` when scraping, and keep request volumes polite (the
scraper already adds delays between requests).

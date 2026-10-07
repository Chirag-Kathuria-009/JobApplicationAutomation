# Setup Guide — Chirag's Job Application Pipeline
## Windows + Claude Code

Complete this once. After setup, the pipeline runs with a single command.

---

## What you'll have after setup

```
job_pipeline/
├── run_pipeline.py       ← Main script — run this daily
├── scraper.py            ← Fetches jobs from all 57 portals
├── matcher.py            ← Scores jobs with Claude API
├── cover_letter.py       ← Generates cover letters with Claude API
├── profile.md            ← YOUR profile — keep this updated
├── companies.json        ← Your 57 companies with fit tiers
├── tracker.json          ← Auto-created — logs every application
└── applications/         ← Auto-created — one folder per company
    ├── N26/
    │   └── cover_letter_Data_Engineer.md
    ├── Zalando/
    │   └── cover_letter_Analytics_Engineer.md
    └── ...
```

---

## Step 1: Install the Claude Code CLI and log in (5 minutes)

The pipeline uses your **Claude subscription** through the Claude Code command line. It needs **no API key** and has no per-call billing.

1. Open **PowerShell** and run: `irm https://claude.ai/install.ps1 | iex`
2. Open a new terminal and run `claude` once. Log in with the account that has your Claude plan, then exit.
3. Check it: `claude --version`

---

## Step 2: Make sure no API key is set

With `ANTHROPIC_API_KEY` set, Claude Code bills the API instead of your subscription. The pipeline removes it from every call, but it is cleaner not to have one:

```cmd
echo %ANTHROPIC_API_KEY%
```

This should print nothing. If the pipeline cannot find `claude`, copy `.env.example` to `.env` and set `CLAUDE_CLI` to the full path of `claude.exe`.

---

## Step 3: Install Python dependencies

Open terminal in the `job_pipeline` folder and run:

```cmd
pip install -r requirements.txt
```

Then install the Playwright browser:

```cmd
playwright install chromium
```

This downloads a ~130MB headless browser. Only needed once.

---

## Step 4: Verify setup

Run a quick test to confirm everything is connected:

```cmd
python run_pipeline.py --dry-run --tier A --company N26
```

You should see:
```
→ N26 (Banking & Finance, Tier A)
  Found 2 job(s)
  🟢 [82%] Data Engineer
  🟢 [75%] Analytics Engineer
  (dry run — no cover letters generated)
```

If you see this: setup is complete.

---

## Step 5: Full first run

Run all Tier A companies first (your best-fit targets):

```cmd
python run_pipeline.py --tier A
```

This will:
- Scan all 11 Tier A companies
- Score every job found
- Generate cover letters for matches above 50%
- Save everything to `applications/` folder
- Log all results to `tracker.json`

Expected time: 5–10 minutes

Then run Tier B:

```cmd
python run_pipeline.py --tier B
```

---

## Step 6: Schedule it to run daily (Windows Task Scheduler)

So it runs automatically every morning at 8am without you touching anything:

1. Open **Task Scheduler** (search in Windows Start)
2. Click **Create Basic Task**
3. Name: `Chirag Job Pipeline`
4. Trigger: **Daily** at 8:00 AM
5. Action: **Start a program**
6. Program: `python`
7. Arguments: `C:\path\to\job_pipeline\run_pipeline.py --tier A`
8. Start in: `C:\path\to\job_pipeline\`
9. Click Finish

Now it runs every morning. You check `tracker.json` and the `applications/` folder when you want.

---

## Daily workflow (5–10 minutes)

```
Morning check (5 min):
  → Open tracker.json or applications/ folder
  → See which new cover letters were generated
  → Review the cover letter (edit if needed)
  → Visit the job URL and apply (Simplify Copilot autofills the form)
  → Paste the cover letter into the form

That's it. The pipeline does everything else.
```

---

## Common commands

```cmd
# Run all companies (full run)
python run_pipeline.py

# Only best-fit companies (Tier A)
python run_pipeline.py --tier A

# Test one company without applying
python run_pipeline.py --dry-run --company Snowflake

# Higher match threshold (70%+)
python run_pipeline.py --min-score 0.7

# Test a single company
python run_pipeline.py --company "Trade Republic"
```

---

## Troubleshooting

**"Claude CLI not usable"**
→ Run `claude` in a terminal and log in again (as the same Windows user that runs the scheduled task). If it says usage limit, wait for the reset; nothing is lost and the next run continues.

**"playwright install chromium" fails**
→ Run Command Prompt as Administrator

**Scraper returns 0 jobs for a company**
→ That company may have changed their portal. Open the URL manually and check.
→ Proprietary portals (Google, Amazon, SAP) are least reliable — check manually.

**Cover letter sounds generic**
→ Open `profile.md` and add more specific details to your Professional Summary section.

**API rate limit error**
→ Add `time.sleep(2)` in `matcher.py` between calls. Free tier has rate limits.

---

## Updating your profile

When anything changes (new project, new certification, availability, salary):
1. Open `profile.md`
2. Edit the relevant section
3. Save
4. All future cover letters will use the updated version automatically

---

## Cost estimate

| Activity | Cost per run |
|---|---|
| Scoring 57 companies × 3 jobs avg | ~$0.50 |
| Cover letters for 10 matches | ~$0.10 |
| **Total per daily run** | **~$0.60** |
| Monthly (daily runs) | ~$18/month |

Your $5 free credit covers the first ~8 full runs (about 1 week).
After that: $10 API credit lasts ~2 weeks.

---

## Files to never delete

- `profile.md` — your entire application profile
- `companies.json` — your 57 company configs
- `tracker.json` — your complete application history

Back these up to Google Drive or GitHub (private repo).

---

## Questions?

Come back to this Claude chat. I have your full profile and company list in context.

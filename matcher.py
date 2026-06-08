"""
job_pipeline/matcher.py
========================
Uses Claude API to score each job against Chirag's profile.
Returns a float (0.0 to 1.0) and a short reasoning string.

Cost estimate: ~$0.003 per job scored (claude-sonnet-4-6).
For 60 companies × ~3 jobs each = ~$0.54 per full run.
"""

import os
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

# Initialise client — reads ANTHROPIC_API_KEY from environment
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SCORING_SYSTEM_PROMPT = """You are a job fit analyser. Your job is to compare a candidate's profile against a job description and return a precise fit score.

You must respond ONLY with valid JSON — no preamble, no markdown, no explanation outside the JSON.

Response format:
{
  "score": 0.0,
  "tier": "A/B/C/D",
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "reasoning": "2-3 sentence summary",
  "apply": true
}

Scoring guide:
- 0.9–1.0: Near-perfect match. Candidate has 90%+ of required skills and relevant domain experience.
- 0.7–0.9: Strong match. Candidate has core skills, some gaps that are learnable.
- 0.5–0.7: Moderate match. Candidate meets minimum requirements, noticeable gaps.
- 0.3–0.5: Weak match. Significant skill gaps or wrong domain.
- 0.0–0.3: Poor match. Wrong profile entirely.

Set "apply": true if score >= 0.5."""


def score_job(profile: str, job_title: str, job_description: str) -> tuple[float, str]:
    """
    Score a job against the candidate profile.

    Returns:
        (score: float, reasoning: str)
    """
    user_message = f"""Candidate profile:
{profile[:4000]}

---

Job title: {job_title}

Job description:
{job_description[:3000]}

---

Analyse the fit and return JSON only."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=SCORING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )

        raw = response.content[0].text.strip()

        # Strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        score = float(result.get("score", 0.0))
        reasoning = result.get("reasoning", "")
        return score, reasoning

    except json.JSONDecodeError:
        # If Claude returns non-JSON for some reason, give a low score
        return 0.0, "Scoring failed — could not parse response"
    except anthropic.APIError as e:
        raise RuntimeError(f"Claude API error during scoring: {e}")


def batch_score_jobs(profile: str, jobs: list[dict]) -> list[dict]:
    """
    Score a list of jobs and return them sorted by score descending.
    Adds 'fit_score' and 'fit_reasoning' keys to each job dict.
    """
    scored = []
    for job in jobs:
        score, reasoning = score_job(profile, job["title"], job["description"])
        job["fit_score"] = score
        job["fit_reasoning"] = reasoning
        scored.append(job)

    # Sort by score descending
    return sorted(scored, key=lambda x: x["fit_score"], reverse=True)

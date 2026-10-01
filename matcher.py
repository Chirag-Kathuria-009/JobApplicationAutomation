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


MULTI_SCORING_SYSTEM_PROMPT = """You are a job fit analyser. You are given several candidate profile variants (each is a different positioning of the same person, e.g. Data Engineer / Data Scientist / AI Engineer) and one job description. Score the fit of EACH profile variant against the job independently.

You must respond ONLY with valid JSON — no preamble, no markdown, no explanation outside the JSON.

Response format (one entry in "scores" per profile key provided, using exactly those keys):
{
  "scores": {"KEY1": 0.0, "KEY2": 0.0},
  "reasoning": "2-3 sentence summary focused on the best-fitting profile"
}

Scoring guide (per profile):
- 0.9–1.0: Near-perfect match. Candidate has 90%+ of required skills and relevant domain experience.
- 0.7–0.9: Strong match. Candidate has core skills, some gaps that are learnable.
- 0.5–0.7: Moderate match. Candidate meets minimum requirements, noticeable gaps.
- 0.3–0.5: Weak match. Significant skill gaps or wrong domain.
- 0.0–0.3: Poor match. Wrong profile entirely."""


def _parse_json_response(raw: str) -> dict:
    """Parse a JSON object from a model reply, tolerating markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def score_job_multi(profiles: dict[str, str], job_title: str,
                    job_description: str) -> dict:
    """
    Score a job against several profile variants in ONE model call.

    Args:
        profiles: {"DE": summary_text, "DS": ..., "AI": ...}

    Returns:
        {"scores": {"DE": float, ...}, "best_profile": "DE", "best_score": float,
         "reasoning": str}
    best_profile / best_score are computed here (not trusted from the model).
    On a parse failure every score is 0.0.
    """
    keys = list(profiles)
    profile_block = "\n\n".join(
        f"=== PROFILE {k} ===\n{text[:2500]}" for k, text in profiles.items()
    )
    user_message = f"""Profile keys to score: {', '.join(keys)}

{profile_block}

---

Job title: {job_title}

Job description:
{job_description[:3000]}

---

Return JSON only."""

    zero = {"scores": {k: 0.0 for k in keys}, "best_profile": keys[0],
            "best_score": 0.0, "reasoning": "Scoring failed — could not parse response"}
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=MULTI_SCORING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        result = _parse_json_response(response.content[0].text)
        raw_scores = result.get("scores") or {}
        scores = {k: max(0.0, min(1.0, float(raw_scores.get(k, 0.0)))) for k in keys}
    except (json.JSONDecodeError, TypeError, ValueError):
        return zero
    except anthropic.APIError as e:
        raise RuntimeError(f"Claude API error during scoring: {e}")

    # max() keeps the first key on ties, so profile order sets tie-break priority.
    best = max(keys, key=lambda k: scores[k])
    return {"scores": scores, "best_profile": best, "best_score": scores[best],
            "reasoning": result.get("reasoning", "")}


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

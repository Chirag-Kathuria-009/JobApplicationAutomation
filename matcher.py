"""
job_pipeline/matcher.py
========================
Scores each job against every profile variant (DE / DS / AI) in ONE model call,
through llm.call_model (Claude Code CLI on the owner's subscription - no API key).
Returns a score per profile (0.0 to 1.0), the best profile and a short reason.
"""

from llm import call_model, CliError

MULTI_SCORING_SYSTEM_PROMPT = """You are a job fit analyser. You are given several candidate profile variants (each is a different positioning of the same person, e.g. Data Engineer / Data Scientist / AI Engineer) and one job description. Score the fit of EACH profile variant against the job independently.

Score only from the text provided. Do not assume skills the candidate lacks.

Scoring guide (per profile, 0.0 to 1.0):
- 0.9-1.0: Near-perfect match. Candidate has 90%+ of required skills and relevant domain experience.
- 0.7-0.9: Strong match. Candidate has core skills, some gaps that are learnable.
- 0.5-0.7: Moderate match. Candidate meets minimum requirements, noticeable gaps.
- 0.3-0.5: Weak match. Significant skill gaps or wrong domain.
- 0.0-0.3: Poor match. Wrong profile entirely.

Give one score per profile key provided, using exactly those keys, and a 2-3 sentence reason focused on the best-fitting profile."""


def _schema(keys: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {k: {"type": "number", "minimum": 0, "maximum": 1} for k in keys},
                "required": keys,
            },
            "reasoning": {"type": "string"},
        },
        "required": ["scores", "reasoning"],
    }


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
    A reply that cannot be used returns every score 0.0 with "failed": True. A
    CliError (login problem, usage limit, repeated CLI failure) is raised.
    """
    keys = list(profiles)
    profile_block = "\n\n".join(
        f"=== PROFILE {k} ===\n{text[:2500]}" for k, text in profiles.items()
    )
    prompt = f"""Profile keys to score: {', '.join(keys)}

{profile_block}

---

Job title: {job_title}

Job description:
{job_description[:3000]}"""

    zero = {"scores": {k: 0.0 for k in keys}, "best_profile": keys[0],
            "best_score": 0.0, "reasoning": "Scoring failed — could not use the reply",
            "failed": True}   # callers must NOT record this job as scored (it would never be retried)
    result, _generated_by = call_model("score", prompt, schema=_schema(keys),
                                       system=MULTI_SCORING_SYSTEM_PROMPT)
    try:
        raw_scores = result["scores"]
        scores = {k: max(0.0, min(1.0, float(raw_scores[k]))) for k in keys}
    except (KeyError, TypeError, ValueError):
        return zero

    # max() keeps the first key on ties, so profile order sets tie-break priority.
    best = max(keys, key=lambda k: scores[k])
    return {"scores": scores, "best_profile": best, "best_score": scores[best],
            "reasoning": result.get("reasoning", "")}

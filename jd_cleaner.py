"""
job_pipeline/jd_cleaner.py
==========================
Trim a job description to the parts that matter for scoring and resume
tailoring, in this priority order:
  1. the role / tasks / responsibilities
  2. the required skills / profile
  3. the "about the company" section
Benefits, perks, legal / equal-opportunity text and "how to apply" sections are
dropped. Works on text where headings sit on their own line (scraper.html_to_text
produces that). Text with no recognisable headings is just truncated.
"""

import re

# Phrases are matched against a normalised (lowercase, letters-only) heading line.
ROLE_HEADINGS = [
    "your role", "the role", "about the role", "role", "your tasks", "tasks", "your responsibilities",
    "responsibilities", "key responsibilities", "what you will do", "what youll do", "what you'll do",
    "what you do", "your mission", "your impact", "in this role", "job description", "the job",
    "day to day", "your day to day", "the opportunity", "about the opportunity", "what you will be doing",
    "aufgaben", "deine aufgaben", "ihre aufgaben", "dein job", "deine rolle", "ihre rolle", "das erwartet dich",
    "das erwartet sie",
    "key job responsibilities", "job responsibilities", "a day in the life", "what you will be doing",
    "your responsibilities include", "what to expect",
]
SKILL_HEADINGS = [
    "your profile", "profile", "requirements", "your qualifications", "qualifications", "what you bring",
    "what you need", "who you are", "skills", "required skills", "must have", "must haves", "nice to have",
    "what were looking for", "what we're looking for", "what we are looking for", "your background",
    "about you", "you bring", "minimum qualifications", "preferred qualifications", "basic qualifications",
    "anforderungen", "dein profil", "ihr profil", "qualifikationen", "das bringst du mit", "das bringen sie mit",
    "dein wissen", "was du mitbringst", "was sie mitbringen",
    "what you ll need", "what youll need", "your skills", "technical skills", "skills and experience",
    "basic requirements", "minimum requirements", "preferred skills", "desired skills", "ideal candidate",
]
COMPANY_HEADINGS = [
    "about us", "about the company", "who we are", "the company", "our company", "our mission", "company",
    "uber uns", "über uns", "wer wir sind", "das unternehmen", "unser unternehmen",
]
DROP_HEADINGS = [
    "benefits", "our benefits", "what we offer", "we offer", "perks", "why join", "why you will love",
    "why youll love", "why you'll love", "compensation", "salary", "our offer", "what you get",
    "equal opportunity", "equal opportunities", "diversity", "diversity and inclusion", "privacy",
    "data protection", "how to apply", "application process", "our hiring process", "interview process",
    "next steps", "legal", "about the team", "additional information",
    "das bieten wir", "wir bieten", "deine benefits", "unsere benefits", "bewerbung", "bewerbungsprozess",
    "datenschutz", "das bieten wir dir", "was wir bieten",
]

_KINDS = (("drop", DROP_HEADINGS), ("role", ROLE_HEADINGS), ("skills", SKILL_HEADINGS),
          ("company", COMPANY_HEADINGS))

# Share of the character budget each priority section may use (company gets the rest).
_CAPS = {"role": 0.50, "skills": 0.40}


def _normalise(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zäöüß' ]", " ", line.lower())).strip()


def _heading_kind(line: str) -> str | None:
    """Return 'role' | 'skills' | 'company' | 'drop' | 'other' for a heading line, None if not a heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None
    norm = _normalise(stripped)
    words = norm.split()
    if not words or len(words) > 8:
        return None
    ends_colon = stripped.endswith(":")
    for kind, phrases in _KINDS:
        for ph in phrases:
            phn = _normalise(ph)
            multi_word = " " in phn      # bare words ("role", "skills") only match exactly
            if norm == phn \
                    or (multi_word and len(words) <= 6 and norm.startswith(phn + " ")) \
                    or (ends_colon and f" {phn} " in f" {norm} "):
                return kind
    # An unrecognised short line ending in ':' is still a heading — just not one we rank.
    if ends_colon and len(words) <= 6:
        return "other"
    return None


def clean_job_description(text: str, limit: int = 4000) -> str:
    """Return the role, skills and company parts of `text` (benefits etc. removed), <= limit chars."""
    text = (text or "").strip()
    if not text:
        return ""

    sections: dict[str, list[str]] = {"company": [], "role": [], "skills": [], "other": []}
    current = "company"          # text before the first heading is normally the company intro
    found_heading = False
    for line in text.splitlines():
        kind = _heading_kind(line)
        if kind is not None:
            found_heading = True
            current = kind
            continue
        if current != "drop" and line.strip():
            sections[current].append(line.strip())

    if not found_heading:
        return text[:limit]

    labels = {"role": "RESPONSIBILITIES", "skills": "REQUIREMENTS", "company": "ABOUT THE COMPANY",
              "other": "OTHER"}
    parts, remaining = [], limit
    for kind in ("role", "skills", "company", "other"):
        body = "\n".join(sections[kind])
        if not body or remaining <= 0:
            continue
        cap = int(limit * _CAPS[kind]) if kind in _CAPS else remaining
        body = body[:min(cap, remaining)]
        parts.append(f"{labels[kind]}:\n{body}")
        remaining -= len(body)
    cleaned = "\n\n".join(parts)
    # Safety net: a posting whose only recognised heading was a drop section.
    return cleaned if cleaned else text[:limit]

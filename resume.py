"""
job_pipeline/resume.py
======================
Generates a job-tailored one-page resume PDF.

The model does NOT write resume text. It only chooses from the owner's
pre-written content in BASE_RESUME/base.json (every bullet has an id):
  - which pre-written summary to use,
  - which bullets to show per role / project, and in what order,
  - which projects to include,
  - the order of the skill groups,
  - which terms to bold.
Code validates every choice against the base file (unknown ids or terms are
rejected, with one retry), assembles the resume from the base text unchanged,
and resume_render.py lays it out as a fixed-format PDF. So nothing in the PDF
can be invented by the model, and the layout is identical for every job.

Only invoked for strong matches (see run_pipeline.py): best-profile score >= 70%.

Manual test on saved job descriptions (needs ANTHROPIC_API_KEY):
    python resume.py --test jd1.txt jd2.txt --profile DE --company Acme
"""

import argparse
import json
import os
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
BASE_RESUME_DIR = BASE_DIR / "BASE_RESUME"
BASE_JSON_PATH = BASE_RESUME_DIR / "base.json"

MODEL = "claude-sonnet-4-6"
MAX_BOLD_TERMS = 15


class ResumeError(Exception):
    """Tailoring failed (bad model output after a retry, or render failure)."""


_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


RESUME_SYSTEM_PROMPT = """You are an expert resume editor for Data Engineering / Data Science / AI Engineering roles.

You are given the candidate's resume content as JSON (every bullet has an id) and a job description. You do NOT write resume text. You only CHOOSE and ORDER existing content, so the resume is tailored to the job.

-----------------------
HARD CONSTRAINTS
-----------------------
- Use ONLY ids that appear in the provided JSON. Never invent, edit or merge bullets.
- Every role in "experience" must appear (do not remove roles); choose and order its bullets.
- The final resume must fit ONE page: choose 1-3 projects, and at most 4 bullets for the main role, 2-3 for the others, 1-3 per project.
- Order bullets and projects MOST RELEVANT FIRST. The last bullet of each list is dropped first if space runs out.
- No fake experience. Maintain realistic, explainable claims.

-----------------------
WHAT TO CHOOSE
-----------------------
1. summary_id: the pre-written summary that best fits the job.
2. experience: for every role id, the ordered bullet ids to show.
3. projects: ordered project ids most relevant to the job; project_bullets: ordered bullet ids per chosen project.
4. skills_order: ordered skill-group ids, most relevant to the job first. Leave out groups that are irrelevant (keep at least 4). The "languages" group is always appended automatically.
5. bold_terms: up to 15 skills/tools that appear in the job description AND in allowed_terms, for ATS keyword emphasis. Use them exactly as written in allowed_terms.

-----------------------
OUTPUT FORMAT
-----------------------
Respond ONLY with valid JSON, no markdown, no commentary:
{
  "summary_id": "de",
  "experience": {"exp_bajaj": ["b_etl", "b_dbx"], "exp_werkstudent": ["w_pbi"]},
  "projects": ["proj_dora", "proj_agent"],
  "project_bullets": {"proj_dora": ["d_stream"], "proj_agent": ["a_guard"]},
  "skills_order": ["de", "prog"],
  "bold_terms": ["Airflow", "Python"]
}"""


# --------------------------------------------------------------------------- #
# Base resume
# --------------------------------------------------------------------------- #
def load_base(path: Path | None = None) -> dict | None:
    """Load and sanity-check BASE_RESUME/base.json. Returns None if the file is absent."""
    path = Path(path or BASE_JSON_PATH)
    if not path.exists():
        return None
    base = json.loads(path.read_text(encoding="utf-8"))

    ids: list[str] = []
    for sec in ("experience", "projects"):
        for entry in base.get(sec, []):
            ids.append(entry["id"])
            ids += [b["id"] for b in entry["bullets"]]
    ids += [s["id"] for s in base.get("skills", [])]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"{path.name}: duplicate ids {sorted(dupes)}")
    for key in ("name", "contact", "summaries", "experience", "education", "skills", "profiles"):
        if not base.get(key):
            raise ValueError(f"{path.name}: missing '{key}'")
    return base


def _expand_term(term: str) -> set[str]:
    """Lowercased variants of a skill/tool name: 'Apache Spark/PySpark' ->
    {'apache spark/pyspark', 'apache spark', 'pyspark', 'spark'};
    'Vector Databases (pgvector)' -> {..., 'vector databases', 'pgvector'}."""
    out: set[str] = set()
    t = term.strip().lower()
    out.add(t)
    inner = re.findall(r"\(([^)]*)\)", t)
    outer = re.sub(r"\([^)]*\)", "", t).strip()
    parts = [outer] + [p for chunk in inner for p in chunk.split(",")]
    for p in parts:
        for q in p.split("/"):
            q = q.strip()
            if q:
                out.add(q)
                if q.startswith("apache "):
                    out.add(q[len("apache "):])
    return out


def allowed_terms(base: dict) -> set[str]:
    """Terms the resume may emphasise: skill items + project tech + extras (lowercased).
    Derived from the base file so it can't drift. Snowflake etc. are absent
    unless the owner adds them to base.json."""
    terms: set[str] = set()
    for group in base["skills"]:
        if group.get("bold_exempt"):
            continue
        for item in group["items"]:
            terms |= _expand_term(item)
    for proj in base.get("projects", []):
        for item in proj.get("tech", "").split("·"):
            if item.strip():
                terms |= _expand_term(item)
    for item in base.get("extra_allowed_terms", []):
        terms |= _expand_term(item)
    return terms


# --------------------------------------------------------------------------- #
# Validation + assembly (enforced in code, not only in the prompt)
# --------------------------------------------------------------------------- #
def validate_choices(base: dict, choices) -> list[str]:
    """Return a list of problems with the model's choices ([] if valid)."""
    if not isinstance(choices, dict):
        return ["response is not a JSON object"]
    errs: list[str] = []

    if choices.get("summary_id") not in base["summaries"]:
        errs.append(f"summary_id must be one of {sorted(base['summaries'])}")

    exp_by_id = {e["id"]: e for e in base["experience"]}
    chosen_exp = choices.get("experience")
    if not isinstance(chosen_exp, dict):
        errs.append("experience must be an object {role_id: [bullet_ids]}")
    else:
        for rid in exp_by_id:
            if rid not in chosen_exp:
                errs.append(f"experience is missing role '{rid}' (roles may not be removed)")
        for rid, bids in chosen_exp.items():
            if rid not in exp_by_id:
                errs.append(f"unknown role id '{rid}'")
                continue
            errs += _check_bullet_ids(rid, bids, exp_by_id[rid])

    proj_by_id = {p["id"]: p for p in base.get("projects", [])}
    chosen_proj = choices.get("projects")
    chosen_pb = choices.get("project_bullets")
    if not isinstance(chosen_proj, list) or not 1 <= len(chosen_proj) <= 4:
        errs.append("projects must be a list of 1-4 project ids")
    else:
        if len(set(chosen_proj)) != len(chosen_proj):
            errs.append("projects contains duplicates")
        if not isinstance(chosen_pb, dict):
            errs.append("project_bullets must be an object {project_id: [bullet_ids]}")
            chosen_pb = {}
        for pid in chosen_proj:
            if pid not in proj_by_id:
                errs.append(f"unknown project id '{pid}'")
            else:
                errs += _check_bullet_ids(pid, chosen_pb.get(pid), proj_by_id[pid])

    group_ids = {s["id"] for s in base["skills"]}
    order = choices.get("skills_order")
    if not isinstance(order, list):
        errs.append("skills_order must be a list of skill-group ids")
    else:
        unknown = [g for g in order if g not in group_ids]
        if unknown:
            errs.append(f"unknown skill group ids {unknown}")
        if len(set(order)) != len(order):
            errs.append("skills_order contains duplicates")
        if len([g for g in order if g in group_ids]) < 3:
            errs.append("skills_order must keep at least 3 groups")

    bold = choices.get("bold_terms", [])
    if not isinstance(bold, list) or len(bold) > MAX_BOLD_TERMS:
        errs.append(f"bold_terms must be a list of at most {MAX_BOLD_TERMS} terms")
    else:
        allowed = allowed_terms(base)
        bad = [t for t in bold if not isinstance(t, str) or t.strip().lower() not in allowed]
        if bad:
            errs.append(f"bold_terms not in allowed terms: {bad}")
    return errs


def _check_bullet_ids(owner: str, bids, entry: dict) -> list[str]:
    valid = {b["id"] for b in entry["bullets"]}
    if not isinstance(bids, list) or not bids:
        return [f"'{owner}' needs a non-empty list of bullet ids"]
    errs = []
    unknown = [b for b in bids if b not in valid]
    if unknown:
        errs.append(f"'{owner}': unknown bullet ids {unknown} (valid: {sorted(valid)})")
    if len(set(bids)) != len(bids):
        errs.append(f"'{owner}': duplicate bullet ids")
    return errs


def assemble_resume(base: dict, choices: dict) -> dict:
    """Map validated choices onto the base file -> the flat dict resume_render expects.
    All text is copied from base.json unchanged."""
    text = {b["id"]: b["text"] for sec in ("experience", "projects")
            for e in base[sec] for b in e["bullets"]}

    experience = []
    for e in base["experience"]:   # base order (reverse-chronological) is kept
        experience.append({**{k: e[k] for k in ("role", "company", "location", "dates")},
                           "bullets": [text[i] for i in choices["experience"][e["id"]]]})

    proj_by_id = {p["id"]: p for p in base.get("projects", [])}
    projects = []
    for pid in choices["projects"]:
        p = proj_by_id[pid]
        projects.append({"title": p["title"], "dates": p.get("dates", ""), "tech": p.get("tech", ""),
                         "bullets": [text[i] for i in choices["project_bullets"][pid]]})

    by_id = {s["id"]: s for s in base["skills"]}
    order = [g for g in choices["skills_order"] if not by_id[g].get("always_last")]
    order += [s["id"] for s in base["skills"] if s.get("always_last")]
    skills = [{"label": by_id[g]["label"], "items": by_id[g].get("sep", " · ").join(by_id[g]["items"])}
              for g in order]

    return {
        "name": base["name"], "contact": base["contact"], "work_auth": base.get("work_auth", ""),
        "summary": base["summaries"][choices["summary_id"]],
        "experience": experience, "projects": projects,
        "education": base["education"], "skills": skills,
        "bold_terms": choices.get("bold_terms", []),
    }


# --------------------------------------------------------------------------- #
# Model call
# --------------------------------------------------------------------------- #
def _base_for_prompt(base: dict, profile: str) -> str:
    hint = base["profiles"].get(profile.upper(), {})
    payload = {
        "profile_hint": {"matched_profile": profile.upper(), **hint},
        "summaries": base["summaries"],
        "experience": [{"id": e["id"], "role": e["role"], "company": e["company"],
                        "bullets": e["bullets"]} for e in base["experience"]],
        "projects": [{"id": p["id"], "title": p["title"], "tech": p.get("tech", ""),
                      "bullets": p["bullets"]} for p in base.get("projects", [])],
        "skills": [{"id": s["id"], "label": s["label"], "items": s["items"]}
                   for s in base["skills"] if not s.get("always_last")],
        "allowed_terms": sorted(allowed_terms(base)),
    }
    return json.dumps(payload, ensure_ascii=False)


def _parse_json(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start:end + 1] if start != -1 and end > start else raw)


def _call_model(user_message: str) -> str:
    response = _get_client().messages.create(
        model=MODEL, max_tokens=1500, system=RESUME_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def select_content(base: dict, profile: str, job_title: str, job_description: str,
                   call_model=_call_model) -> dict:
    """Ask the model for choices, validate them, retry once with the errors.
    Raises ResumeError if the second attempt is also invalid."""
    message = (f"JOB TITLE: {job_title}\n\nJOB DESCRIPTION:\n{job_description[:4000]}\n\n"
               f"RESUME CONTENT (JSON):\n{_base_for_prompt(base, profile)}")
    problems: list[str] = []
    for attempt in (1, 2):
        user = message if attempt == 1 else (
            message + "\n\nYour previous answer was rejected for these reasons:\n- "
            + "\n- ".join(problems) + "\nReturn corrected JSON only.")
        try:
            choices = _parse_json(call_model(user))
        except (json.JSONDecodeError, ValueError) as e:
            problems = [f"not valid JSON ({e})"]
            continue
        problems = validate_choices(base, choices)
        if not problems:
            return choices
    raise ResumeError("invalid model choices after retry: " + "; ".join(problems))


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def _slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:limit]


def generate_tailored_resume(base: dict, profile: str, company_name: str, job_title: str,
                             job_description: str, out_dir: Path,
                             call_model=_call_model) -> dict:
    """Choose content, validate, assemble, render a one-page PDF into out_dir.

    Returns {"pdf_path": str, "choices_path": str}. Raises ResumeError on failure.
    The model's choices are saved next to the PDF so every selection is auditable.
    """
    from resume_render import render_resume_pdf

    choices = select_content(base, profile, job_title, job_description, call_model)
    resume = assemble_resume(base, choices)

    stem = f"{_slug(base['name'].replace(' ', '_'), 30)}_Resume_{_slug(company_name, 30)}_{_slug(job_title)}"
    out_dir = Path(out_dir)
    pdf_path = out_dir / f"{stem}.pdf"
    choices_path = out_dir / f"{stem}.choices.json"
    try:
        render_resume_pdf(resume, pdf_path)
    except Exception as e:
        raise ResumeError(f"render failed: {e}") from e
    out_dir.mkdir(parents=True, exist_ok=True)
    choices_path.write_text(json.dumps(
        {"company": company_name, "job_title": job_title, "profile": profile, "choices": choices},
        indent=2, ensure_ascii=False), encoding="utf-8")
    return {"pdf_path": str(pdf_path), "choices_path": str(choices_path)}


def check_pdf(pdf_path: str, base: dict) -> dict:
    """Test helper: page count, text extractable, and how many base bullets appear in the PDF."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    squash = lambda s: re.sub(r"\s+", "", s)
    pdf_text = squash("".join(p.extract_text() or "" for p in reader.pages))
    bullets = [b["text"] for sec in ("experience", "projects") for e in base[sec] for b in e["bullets"]]
    return {"pages": len(reader.pages), "text_chars": len(pdf_text),
            "bullets_found": sum(squash(b) in pdf_text for b in bullets)}


def _main():
    ap = argparse.ArgumentParser(description="Render tailored resume PDFs from saved job descriptions.")
    ap.add_argument("--test", nargs="+", required=True, metavar="JD.txt", help="job description text files")
    ap.add_argument("--profile", default="DE", choices=["DE", "DS", "AI"])
    ap.add_argument("--company", default="TestCo")
    args = ap.parse_args()

    base = load_base()
    if base is None:
        raise SystemExit(f"{BASE_JSON_PATH} not found")
    out_dir = BASE_DIR / "applications" / "_resume_test"
    for jd_file in args.test:
        jd = Path(jd_file).read_text(encoding="utf-8")
        title = Path(jd_file).stem.replace("_", " ")
        try:
            res = generate_tailored_resume(base, args.profile, args.company, title, jd, out_dir)
        except ResumeError as e:
            print(f"{jd_file}: FAILED — {e}")
            continue
        print(f"{jd_file}: {res['pdf_path']}  {check_pdf(res['pdf_path'], base)}")


if __name__ == "__main__":
    _main()

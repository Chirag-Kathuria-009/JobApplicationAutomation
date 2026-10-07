"""
job_pipeline/resume.py
======================
Generates a job-tailored one-page resume PDF.

The model rewrites the owner's resume for the job (summary, skills order,
STAR-style bullets, choice of 3 projects) and returns it as JSON in the shape of
BASE_RESUME/base.json. Fixed facts (name, contact, role titles, companies,
dates, project titles/tech, education) are taken from base.json by code and
cannot be changed by the model. Rewritten text is then checked in code:
  - every number must already appear in the original entry (no invented metrics),
  - no technology from tech_vocabulary.json may appear unless it is already in
    the base resume (no invented tools),
  - skills the base resume cannot back up are removed,
  - all roles are present, exactly 3 projects (or all if fewer), length limits.
Invalid output is retried once with the reasons, then the job is marked
resume_failed. resume_render.py lays the result out as a fixed-format PDF, and
an audit file (original vs rewritten bullets) is saved beside it.

Only invoked for strong matches (see run_pipeline.py): best-profile score >= 70%.

The model is reached through llm.call_model (Claude Code CLI on the owner's
subscription - no API key).

Manual test on saved job descriptions (needs the Claude Code CLI logged in):
    python resume.py --test jd1.txt jd2.txt --profile DE --company Acme
"""

import argparse
import json
import re
from pathlib import Path

import llm
from jd_cleaner import clean_job_description

BASE_DIR = Path(__file__).parent
BASE_RESUME_DIR = BASE_DIR / "BASE_RESUME"
BASE_JSON_PATH = BASE_RESUME_DIR / "base.json"
TECH_VOCAB_PATH = BASE_DIR / "tech_vocabulary.json"

PROJECTS_SHOWN = 3
MAX_BOLD_TERMS = 15
MAX_BULLET_WORDS = 50
MAX_SUMMARY_WORDS = 85
JD_CHARS_FOR_RESUME = 4000


class ResumeError(Exception):
    """Tailoring failed (invalid model output after a retry, or render failure)."""


RESUME_SYSTEM_PROMPT = """You are an expert resume editor specializing in Data Engineering / Data Science / AI Engineering roles.

I will provide:
1. My base resume (as JSON)
2. A job description (responsibilities, requirements and company context only)

Your task is to tailor my resume specifically for this job while following STRICT constraints:

-----------------------
HARD CONSTRAINTS
-----------------------
- Keep the overall structure EXACTLY the same (Summary, Skills, Experience, Projects, Education).
- Do NOT add new sections. Do NOT remove any experience role.
- Show exactly 3 projects (the most relevant to the job, most relevant first), as my individual resumes do.
- Resume must remain within ONE PAGE: keep text concise, at most 4 bullets for the most recent role, 2-3 for others, 2-3 per project.
- Avoid fluff or generic statements.
- No fake experience - only reframe and optimize existing content.
- NEVER invent a number, metric, tool, technology, employer, certification or business impact. Every number in a bullet must already appear in that entry's original bullets. Do NOT mention any technology that is not already in my base resume, even if the job description asks for it (a skill I lack must simply be left out).
- Maintain realistic, explainable claims I can justify in an interview.

-----------------------
OBJECTIVE
-----------------------
Maximize alignment with the job description, improve ATS keyword matching, highlight the most relevant skills and experience, and make everything easy to justify in interviews.

-----------------------
WHAT TO MODIFY
-----------------------
1. SUMMARY
- Rewrite to align with the job role; 3-4 lines (max ~70 words); impact + specialization.
- Include key tools/skills from the JD ONLY if they are already in my base resume. No keyword stuffing.

2. SKILLS
- Reorder groups and items by relevance to the JD; group logically; remove less relevant items.
- Add a skill ONLY if it is already mentioned in my base resume (bullets, projects or skills).
- Do NOT include a "Languages" group (it is added automatically).

3. EXPERIENCE AND PROJECTS - STAR-STYLE BULLETS
- Rewrite every bullet as ONE short sentence (max ~30 words) in STAR form: the problem/situation, the steps I took (with the tools), and the outcome/achievement. For example: "<problem or goal>; <what I did, with tools>, <result with its metric>."
- Use only facts already in the original bullet. If the original states no problem, or no measurable outcome, leave that part out rather than inventing it.
- Put the most relevant bullets first. Use the JD's terminology where it truthfully describes what I did.

4. KEYWORD ALIGNMENT
- Naturally include JD keywords that I genuinely have; avoid unnatural stuffing.

-----------------------
OUTPUT FORMAT
-----------------------
Respond ONLY with valid JSON (no markdown, no explanation, no comments):
{
  "summary": "...",
  "skills": [{"label": "Programming", "items": ["Python", "SQL"]}],
  "experience": {"<role id>": ["bullet 1", "bullet 2"]},
  "projects": [{"id": "<project id>", "bullets": ["bullet 1", "bullet 2"]}]
}
"experience" must contain every role id from the base resume. "projects" must contain exactly 3 project ids from the base resume."""


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
    """Skill items + project tech + extras, lowercased (derived from base.json so it can't drift)."""
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


def _all_base_text(base: dict) -> str:
    """Everything the owner has written (lowercased) - the universe of claims the model may reuse."""
    parts = list(base["summaries"].values())
    for sec in ("experience", "projects"):
        for e in base[sec]:
            parts += [b["text"] for b in e["bullets"]] + [e.get("tech", "")]
    for s in base["skills"]:
        parts += s["items"]
    parts += base.get("extra_allowed_terms", [])
    return "\n".join(parts).lower()


# --------------------------------------------------------------------------- #
# Checks on the model's rewrite (enforced in code, not only in the prompt)
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUM.findall(text)}


def _load_vocab() -> list[str]:
    if TECH_VOCAB_PATH.exists():
        return json.loads(TECH_VOCAB_PATH.read_text(encoding="utf-8"))
    return []


def _mentions(term: str, text: str) -> bool:
    """Whole-word, case-insensitive occurrence of `term` in `text`."""
    return re.search(rf"(?<![\w+#.]){re.escape(term)}(?![\w+#])", text, flags=re.I) is not None


def unowned_tools(text: str, base_text: str, owned: set[str], vocab: list[str]) -> list[str]:
    """Technologies named in `text` (from tech_vocabulary.json) that the base resume does not contain."""
    return [t for t in vocab
            if _mentions(t, text) and t.lower() not in owned and not _mentions(t, base_text)]


def _skill_item_ok(item: str, base_text: str, owned: set[str], vocab: list[str]) -> bool:
    """A skill may stay only if the base resume already backs it up and it is not a known foreign tool."""
    if unowned_tools(item, base_text, owned, vocab):
        return False
    return bool(_expand_term(item) & owned) or any(v in base_text for v in _expand_term(item))


def validate_tailored(base: dict, out) -> tuple[list[str], dict, list[str]]:
    """Check the model's JSON. Returns (errors, cleaned_output, removed_skills).

    Skills the base can't back up are removed (not an error). Anything else
    wrong - missing roles, wrong project count, invented numbers or tools,
    over-long text - is an error and triggers the retry.
    """
    if not isinstance(out, dict):
        return ["response is not a JSON object"], {}, []
    errs: list[str] = []
    base_text = _all_base_text(base)
    owned = allowed_terms(base)
    vocab = _load_vocab()
    all_numbers = _numbers(base_text)

    # summary
    summary = out.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        errs.append("summary missing")
        summary = ""
    else:
        if len(summary.split()) > MAX_SUMMARY_WORDS:
            errs.append(f"summary is over {MAX_SUMMARY_WORDS} words")
        extra = _numbers(summary) - all_numbers
        if extra:
            errs.append(f"summary contains numbers not in the base resume: {sorted(extra)}")
        tools = unowned_tools(summary, base_text, owned, vocab)
        if tools:
            errs.append(f"summary mentions technologies not in the base resume: {tools}")

    # experience
    exp_out = out.get("experience")
    exp_clean: dict[str, list[str]] = {}
    if not isinstance(exp_out, dict):
        errs.append('"experience" must be an object {role id: [bullets]}')
    else:
        for e in base["experience"]:
            bullets = exp_out.get(e["id"])
            if not isinstance(bullets, list) or not bullets or not all(isinstance(b, str) for b in bullets):
                errs.append(f"role '{e['id']}' needs a non-empty list of bullets (roles may not be removed)")
                continue
            errs += _check_bullets(e["id"], bullets, [b["text"] for b in e["bullets"]], base_text, owned, vocab)
            exp_clean[e["id"]] = [b.strip() for b in bullets]
        unknown = [k for k in exp_out if k not in {e["id"] for e in base["experience"]}]
        if unknown:
            errs.append(f"unknown role ids {unknown}")

    # projects
    proj_by_id = {p["id"]: p for p in base.get("projects", [])}
    want = min(PROJECTS_SHOWN, len(proj_by_id))
    proj_out = out.get("projects")
    proj_clean: list[dict] = []
    if not isinstance(proj_out, list) or len(proj_out) != want:
        errs.append(f'"projects" must be a list of exactly {want} projects')
    else:
        seen = set()
        for p in proj_out:
            pid = p.get("id") if isinstance(p, dict) else None
            if pid not in proj_by_id or pid in seen:
                errs.append(f"invalid or duplicate project id {pid!r}")
                continue
            seen.add(pid)
            bullets = p.get("bullets")
            if not isinstance(bullets, list) or not 1 <= len(bullets) <= 4 or not all(isinstance(b, str) for b in bullets):
                errs.append(f"project '{pid}' needs 1-4 bullets")
                continue
            errs += _check_bullets(pid, bullets, [b["text"] for b in proj_by_id[pid]["bullets"]], base_text, owned, vocab)
            proj_clean.append({"id": pid, "bullets": [b.strip() for b in bullets]})

    # skills: unbacked items are removed, not rejected
    skills_clean, removed = [], []
    skills_out = out.get("skills")
    if not isinstance(skills_out, list):
        errs.append('"skills" must be a list of {label, items}')
    else:
        for g in skills_out:
            if not isinstance(g, dict) or not isinstance(g.get("items"), list) or not g.get("label"):
                continue
            if g["label"].strip().lower() == "languages":
                continue    # added by code
            keep = []
            for item in g["items"]:
                if isinstance(item, str) and _skill_item_ok(item, base_text, owned, vocab):
                    keep.append(item)
                else:
                    removed.append(str(item))
            if keep:
                skills_clean.append({"label": str(g["label"]), "items": keep})
        if len(skills_clean) < 3:
            errs.append("skills must keep at least 3 groups of skills that exist in the base resume")

    cleaned = {"summary": summary.strip(), "skills": skills_clean,
               "experience": exp_clean, "projects": proj_clean}
    return errs, cleaned, removed


def _check_bullets(owner: str, bullets: list[str], originals: list[str], base_text: str,
                   owned: set[str], vocab: list[str]) -> list[str]:
    errs = []
    allowed_nums = _numbers(" ".join(originals))
    for b in bullets:
        if len(b.split()) > MAX_BULLET_WORDS:
            errs.append(f"'{owner}': a bullet is over {MAX_BULLET_WORDS} words: {b[:60]!r}")
        extra = _numbers(b) - allowed_nums
        if extra:
            errs.append(f"'{owner}': numbers not in the original entry {sorted(extra)} in: {b[:60]!r}")
        tools = unowned_tools(b, base_text, owned, vocab)
        if tools:
            errs.append(f"'{owner}': technologies not in the base resume {tools} in: {b[:60]!r}")
    return errs


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _display_terms(base: dict) -> list[str]:
    """Canonical names of owned skills/tools, used to pick which words to bold."""
    names: set[str] = set()
    for g in base["skills"]:
        if g.get("bold_exempt"):
            continue
        for item in g["items"]:
            main = re.sub(r"\([^)]*\)", "", item).strip()
            for q in main.split("/"):
                q = q.strip()
                if len(q) > 1:
                    names.add(q)
                    if q.lower().startswith("apache "):
                        names.add(q[7:])
    for p in base.get("projects", []):
        for item in p.get("tech", "").split("·"):
            if len(item.strip()) > 1:
                names.add(item.strip())
    names |= {x for x in base.get("extra_allowed_terms", []) if len(x) > 1}
    return sorted(names)


def pick_bold_terms(base: dict, jd_text: str, resume: dict) -> list[str]:
    """Owned terms that appear in BOTH the job text and the tailored resume text (ATS emphasis)."""
    body = " ".join([resume["summary"]] + [b for sec in ("experience", "projects")
                                            for e in resume[sec] for b in e["bullets"]])
    hits = [t for t in _display_terms(base) if _mentions(t, jd_text) and _mentions(t, body)]
    hits.sort(key=lambda t: (-len(t), t.lower()))
    return hits[:MAX_BOLD_TERMS]


def assemble_resume(base: dict, cleaned: dict) -> dict:
    """Combine the model's validated rewrite with the fixed facts in base.json -> flat dict for the renderer."""
    experience = [{**{k: e[k] for k in ("role", "company", "location", "dates")},
                   "bullets": cleaned["experience"][e["id"]]} for e in base["experience"]]
    proj_by_id = {p["id"]: p for p in base["projects"]}
    projects = [{"title": proj_by_id[p["id"]]["title"], "dates": proj_by_id[p["id"]].get("dates", ""),
                 "tech": proj_by_id[p["id"]].get("tech", ""), "bullets": p["bullets"]}
                for p in cleaned["projects"]]
    skills = [{"label": g["label"], "items": " · ".join(g["items"])} for g in cleaned["skills"]]
    for s in base["skills"]:
        if s.get("always_last"):
            skills.append({"label": s["label"], "items": s.get("sep", " · ").join(s["items"])})
    return {"name": base["name"], "contact": base["contact"], "work_auth": base.get("work_auth", ""),
            "summary": cleaned["summary"], "experience": experience, "projects": projects,
            "education": base["education"], "skills": skills, "bold_terms": []}


# --------------------------------------------------------------------------- #
# Model call
# --------------------------------------------------------------------------- #
def _base_for_prompt(base: dict, profile: str) -> str:
    hint = base["profiles"].get(profile.upper(), {})
    payload = {
        "matched_profile": profile.upper(),
        "reference_summary_for_this_profile": base["summaries"].get(hint.get("summary_id", ""), ""),
        "default_projects_for_this_profile": hint.get("default_projects", []),
        "default_skill_group_order_for_this_profile": hint.get("skills_order", []),
        "experience": [{"id": e["id"], "role": e["role"], "company": e["company"],
                       "bullets": [b["text"] for b in e["bullets"]]} for e in base["experience"]],
        "projects": [{"id": p["id"], "title": p["title"], "tech": p.get("tech", ""),
                      "bullets": [b["text"] for b in p["bullets"]]} for p in base["projects"]],
        "skills": [{"label": s["label"], "items": s["items"]} for s in base["skills"] if not s.get("always_last")],
    }
    return json.dumps(payload, ensure_ascii=False, indent=1)


def _parse_json(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start:end + 1] if start != -1 and end > start else raw)


_STR_LIST = {"type": "array", "items": {"type": "string"}}
RESUME_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "skills": {"type": "array", "items": {
            "type": "object",
            "properties": {"label": {"type": "string"}, "items": _STR_LIST},
            "required": ["label", "items"]}},
        "experience": {"type": "object", "additionalProperties": _STR_LIST},
        "projects": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "bullets": _STR_LIST},
            "required": ["id", "bullets"]}},
    },
    "required": ["summary", "skills", "experience", "projects"],
}


def _call_model(user_message: str) -> str:
    """One CLI call; returns the structured reply as JSON text (validated by the caller)."""
    structured, _generated_by = llm.call_model("write", user_message, schema=RESUME_OUTPUT_SCHEMA,
                                                system=RESUME_SYSTEM_PROMPT)
    return json.dumps(structured, ensure_ascii=False)


def rewrite_resume(base: dict, profile: str, job_title: str, jd_clean: str,
                   call_model=_call_model) -> tuple[dict, dict, list[str]]:
    """Ask the model for the rewrite, validate, retry once with the reasons.

    Returns (raw_model_output, cleaned_output, removed_skills). Raises ResumeError
    if the second attempt is still invalid.
    """
    message = (f"JOB TITLE: {job_title}\n\nJOB DESCRIPTION:\n{jd_clean}\n\n"
               f"BASE RESUME (JSON):\n{_base_for_prompt(base, profile)}")
    problems: list[str] = []
    for attempt in (1, 2):
        user = message if attempt == 1 else (
            message + "\n\nYour previous answer was rejected for these reasons:\n- "
            + "\n- ".join(problems) + "\nReturn corrected JSON only.")
        try:
            raw = _parse_json(call_model(user))
        except (json.JSONDecodeError, ValueError) as e:
            problems = [f"not valid JSON ({e})"]
            continue
        problems, cleaned, removed = validate_tailored(base, raw)
        if not problems:
            return raw, cleaned, removed
    raise ResumeError("invalid model output after retry: " + "; ".join(problems[:6]))


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def _slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:limit]


def resume_paths(base: dict, company_name: str, job_title: str, out_dir: Path) -> tuple[Path, Path]:
    """Deterministic (pdf, audit) paths for one application, so a finished resume is found again."""
    stem = f"{_slug(base['name'].replace(' ', '_'), 30)}_Resume_{_slug(company_name, 30)}_{_slug(job_title)}"
    return Path(out_dir) / f"{stem}.pdf", Path(out_dir) / f"{stem}.audit.json"


def existing_resume(base: dict, company_name: str, job_title: str, out_dir: Path) -> dict | None:
    """The finished resume for this application if both files exist (never regenerate it)."""
    pdf_path, audit_path = resume_paths(base, company_name, job_title, out_dir)
    if pdf_path.exists() and pdf_path.stat().st_size > 0 and audit_path.exists():
        return {"pdf_path": str(pdf_path), "audit_path": str(audit_path)}
    return None


def generate_tailored_resume(base: dict, profile: str, company_name: str, job_title: str,
                             job_description: str, out_dir: Path,
                             call_model=_call_model) -> dict:
    """Rewrite, validate, assemble and render a one-page PDF into out_dir.

    Returns {"pdf_path", "audit_path"}. Raises ResumeError on failure. The audit
    file records the model output, removed skills and original-vs-rewritten bullets.
    """
    from resume_render import render_resume_pdf

    jd_clean = clean_job_description(job_description, JD_CHARS_FOR_RESUME)
    raw, cleaned, removed = rewrite_resume(base, profile, job_title, jd_clean, call_model)
    resume = assemble_resume(base, cleaned)
    resume["bold_terms"] = pick_bold_terms(base, jd_clean, resume)

    out_dir = Path(out_dir)
    pdf_path, audit_path = resume_paths(base, company_name, job_title, out_dir)
    try:
        render_resume_pdf(resume, pdf_path)
    except Exception as e:
        raise ResumeError(f"render failed: {e}") from e

    originals = {e["id"]: [b["text"] for b in e["bullets"]] for sec in ("experience", "projects") for e in base[sec]}
    audit = {
        "company": company_name, "job_title": job_title, "profile": profile,
        "job_text_used": jd_clean, "removed_skills": removed, "bold_terms": resume["bold_terms"],
        "summary": cleaned["summary"],
        "experience": {rid: {"original": originals[rid], "rewritten": b} for rid, b in cleaned["experience"].items()},
        "projects": {p["id"]: {"original": originals[p["id"]], "rewritten": p["bullets"]} for p in cleaned["projects"]},
        "model_output": raw,
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"pdf_path": str(pdf_path), "audit_path": str(audit_path)}


def check_pdf(pdf_path: str, resume_bullets: list[str]) -> dict:
    """Test helper: page count, extractable text, and how many assembled bullets appear in the PDF."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    squash = lambda s: re.sub(r"\s+", "", s)
    pdf_text = squash("".join(p.extract_text() or "" for p in reader.pages))
    return {"pages": len(reader.pages), "text_chars": len(pdf_text),
            "bullets_found": sum(squash(b) in pdf_text for b in resume_bullets),
            "bullets_total": len(resume_bullets)}


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
            print(f"{jd_file}: FAILED - {e}")
            continue
        audit = json.loads(Path(res["audit_path"]).read_text(encoding="utf-8"))
        bullets = [b for sec in ("experience", "projects") for v in audit[sec].values() for b in v["rewritten"]]
        print(f"{jd_file}: {res['pdf_path']}  {check_pdf(res['pdf_path'], bullets)}  "
              f"removed skills: {audit['removed_skills']}")


if __name__ == "__main__":
    _main()

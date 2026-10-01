"""
job_pipeline/resume.py
======================
Generates a job-tailored version of the base resume using the Claude API.
Only invoked for strong matches (see run_pipeline.py) — roles in any tier scoring
>= 70%. The base resume lives in the BASE_RESUME/ folder (.docx preferred).

Output is saved as Markdown under applications/<Company>/, mirroring how cover
letters are stored.
"""

import os
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import anthropic
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

BASE_DIR = Path(__file__).parent
BASE_RESUME_DIR = BASE_DIR / "BASE_RESUME"

# The user's resume-editing prompt, used verbatim as the system instruction.
RESUME_SYSTEM_PROMPT = """You are an expert resume editor specializing in Data Engineering / Data Science roles.

I will provide:
1. My base resume
2. A job description

Your task is to tailor my resume specifically for this job while following STRICT constraints:

-----------------------
HARD CONSTRAINTS
-----------------------
- Keep the overall structure EXACTLY the same (Summary, Skills, Experience, Projects, Education).
- Do NOT add new sections.
- Do NOT remove existing roles or projects.
- Resume must remain within ONE PAGE.
- Avoid adding fluff or generic statements.
- No fake experience — only reframe and optimize existing content.
- Maintain realistic, explainable claims aligned with my background.

-----------------------
OBJECTIVE
-----------------------
Optimize my resume to:
- Maximize alignment with the job description
- Improve ATS keyword matching
- Highlight most relevant skills and experience
- Make it easy to justify everything in interviews

-----------------------
WHAT TO MODIFY
-----------------------

1. SUMMARY SECTION
- Rewrite to align strongly with the job role
- Include key tools/skills from JD (without keyword stuffing)
- Keep it concise (3–4 lines max)
- Focus on impact + specialization

2. SKILLS SECTION
- Reorder skills based on relevance to JD
- Add missing skills ONLY if they can be justified from experience/projects
- Group skills logically (e.g., Programming, ML, Cloud, BI)
- Remove less relevant skills if needed

3. EXPERIENCE SECTION
- Rephrase bullet points to reflect:
  - Impact (metrics-driven)
  - Relevance to JD (tools, techniques, domain)
- Prioritize most relevant bullets at top
- Adjust wording to match JD terminology
- Highlight:
  - Data pipelines / ML / analytics / BI depending on role
- Keep 4–6 bullets per role max

4. PROJECTS SECTION
- Emphasize projects most relevant to job
- Reorder projects if needed
- Adjust descriptions to highlight:
  - tools from JD
  - business impact
  - ML/analytics/system design relevance

5. KEYWORD ALIGNMENT
- Ensure important keywords from JD are naturally included:
  (e.g., ETL, ML models, Power BI, SQL, pipelines, etc.)
- Avoid unnatural keyword stuffing

6. SPACE OPTIMIZATION
- Ensure no excessive white space
- Keep content dense but readable
- Use concise bullet points

-----------------------
OUTPUT FORMAT
-----------------------
- Return FULL updated resume
- Keep formatting clean and consistent
- Do NOT explain changes
- Do NOT add comments
- Only output final resume"""


def _docx_to_text(path: Path) -> str:
    """Extract plain text from a .docx. Uses python-docx if available, else
    falls back to parsing word/document.xml directly (stdlib only)."""
    try:
        import docx  # python-docx, if installed
        document = docx.Document(str(path))
        parts = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append("\t".join(c.text for c in row.cells))
        return "\n".join(p for p in parts if p.strip())
    except ImportError:
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        with zipfile.ZipFile(path) as z:
            root = ET.fromstring(z.read("word/document.xml"))
        lines = []
        for para in root.iter(ns + "p"):
            line = "".join(t.text for t in para.iter(ns + "t") if t.text).strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

def _pdf_to_text(path: Path) -> str:
    """Extract plain text from a .pdf (needs the optional `pypdf` package)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"  [resume] pypdf not installed — cannot read {path.name} "
              f"(pip install pypdf)")
        return ""
    return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)


def _read_resume_file(path: Path) -> str:
    if path.suffix == ".docx":
        return _docx_to_text(path)
    if path.suffix == ".pdf":
        return _pdf_to_text(path)
    return path.read_text(encoding="utf-8")


# Filename tokens that mark a base resume as belonging to a profile. A file name
# is split on non-alphanumerics and lowercased, so e.g.
#   ChiragKathuria_Resume_June26_DE.pdf  -> {"chiragkathuria","resume","june26","de"}
#   ChiragKathuria_Resume_DataScience.pdf -> contains "datascience"
#   ChiragKathuria_Resume_AI_Engineer.pdf -> contains "ai"
PROFILE_RESUME_TOKENS = {
    "DE": {"de"},
    "DS": {"ds", "datascience"},
    "AI": {"ai"},
}
_RESUME_SUFFIXES = (".docx", ".md", ".txt", ".pdf")


def _resume_files() -> list[Path]:
    """Candidate base-resume files, preferred formats first."""
    if not BASE_RESUME_DIR.exists():
        return []
    files = []
    for suffix in _RESUME_SUFFIXES:
        files += [p for p in sorted(BASE_RESUME_DIR.glob(f"*{suffix}"))
                  if not p.name.startswith("~$")]  # skip Word lock files
    return files


def load_base_resume(profile: str | None = None) -> str:
    """Load base resume text from the BASE_RESUME/ folder.

    With `profile` ("DE" / "DS" / "AI"): only a file whose name carries that
    profile's token (see PROFILE_RESUME_TOKENS) is considered; returns "" if
    there is none. With no profile: the first readable resume (.docx preferred,
    then .md/.txt/.pdf), as before. Returns "" if nothing is found.
    """
    for path in _resume_files():
        if profile:
            tokens = set(re.split(r"[^a-z0-9]+", path.stem.lower()))
            if not tokens & PROFILE_RESUME_TOKENS.get(profile.upper(), set()):
                continue
        text = _read_resume_file(path)
        if text.strip():
            return text
    return ""


def load_base_resumes(profile_keys) -> dict[str, str]:
    """Load one base resume per profile key.

    A profile with no dedicated file falls back to the generic first resume
    (the pre-multi-profile behaviour), so a single-resume setup keeps working.
    Keys with no resume at all are omitted.
    """
    generic = None
    resumes = {}
    for key in profile_keys:
        text = load_base_resume(key)
        if not text:
            if generic is None:
                generic = load_base_resume()
            text = generic
        if text:
            resumes[key] = text
    return resumes


def generate_tailored_resume(base_resume: str, company_name: str, job_title: str,
                             job_description: str) -> str:
    """Return a job-tailored version of the base resume (Markdown text)."""
    user_message = f"""JOB DESCRIPTION:
{job_description[:4000]}

BASE RESUME:
{base_resume[:6000]}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        system=RESUME_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    resume_body = response.content[0].text.strip()
    today = datetime.now().strftime("%B %d, %Y")

    # Light metadata header for file traceability (mirrors cover letters).
    return (f"<!-- Tailored resume | {company_name} — {job_title} | "
            f"Generated {today} -->\n\n{resume_body}\n")

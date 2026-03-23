import re
import json
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("Run: pip install pymupdf")

try:
    import anthropic
except ImportError:
    raise ImportError("Run: pip install anthropic")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PDF TEXT EXTRACTION  (PyMuPDF)
# ══════════════════════════════════════════════════════════════════════════════

def extract_text(pdf_path: str) -> str:
    """
    Extract text from PDF.
    Uses 'blocks' mode to preserve reading order even for multi-column layouts.
    Falls back to simple text mode if block sorting seems off.
    """
    doc = fitz.open(pdf_path)
    all_text = []

    for page in doc:
        # Get text blocks sorted top-to-bottom, left-to-right
        blocks = page.get_text("blocks", sort=True)
        page_lines = []
        for block in blocks:
            # block = (x0, y0, x1, y1, "text", block_no, block_type)
            if block[6] == 0:  # text block (not image)
                text = block[4].strip()
                if text:
                    page_lines.append(text)
        all_text.append("\n".join(page_lines))

    doc.close()
    raw = "\n\n".join(all_text)

    # Light cleanup — remove excessive blank lines, fix encoding artifacts
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = raw.replace("\uf0b7", "•").replace("\u2022", "•")  # bullet variants
    return raw.strip()


def extract_text_with_metadata(pdf_path: str) -> dict:
    """Also return page count and any embedded metadata."""
    doc = fitz.open(pdf_path)
    meta = doc.metadata
    page_count = len(doc)
    doc.close()
    return {
        "text": extract_text(pdf_path),
        "page_count": page_count,
        "pdf_title": meta.get("title", ""),
        "pdf_author": meta.get("author", ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CLAUDE API PARSING
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert resume parser. Extract structured information from the resume text provided.

Return ONLY a valid JSON object — no markdown fences, no explanation, no preamble.

The JSON must follow this exact schema:
{
  "contact": {
    "name": "Full Name",
    "email": "email@example.com",
    "phone": "+1 555 123 4567",
    "location": "City, Country",
    "linkedin": "linkedin.com/in/username",
    "github": "github.com/username",
    "website": "https://example.com",
    "other_links": []
  },
  "summary": "Professional summary paragraph",
  "skills": {
    "programming_languages": ["Python", "SQL"],
    "frameworks_and_tools": ["Django", "Docker"],
    "cloud_and_data": ["AWS", "Databricks"],
    "soft_skills": ["Leadership", "Communication"],
    "other": []
  },
  "experience": [
    {
      "title": "Senior Data Engineer",
      "company": "Acme Corp",
      "location": "Berlin, Germany",
      "start_date": "Jul 2022",
      "end_date": "Present",
      "is_current": true,
      "responsibilities": [
        "Built ETL pipelines reducing processing time by 40%"
      ],
      "technologies_used": ["Python", "Spark", "Azure"]
    }
  ],
  "education": [
    {
      "degree": "B.Tech",
      "field": "Computer Science",
      "institution": "IIT Bombay",
      "location": "Mumbai, India",
      "start_date": "2018",
      "end_date": "2022",
      "gpa": "8.5/10",
      "achievements": []
    }
  ],
  "certifications": [
    {
      "name": "AWS Solutions Architect",
      "issuer": "Amazon",
      "date": "2023",
      "credential_id": ""
    }
  ],
  "projects": [
    {
      "name": "Project Name",
      "description": "What it does",
      "technologies": ["Python", "React"],
      "url": ""
    }
  ],
  "languages": [
    {"language": "English", "proficiency": "Native"},
    {"language": "German", "proficiency": "B2"}
  ],
  "awards_and_achievements": [],
  "publications": [],
  "volunteer_work": [],
  "inferred_insights": {
    "years_of_experience": 3,
    "seniority_level": "Mid-level",
    "primary_domain": "Data Engineering",
    "key_strengths": ["ETL pipelines", "Cloud platforms", "BI dashboards"],
    "open_to_relocation": true,
    "visa_status_mentioned": false,
    "current_location": "Koblenz, Germany"
  }
}

Rules:
- Use null for missing fields, never omit keys
- Infer skills mentioned in job bullet points (e.g. "built pipeline using Kafka" → add Kafka to skills)
- Infer is_current=true if end_date is "Present" or "Current"
- years_of_experience: calculate from earliest job start to today
- seniority_level: Junior / Mid-level / Senior / Lead / Principal / Executive
- If multiple phone numbers, put the most prominent in "phone" and others in "other_links"
"""


def parse_with_claude(text: str, model: str = "claude-opus-4-5") -> dict:
    """Send resume text to Claude and get structured JSON back."""
    api_key = "D:\Practice\AutomaticJobApplicationProject\API_Keys.txt"
    with open(api_key, 'r') as file:
        data = json.load(file)
    claude_api_key = data["Claude"]["api_key"]
    client = anthropic.Anthropic(api_key=claude_api_key)

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Parse this resume and return JSON:\n\n{text}"
            }
        ]
    )

    raw_response = message.content[0].text.strip()

    # Strip any accidental markdown fences
    raw_response = re.sub(r"^```(?:json)?\s*", "", raw_response)
    raw_response = re.sub(r"\s*```$", "", raw_response)

    return json.loads(raw_response)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  OPTIONAL: ENHANCE WITH REGEX FALLBACKS
# ══════════════════════════════════════════════════════════════════════════════
# These only run if Claude returns null for a field — extra safety net

EMAIL_RE    = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PHONE_RE    = re.compile(r"(?:\+?\d{1,3}[\s\-.]?)?(?:\(?\d{2,4}\)?[\s\-.]?)\d{3,4}[\s\-.]?\d{3,5}")
LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
GITHUB_RE   = re.compile(r"github\.com/[\w\-]+", re.IGNORECASE)


def apply_regex_fallbacks(parsed: dict, raw_text: str) -> dict:
    """Fill null contact fields using regex if Claude missed them."""
    c = parsed.get("contact", {})

    if not c.get("email"):
        m = EMAIL_RE.search(raw_text)
        c["email"] = m.group() if m else None

    if not c.get("phone"):
        matches = PHONE_RE.findall(raw_text)
        filtered = [p.strip() for p in matches if len(re.sub(r"\D", "", p)) >= 7]
        c["phone"] = filtered[0] if filtered else None

    if not c.get("linkedin"):
        m = LINKEDIN_RE.search(raw_text)
        c["linkedin"] = m.group() if m else None

    if not c.get("github"):
        m = GITHUB_RE.search(raw_text)
        c["github"] = m.group() if m else None

    parsed["contact"] = c
    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def parse_resume(pdf_path: str, verbose: bool = False) -> dict:
    """
    Full pipeline:
      1. Extract text with PyMuPDF
      2. Send to Claude for intelligent parsing
      3. Apply regex fallbacks for safety
      4. Add metadata
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    if verbose:
        print(f"[1/3] Extracting text from {path.name}...")

    pdf_data = extract_text_with_metadata(pdf_path)
    raw_text = pdf_data["text"]

    if verbose:
        print(f"      Extracted {len(raw_text)} characters from {pdf_data['page_count']} page(s)")
        print("[2/3] Sending to Claude API for parsing...")

    parsed = parse_with_claude(raw_text)

    if verbose:
        print("[3/3] Applying regex fallbacks for null fields...")

    parsed = apply_regex_fallbacks(parsed, raw_text)

    # Attach metadata
    parsed["_meta"] = {
        "source_file": path.name,
        "page_count": pdf_data["page_count"],
        "parsed_at": datetime.now().isoformat(timespec="seconds"),
        "parser_version": "2.0",
        "pdf_embedded_author": pdf_data.get("pdf_author") or None,
    }

    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PRETTY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def pretty_print(data: dict) -> None:
    c = data.get("contact", {})
    ins = data.get("inferred_insights", {})
    skills = data.get("skills", {})

    w = 65
    print("\n" + "═" * w)
    print(f"  {c.get('name', 'Unknown')}".center(w))
    print("═" * w)
    print(f"  Email    : {c.get('email') or '—'}")
    print(f"  Phone    : {c.get('phone') or '—'}")
    print(f"  Location : {c.get('location') or '—'}")
    print(f"  LinkedIn : {c.get('linkedin') or '—'}")
    print(f"  GitHub   : {c.get('github') or '—'}")

    print("\n── Inferred insights ────────────────────────────────────")
    print(f"  Domain      : {ins.get('primary_domain', '—')}")
    print(f"  Seniority   : {ins.get('seniority_level', '—')}")
    print(f"  Experience  : {ins.get('years_of_experience', '?')} year(s)")
    strengths = ins.get("key_strengths", [])
    if strengths:
        print(f"  Strengths   : {', '.join(strengths[:4])}")

    print("\n── Skills ───────────────────────────────────────────────")
    for category, skill_list in skills.items():
        if skill_list:
            label = category.replace("_", " ").title()
            joined = ", ".join(skill_list[:8])
            suffix = f" (+{len(skill_list)-8} more)" if len(skill_list) > 8 else ""
            print(f"  {label:<22}: {joined}{suffix}")

    print("\n── Experience ───────────────────────────────────────────")
    for exp in data.get("experience", []):
        badge = " ◉" if exp.get("is_current") else ""
        print(f"  {exp.get('title','')}  @  {exp.get('company','')}{badge}")
        print(f"    {exp.get('start_date','')} – {exp.get('end_date','')}"
              f"  |  {exp.get('location','')}")
        for r in exp.get("responsibilities", [])[:2]:
            print(f"    • {r[:85]}")

    print("\n── Education ────────────────────────────────────────────")
    for edu in data.get("education", []):
        gpa = f"  GPA: {edu['gpa']}" if edu.get("gpa") else ""
        print(f"  {edu.get('degree','')} in {edu.get('field','')}")
        print(f"    {edu.get('institution','')}  [{edu.get('start_date','')}–{edu.get('end_date','')}]{gpa}")

    certifications = data.get("certifications", [])
    if certifications:
        print("\n── Certifications ───────────────────────────────────────")
        for cert in certifications:
            print(f"  {cert.get('name','')}  —  {cert.get('issuer','')}  ({cert.get('date','')})")

    print("\n" + "═" * w + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Parse a resume PDF using PyMuPDF + Claude API"
    )
    parser.add_argument("pdf", help="Path to the resume PDF")
    parser.add_argument("--output", "-o", help="Output JSON file path (optional)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Skip pretty print")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show progress")
    args = parser.parse_args()

    """if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)"""

    result = parse_resume(args.pdf, verbose=args.verbose)

    output_path = args.output or (Path(args.pdf).stem + "_parsed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if not args.quiet:
        pretty_print(result)

    print(f"JSON saved → {output_path}")


if __name__ == "__main__":
    main()
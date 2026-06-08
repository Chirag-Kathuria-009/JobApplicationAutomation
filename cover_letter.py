"""
job_pipeline/cover_letter.py
==============================
Generates tailored cover letters using Claude API.
Each letter is specific to the company, role, and job description.
Output is saved as a Markdown file ready to copy-paste or convert to PDF.
"""

import os
import anthropic
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

COVER_LETTER_SYSTEM_PROMPT = """You are an expert career coach writing cover letters for a Data Engineer / Data Analyst / Data Scientist applying to German companies.

Rules:
1. Write in English unless the job posting is in German.
2. Keep it to 3 paragraphs + opening/closing. Max 350 words total.
3. Be specific — reference actual projects, technologies, and metrics from the candidate's profile.
4. Match the tone to the company: startup = enthusiastic and direct, bank/consulting = professional and formal, tech = technical and concise.
5. Never use generic phrases like "I am a hard worker" or "I am passionate about data".
6. Always reference the DORA/BaFin compliance project if applying to a financial services company.
7. Always mention Germany student visa + no sponsorship required to reassure German employers.
8. End with a specific call to action.
9. Output plain text only — no markdown headers, no bullet points, just prose paragraphs."""


def detect_company_tone(company_name: str, category: str) -> str:
    """Determine appropriate tone based on company type."""
    formal_categories = ["Banking & Finance", "Consulting", "Automotive"]
    startup_companies = ["N26", "Trade Republic", "Klarna", "SumUp", "Solarisbank", "Raisin",
                         "Celonis", "About You", "Trivago", "Flix", "Miro", "Adjust",
                         "Contentful", "Ada Health", "BioNTech", "Scout24"]

    if company_name in startup_companies:
        return "enthusiastic and direct — this is a growth-stage tech company that values energy and ownership"
    elif category in formal_categories:
        return "professional and formal — this is an established institution that values precision and reliability"
    else:
        return "professional but approachable — balance technical depth with clear communication"


def generate_cover_letter(profile: str, company_name: str, job_title: str,
                           job_description: str, category: str = "") -> str:
    """
    Generate a tailored cover letter for a specific role.

    Returns the cover letter as a string (Markdown format with metadata header).
    """
    tone = detect_company_tone(company_name, category)
    today = datetime.now().strftime("%B %d, %Y")
    is_finance = any(term in company_name.lower() or company_name in [
        "Deutsche Bank", "Commerzbank", "DWS", "Allianz", "N26",
        "Trade Republic", "Solarisbank", "Raisin", "SumUp", "Klarna", "DWS"
    ] for term in ["bank", "finance", "capital", "insurance"])

    user_message = f"""Write a cover letter for this application.

Company: {company_name}
Role: {job_title}
Company type: {category}
Tone: {tone}
Financial services company: {is_finance}
Today's date: {today}

Candidate profile (use specific details from this):
{profile[:3500]}

Job description:
{job_description[:2500]}

Important:
- If this is a financial services company, specifically mention the DORA ICT Incident Intelligence Pipeline project and BaFin compliance experience.
- Always include a line about German student visa (§16b AufenthG) — no sponsorship required.
- Reference specific technologies from the job description that the candidate actually has.
- Use specific numbers/metrics from the profile (67% faster, 93% ROC-AUC, etc.)

Write the letter now:"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=COVER_LETTER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    letter_body = response.content[0].text.strip()

    # Wrap with metadata header for easy reference
    output = f"""---
Company: {company_name}
Role: {job_title}
Generated: {today}
---

{letter_body}

---
Chirag Kathuria
chiragkathuria24de@gmail.com
+49 155 106 33285
Frankfurt, Germany
linkedin.com/in/chirag-kathuria
"""

    return output


def generate_cold_email(profile: str, company_name: str, role_title: str,
                         hiring_manager_name: str = None) -> str:
    """
    Generate a short cold email to a hiring manager.
    Use when you have the hiring manager's email.
    """
    system = """Write a cold email from a job candidate to a hiring manager.
Rules:
- Max 150 words total.
- Subject line on first line prefixed with 'Subject: '
- 3 short paragraphs: hook, why me (1-2 specific facts), call to action.
- Professional but not stiff.
- No attachments mentioned — just ask for a brief call."""

    name_line = f"Hiring Manager's name: {hiring_manager_name}" if hiring_manager_name else "Hiring Manager's name: unknown — use 'Dear Hiring Manager'"

    user_message = f"""Write a cold email.

Company: {company_name}
Role I'm interested in: {role_title}
{name_line}

My profile (use 1-2 specific facts):
{profile[:2000]}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=system,
        messages=[{"role": "user", "content": user_message}]
    )

    return response.content[0].text.strip()

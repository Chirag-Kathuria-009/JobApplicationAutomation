"""
Resume tailoring rules: prompt adherence checks, skills pruning, keyword hints,
section order, job-text cleaning.  No model or network needed.
Run:  python tests/test_resume.py        (needs BASE_RESUME/base.json)
"""
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import resume  # noqa: E402
from jd_cleaner import clean_job_description  # noqa: E402

BASE = resume.load_base()


def good() -> dict:
    return {
        "summary": "Data Engineer with 2+ years building ETL/ELT pipelines at Bajaj Finserv using SQL, Python and Azure Data Factory.",
        "skills": [{"label": f"G{i}", "items": ["Python", "SQL", "PostgreSQL"]} for i in range(4)],
        "experience": {
            "exp_werkstudent": ["Manual reporting was slow; built Power BI dashboards, cutting reporting time by ~70%."],
            "exp_bajaj": ["Issue detection across 5+ sources took 30 min; built ETL/ELT pipelines in Python and SQL, cutting it to 10 min (67% faster)."],
            "exp_intern": ["Built CI/CD pipelines with Jenkins and Azure DevOps, cutting deployment failures by 50%."],
        },
        "projects": [
            {"id": "proj_fraud", "bullets": ["Fraud was 0.3% of 6M+ transactions; trained LightGBM to 93% ROC-AUC with SHAP."]},
            {"id": "proj_dora", "bullets": ["Built a Kafka and PySpark pipeline handling 100+ events/min with 100% unit test coverage."]},
            {"id": "proj_nlp", "bullets": ["Built BERT NLI on 185K+ FEVER claims, reaching 87% accuracy."]},
        ],
    }


def errors(mutate) -> list[str]:
    d = copy.deepcopy(good())
    mutate(d)
    return validate(d)


def validate(d):
    return resume.validate_tailored(BASE, d)[0]


def test_valid_output_passes():
    assert validate(good()) == []


def test_invented_need_opener_is_rejected():
    e = errors(lambda d: d["experience"].__setitem__("exp_bajaj", ["Needed reliable data for analytics; built ETL pipelines, cutting it by 67%."]))
    assert any("generic need" in x for x in e)
    e = errors(lambda d: d["projects"][0].__setitem__("bullets", ["The business needed visibility; built a model reaching 93% ROC-AUC."]))
    assert any("generic need" in x for x in e)
    # a problem taken from the original wording is fine
    assert errors(lambda d: d["experience"].__setitem__("exp_bajaj", ["Legacy pipelines needed modernising; moved them to Databricks, gaining 35% compute efficiency."])) == []


def test_invented_number_and_foreign_tool_are_rejected():
    assert any("numbers" in x for x in errors(lambda d: d["experience"].__setitem__("exp_bajaj", ["Cut load time by 82% with Python."])))
    assert any("Tableau" in x for x in errors(lambda d: d["summary"].__class__ and d.__setitem__("summary", "Data Engineer skilled in Tableau and SQL.")))
    assert any("Talend" in x for x in errors(lambda d: d["experience"].__setitem__("exp_bajaj", ["Built Talend jobs integrating 5+ sources in SQL."])))


def test_summary_is_capped_at_about_four_lines():
    long = " ".join(["Python"] * (resume.MAX_SUMMARY_WORDS + 5))
    assert any("words" in x for x in errors(lambda d: d.__setitem__("summary", long)))


def test_skills_are_pruned_to_six_groups_and_eight_items():
    def many(d):
        d["skills"] = [{"label": f"Group{i}", "items": ["Python", "SQL", "PostgreSQL", "Docker", "Git", "Airflow",
                                                         "Apache Kafka", "Databricks", "Great Expectations", "dbt"]} for i in range(8)]
    d = copy.deepcopy(good()); many(d)
    errs, cleaned, _removed = resume.validate_tailored(BASE, d)
    assert errs == []
    assert len(cleaned["skills"]) == resume.MAX_SKILL_GROUPS
    assert all(len(g["items"]) <= resume.MAX_SKILL_ITEMS for g in cleaned["skills"])
    assert cleaned["skills_trimmed"], "what was cut is recorded, not silent"


def test_skills_not_backed_by_base_resume_are_removed():
    d = copy.deepcopy(good())
    d["skills"][0]["items"] = ["Python", "Talend", "Dagster", "SQL"]
    errs, cleaned, removed = resume.validate_tailored(BASE, d)
    assert errs == [] and set(removed) == {"Talend", "Dagster"}


def test_job_hints_split_owned_keywords_from_gaps():
    h = resume.jd_hints(BASE, "We use Python, SQL, PowerBI, Kafka, Tableau, Terraform and Talend.")
    assert {"Python", "SQL", "Power BI", "Kafka"} <= set(h["owned"])
    assert {"Tableau", "Terraform", "Talend"} <= set(h["gaps"]) and "Python" not in h["gaps"]


def test_pdf_sections_follow_the_prompt_order_and_nothing_is_dropped_silently():
    from resume_render import render_resume_pdf
    from pypdf import PdfReader
    r = resume.assemble_resume(BASE, resume.validate_tailored(BASE, good())[1])
    out = Path(tempfile.mkdtemp()) / "t.pdf"
    report: dict = {}
    render_resume_pdf(r, out, report)
    text = "".join(p.extract_text() for p in PdfReader(str(out)).pages)
    order = [text.index(h) for h in ("PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "EXPERIENCE", "PROJECTS", "EDUCATION")]
    assert order == sorted(order), "Summary, Skills, Experience, Projects, Education"
    assert len(PdfReader(str(out)).pages) == 1 and report["dropped_bullets"] == [] and report["font_pt"] >= 9


def test_job_text_keeps_role_and_requirements_ahead_of_company_pitch():
    text = ("2027 BI Internship\nWe are on the lookout for the curious. At Amazon you will grow.\n"
            "Key job responsibilities\n- Develop analytical solutions\n- Write code to retrieve data\n"
            "A day in the life\nWe build pipelines and dashboards.\n"
            "Basic Qualifications\n- Enrolled in a Bachelor's degree\nPreferred Qualifications\n- Tableau, Excel or PowerBI\n")
    out = clean_job_description(text, 4000)
    order = [out.index(h) for h in ("RESPONSIBILITIES:", "REQUIREMENTS:", "ABOUT THE COMPANY:")]
    assert order == sorted(order)
    assert "Tableau" in out and "Develop analytical solutions" in out


def test_benefits_and_apply_sections_are_dropped():
    out = clean_job_description("Your Role:\nBuild pipelines.\nBenefits:\n30 days holiday\nHow to apply:\nSend CV", 4000)
    assert "holiday" not in out and "Send CV" not in out and "Build pipelines" in out


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:           # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)

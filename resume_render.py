"""
job_pipeline/resume_render.py
=============================
Renders a flat resume dict (see resume.assemble_resume) to a one-page,
ATS-friendly A4 PDF: HTML -> Chromium (Playwright) -> PDF. The layout is fixed,
so every application looks identical; only the content changes.

Fitting strategy: try font sizes from large to small. If it still doesn't fit
at the smallest allowed size, drop the lowest-ranked bullet (the model returns
bullets most-relevant-first, so the last one is the least relevant) and retry
from the largest size. The font never goes below FONT_SIZES_PT[-1].
"""

import copy
import html
import re
from pathlib import Path

from pypdf import PdfReader

FONT_SIZES_PT = [10, 9.5, 9]   # tried in order until it fits on one page
MAX_TRIMS = 6                         # bullets that may be dropped to fit

CSS = """
@page { size: A4; margin: 9mm 12mm; }
body { font-family: Arial, Helvetica, sans-serif; font-size: @FS@pt; line-height: 1.2; color: #111; margin: 0; }
h1 { font-size: 16pt; margin: 0 0 1px 0; }
.contact { font-size: @FS_SMALL@pt; color: #333; margin-bottom: 2px; }
.auth { font-size: @FS_SMALL@pt; color: #333; margin-bottom: 4px; }
h2 { font-size: @FS_H2@pt; text-transform: uppercase; letter-spacing: .5px; border-bottom: 1px solid #111;
     margin: 5px 0 3px 0; padding-bottom: 1px; }
.row { display: flex; justify-content: space-between; font-weight: bold; }
.sub { display: flex; justify-content: space-between; font-style: italic; }
.tech { font-size: @FS_SMALL@pt; font-style: italic; color: #333; }
ul { margin: 1px 0 3px 15px; padding: 0; }
li { margin: 0 0 1px 0; }
.skills p { margin: 1px 0; }
"""


def _bold(text: str, terms: list[str]) -> str:
    """HTML-escape text and wrap each bold term (whole-word, case-insensitive) in <b>."""
    out = html.escape(text)
    for t in sorted(terms, key=len, reverse=True):
        out = re.sub(rf"(?<![\w>])({re.escape(html.escape(t))})(?![\w<])",
                     r"<b>\1</b>", out, flags=re.I)
    return out


def _html(r: dict, fs: float) -> str:
    terms = r.get("bold_terms", [])
    b = lambda s: _bold(s, terms)
    p = [f"<h1>{html.escape(r['name'])}</h1>",
         f"<div class='contact'>{' | '.join(html.escape(c) for c in r['contact'])}</div>"]
    if r.get("work_auth"):
        p.append(f"<div class='auth'>{html.escape(r['work_auth'])}</div>")
    if r.get("summary"):
        p.append(f"<h2>Professional Summary</h2><div>{b(r['summary'])}</div>")
    p.append("<h2>Technical Skills</h2><div class='skills'>")
    for s in r["skills"]:
        p.append(f"<p><b>{html.escape(s['label'])}:</b> {html.escape(s['items'])}</p>")
    p.append("</div>")
    p.append("<h2>Experience</h2>")
    for e in r["experience"]:
        p.append(f"<div class='row'><span>{html.escape(e['role'])}</span><span>{html.escape(e['dates'])}</span></div>")
        p.append(f"<div class='sub'><span>{html.escape(e['company'])}</span><span>{html.escape(e.get('location', ''))}</span></div>")
        p.append("<ul>" + "".join(f"<li>{b(x)}</li>" for x in e["bullets"]) + "</ul>")
    if r.get("projects"):
        p.append("<h2>Projects</h2>")
        for pr in r["projects"]:
            p.append(f"<div class='row'><span>{html.escape(pr['title'])}</span><span>{html.escape(pr.get('dates', ''))}</span></div>")
            if pr.get("tech"):
                p.append(f"<div class='tech'>{html.escape(pr['tech'])}</div>")
            p.append("<ul>" + "".join(f"<li>{b(x)}</li>" for x in pr["bullets"]) + "</ul>")
    p.append("<h2>Education</h2>")
    for ed in r["education"]:
        p.append(f"<div class='row'><span>{html.escape(ed['degree'])}</span><span>{html.escape(ed['dates'])}</span></div>")
        p.append(f"<div class='sub'><span>{html.escape(ed['school'])}</span><span></span></div>")
    css = (CSS.replace("@FS@", str(fs)).replace("@FS_SMALL@", str(round(fs - 0.5, 1)))
              .replace("@FS_H2@", str(round(fs + 0.5, 1))))
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(r['name'])} — Resume</title>"
            f"<style>{css}</style></head><body>{''.join(p)}</body></html>")


def _trim_one(resume: dict) -> str | None:
    """Drop the lowest-ranked bullet from the section with the most bullets.

    Projects are trimmed first (down to 1 bullet each), then experience (down
    to 2 per role). Returns the dropped bullet text, or None when nothing more
    can be dropped.
    """
    for entries, floor in ((resume.get("projects", []), 1), (resume["experience"], 2)):
        candidates = [e for e in entries if len(e["bullets"]) > floor]
        if candidates:
            return max(candidates, key=lambda e: len(e["bullets"]))["bullets"].pop()
    return None


def render_resume_pdf(resume: dict, out_path: str | Path, report: dict | None = None) -> Path:
    """Render `resume` to a one-page PDF at out_path (trims bullets if needed).

    Raises RuntimeError if it can't fit on one page. The resume dict passed in
    is not modified. If `report` is given it is filled with {"font_pt": size used,
    "dropped_bullets": [text, ...]} so nothing is cut silently.
    """
    from playwright.sync_api import sync_playwright

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = copy.deepcopy(resume)
    dropped: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            for _ in range(MAX_TRIMS + 1):
                for fs in FONT_SIZES_PT:
                    page.set_content(_html(work, fs), wait_until="load")
                    page.pdf(path=str(out_path), format="A4", print_background=True,
                             prefer_css_page_size=True)
                    if len(PdfReader(str(out_path)).pages) == 1:
                        if report is not None:
                            report.update({"font_pt": fs, "dropped_bullets": dropped})
                        return out_path
                cut = _trim_one(work)
                if cut is None:
                    break
                dropped.append(cut)
        finally:
            browser.close()
    if out_path.exists():
        out_path.unlink()   # never leave a multi-page PDF behind
    raise RuntimeError(f"Resume does not fit on one page at {FONT_SIZES_PT[-1]}pt "
                       f"even after trimming bullets.")

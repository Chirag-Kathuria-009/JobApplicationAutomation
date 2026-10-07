"""
job_pipeline/dedupe.py
======================
Decides whether a scraped job is one we already have, so scoring, the cover
letter and the tailored resume are never produced twice for the same application.

A job counts as a duplicate when ANY of these match an earlier tracker entry:
  ats      the same posting id on a known ATS (Greenhouse / Lever / Ashby /
           Personio / SmartRecruiters / Recruitee), whatever the URL decoration
  url      the same URL after dropping tracking parameters, "www.", fragments,
           trailing slashes and "/apply"
  ct       the same company + job title after normalising legal suffixes
           (GmbH, AG, SE, Inc ...) and gender markers ((m/w/d), (f/m/x) ...)
  jd       the same job-description text under the same company (a repost under
           a new URL or a slightly different title)
Location is deliberately NOT part of the key: the same role advertised in
several cities is one application.
"""

import hashlib
import re
from urllib.parse import parse_qsl, urlparse

_TRACKING_PARAMS = {
    "gh_src", "source", "src", "ref", "referrer", "lever-source", "lever-origin", "trk",
    "trackingid", "gclid", "fbclid", "mc_cid", "mc_eid", "ccuid", "sid", "li_fat_id",
    "jobboard", "board", "origin", "campaign", "channel", "medium",
}

_ATS_PATTERNS = [
    (re.compile(r"(?:boards|job-boards)\.greenhouse\.io/[^/]+/jobs/(\d+)", re.I), "gh:{0}"),
    (re.compile(r"[?&]gh_jid=(\d+)", re.I), "gh:{0}"),
    (re.compile(r"jobs\.lever\.co/[^/]+/([0-9a-f-]{36})", re.I), "lever:{0}"),
    (re.compile(r"jobs\.ashbyhq\.com/[^/]+/([0-9a-f-]{36})", re.I), "ashby:{0}"),
    (re.compile(r"//([a-z0-9-]+)\.jobs\.personio\.(?:de|com)/job/(\d+)", re.I), "personio:{0}:{1}"),
    (re.compile(r"smartrecruiters\.com/[^/]+/(\d+)", re.I), "sr:{0}"),
    (re.compile(r"//([a-z0-9-]+)\.recruitee\.com/o/([^/?#]+)", re.I), "recruitee:{0}:{1}"),
]

_COMPANY_STOPWORDS = {
    "gmbh", "mbh", "ag", "se", "kg", "ug", "inc", "ltd", "llc", "plc", "bv", "nv", "sa", "sas",
    "srl", "spa", "oy", "ab", "as", "co", "corp", "corporation", "company", "group", "holding",
    "holdings", "deutschland", "germany", "the", "limited", "und", "and",
}

_GENDER_BRACKETS = re.compile(
    r"[\(\[]\s*(?:all\s+genders?|gn|(?:[mfwdxi*]\s*[/|,]\s*){1,3}[mfwdxi*])\s*[\)\]]", re.I)
_GENDER_BARE = re.compile(r"\b(?:[mfwdx]\s*/\s*){1,3}[mfwdx]\b|\ball\s+genders?\b", re.I)


def normalize_url(url: str) -> str:
    """host/path?kept-params with tracking noise removed ('' for an empty URL)."""
    url = (url or "").strip()
    if not url:
        return ""
    p = urlparse(url if "//" in url else "//" + url)
    host = p.netloc.lower().removeprefix("www.")
    path = re.sub(r"/(apply|application)/?$", "", p.path.rstrip("/"), flags=re.I)
    query = sorted((k.lower(), v) for k, v in parse_qsl(p.query)
                   if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith("utm_"))
    q = "&".join(f"{k}={v}" for k, v in query)
    return f"{host}{path}" + (f"?{q}" if q else "")


def ats_key(url: str) -> str:
    """Stable posting id for known ATS URLs, '' otherwise."""
    for pattern, fmt in _ATS_PATTERNS:
        m = pattern.search(url or "")
        if m:
            return fmt.format(*[g.lower() for g in m.groups()])
    return ""


def normalize_company(name: str) -> str:
    tokens = re.sub(r"[^\w]+", " ", (name or "").lower()).split()
    kept = [t for t in tokens if t not in _COMPANY_STOPWORDS]
    return " ".join(kept or tokens)


def normalize_title(title: str) -> str:
    t = _GENDER_BRACKETS.sub(" ", title or "")
    t = _GENDER_BARE.sub(" ", t.lower())
    t = t.replace("&", " and ")
    t = re.sub(r"\bsr\.?(?=\s)", "senior", t)
    t = re.sub(r"\bjr\.?(?=\s)", "junior", t)
    return " ".join(re.sub(r"[^\w]+", " ", t).split())


def jd_hash(text: str) -> str:
    """Hash of the normalised start of the description; '' when the text is too short to be meaningful
    (e.g. the 'Role at X. Visit URL' placeholders some scrapers produce)."""
    words = re.findall(r"\w+", (text or "").lower())
    body = " ".join(words)[:1500]
    if len(body) < 400:
        return ""
    return hashlib.md5(body.encode("utf-8")).hexdigest()


def fingerprints(company: str, title: str, url: str = "", jd_text: str = "") -> dict:
    comp = normalize_company(company)
    return {
        "ats": ats_key(url),
        "url": normalize_url(url),
        "ct": f"{comp}|{normalize_title(title)}",
        "company": comp,
        "jd": jd_hash(jd_text),
    }


class DedupeIndex:
    """In-memory lookup over tracker entries; O(1) per job instead of scanning the tracker."""

    def __init__(self, applications: list[dict] | None = None):
        self._by: dict[tuple[str, str], dict] = {}
        for entry in applications or []:
            self.add(entry_fingerprints(entry), entry)

    def add(self, fp: dict, entry: dict) -> None:
        for kind in ("ats", "url", "ct"):
            if fp.get(kind):
                self._by.setdefault((kind, fp[kind]), entry)
        if fp.get("jd") and fp.get("company"):
            self._by.setdefault(("jd", f"{fp['company']}|{fp['jd']}"), entry)
        for alt_ats, alt_url in fp.get("alt", []):
            if alt_ats:
                self._by.setdefault(("ats", alt_ats), entry)
            if alt_url:
                self._by.setdefault(("url", alt_url), entry)

    def find(self, fp: dict) -> tuple[dict | None, str]:
        """(existing entry, reason) or (None, '')."""
        for kind in ("ats", "url", "ct"):
            if fp.get(kind) and (kind, fp[kind]) in self._by:
                return self._by[(kind, fp[kind])], kind
        if fp.get("jd") and fp.get("company"):
            hit = self._by.get(("jd", f"{fp['company']}|{fp['jd']}"))
            if hit:
                return hit, "jd"
        return None, ""


def entry_fingerprints(entry: dict) -> dict:
    """Fingerprints stored on a tracker entry, or recomputed for older entries without them."""
    stored = entry.get("fingerprints")
    if stored:
        fp = dict(stored)
    else:
        fp = fingerprints(entry.get("company", ""), entry.get("role_title", ""), entry.get("role_url", ""))
    for alt in entry.get("alt_urls", []):      # other URLs this same job was seen under
        fp.setdefault("alt", []).append((ats_key(alt), normalize_url(alt)))
    return fp

"""
job_pipeline/scraper.py
========================
Scrapes job listings from company career portals.
Handles Greenhouse, Workday, Lever, and Proprietary portals via Playwright.

Returns a list of dicts: [{title, description, url, location, date_posted, ats}]
"""

import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, quote
import httpx
from bs4 import BeautifulSoup

# Data-domain role keywords (English + German variants).
# Used as a coarse title pre-filter; matched as case-insensitive substrings.
ROLE_KEYWORDS = [
    # English data roles
    "data engineer", "data analyst", "data scientist", "data science",
    "data analytics", "data platform", "data warehouse", "data quality",
    "data governance", "data management", "database",
    "analytics engineer", "analytics", "business intelligence",
    "bi engineer", "bi developer", "bi analyst",
    "ml engineer", "machine learning", "ai engineer", "mlops",
    "big data", "etl", "data pipeline", "data architect",
    "reporting analyst", "quantitative analyst",
    # German data roles
    "datenanalyst", "datenanalyse", "datenwissenschaft", "data scientist",
    "dateningenieur", "datenmanagement", "datenbank",
    # Student/junior data roles (kept for backwards-compatibility)
    "werkstudent data", "working student data",
]

# Internship / working-student keywords (English + German).
# Short tokens like "intern" use word boundaries so "international" does NOT match.
INTERNSHIP_KEYWORDS_PATTERN = re.compile(
    r"\b("
    r"intern|interns|internship|internships"          # English
    r"|praktikum|praktikant|praktikantin|praktika"     # German (Praktikum/Praktikant)
    r"|werkstudent|werkstudentin"                       # German (working student)
    r"|working student"                                 # English
    r"|trainee|graduate program|graduate programme"     # entry-level programs
    r")\b",
    re.IGNORECASE,
)

EXCLUDE_KEYWORDS = [
    "senior director", "director of", "vp of", "vice president", "head of",
    "principal architect", "staff engineer", "engineering manager", "chief"
]

# Germany + Luxembourg + nearby European countries that share a border with
# Germany (and their major cities). Matched as case-sensitive substrings in
# _in_target_location(); remote/unknown locations are allowed through there.
TARGET_LOCATIONS = [
    # Remote / flexible
    "Remote", "Anywhere", "Hybrid",
    # Germany (country + major cities) and DACH
    "Germany", "Deutschland", "DACH",
    "Berlin", "Munich", "München", "Frankfurt", "Hamburg", "Cologne",
    "Köln", "Stuttgart", "Düsseldorf", "Dusseldorf", "Leipzig", "Dortmund",
    "Essen", "Bremen", "Dresden", "Hanover", "Hannover", "Nuremberg",
    "Nürnberg", "Duisburg", "Bonn", "Mannheim", "Karlsruhe", "Wiesbaden",
    "Münster", "Aachen", "Freiburg", "Mainz", "Kiel", "Heidelberg",
    # Luxembourg
    "Luxembourg",
    # Bordering countries + key cities
    "Netherlands", "Nederland", "Amsterdam", "Rotterdam", "Eindhoven", "Utrecht",
    "The Hague", "Amstelveen",
    "Belgium", "Brussels", "Antwerp", "Ghent",
    "France", "Paris", "Strasbourg", "Lille", "Metz", "Nancy",
    "Switzerland", "Schweiz", "Zurich", "Zürich", "Basel", "Geneva", "Bern",
    "Austria", "Österreich", "Vienna", "Wien", "Salzburg", "Innsbruck", "Linz", "Graz",
    "Czech", "Czechia", "Prague", "Brno", "Plzen",
    "Poland", "Polska", "Warsaw", "Kraków", "Krakow", "Wrocław", "Wroclaw", "Poznań",
    "Denmark", "Copenhagen", "Aarhus", "Odense",
    # ISO country codes, comma-delimited as they appear in some ATS location
    # strings (e.g. "München, BY, DE, 80809"). Germany + bordering countries.
    ", DE", ", LU", ", AT", ", CH", ", FR", ", NL", ", BE", ", CZ", ", PL", ", DK",
]

# The word "English" in several languages, so we detect an English-language
# requirement even when the job description itself is written in another language.
ENGLISH_LANGUAGE_TOKENS = [
    "english",            # English
    "englisch",           # German (incl. "Englischkenntnisse")
    "anglais",            # French
    "inglés", "ingles",   # Spanish
    "inglese",            # Italian
    "engels",             # Dutch
    "inglês",             # Portuguese
]

# Search terms used to query keyword-search portals (Workday, Radancy/TalentBrew).
# Covers the profile's target roles across all job types — Werkstudent /
# part-time / full-time / internship — in both English and German.
DATA_SEARCH_TERMS = [
    # Core data roles
    "data engineer", "data analyst", "data scientist", "data science",
    "analytics engineer", "data platform", "business intelligence",
    "bi engineer", "bi developer", "ml engineer", "machine learning",
    "data analytics", "analytics",
    # German variants
    "datenanalyse", "datenanalyst", "data science",
    # Werkstudent / part-time / internship (English + German)
    "werkstudent data", "working student data",
    "data internship", "data intern",
    "praktikum data", "praktikum business intelligence", "praktikant data",
]

MAX_AGE_DAYS = 7

# Used to pull a contact (SPOC) email out of a job description when present.
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Addresses that are never a real recruiter contact (assets, tracking, examples).
_EMAIL_NOISE = ("example.com", "sentry.io", "@2x", ".png", ".jpg", ".gif",
                "wixpress", "no-reply", "noreply", "donotreply")


def extract_spoc_email(*texts: str) -> str:
    """Best-effort: return the first contact (SPOC) email found in the text(s).

    Greenhouse/Lever rarely expose a recruiter email via their API, but some
    descriptions include a contact address (e.g. "questions? jobs@company.com").
    Returns "" when none is found.
    """
    for text in texts:
        if not text:
            continue
        for match in EMAIL_PATTERN.findall(text):
            if not any(bad in match.lower() for bad in _EMAIL_NOISE):
                return match
    return ""


def _greenhouse_first_published(handle: str, job_id) -> str:
    """Fetch one Greenhouse job's `first_published` date (the original post date).

    Used when the bulk listing endpoint doesn't include it. Returns "" on failure.
    """
    if not job_id:
        return ""
    try:
        url = f"https://boards-api.greenhouse.io/v1/boards/{handle}/jobs/{job_id}"
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        if resp.status_code == 200:
            return resp.json().get("first_published", "")
    except Exception:
        pass
    return ""


def requires_english(description: str) -> bool:
    """Return True if the posting references English as a language.

    Matches the word "English" across several languages so that a posting
    written in (e.g.) German that asks for "gute Englischkenntnisse" is still
    recognised as requiring English.
    """
    desc_lower = description.lower()
    return any(token in desc_lower for token in ENGLISH_LANGUAGE_TOKENS)


def is_posted_within_days(date_str: str, days: int = MAX_AGE_DAYS) -> bool:
    """Return True if job was posted within `days` days. Passes through if date is unknown."""
    if not date_str:
        return True
    try:
        posted = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if posted.tzinfo is None:
            posted = posted.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - posted <= timedelta(days=days)
    except (ValueError, TypeError):
        return True


def is_relevant_role(title: str) -> bool:
    """Check if a job title is relevant, excluding too-senior positions.

    A title is relevant if it matches a data-domain role keyword OR an
    internship / working-student keyword (English or German).
    """
    title_lower = title.lower()
    if any(excl in title_lower for excl in EXCLUDE_KEYWORDS):
        return False
    if any(kw in title_lower for kw in ROLE_KEYWORDS):
        return True
    return bool(INTERNSHIP_KEYWORDS_PATTERN.search(title_lower))


def _in_target_location(location: str) -> bool:
    """Return True if location is Germany, Luxembourg, remote, or unknown."""
    if not location:
        return True  # unknown location — allow through
    return any(loc in location for loc in TARGET_LOCATIONS)


def scrape_greenhouse(company: dict) -> list[dict]:
    """
    Greenhouse JSON API: boards-api.greenhouse.io/v1/boards/{handle}/jobs
    Very reliable — structured JSON response.
    """
    url = company["careers_url"]

    if company.get("ats_handle"):
        handle = company["ats_handle"]
    elif "greenhouse.io" in url:
        handle = url.rstrip("/").split("/")[-1]
    else:
        # Custom domain — derive handle from company name
        handle = company["name"].lower().replace(" ", "").replace("-", "").replace("_", "")

    api_url = f"https://boards-api.greenhouse.io/v1/boards/{handle}/jobs?content=true"
    jobs = []

    try:
        resp = httpx.get(api_url, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            print(f"  [Greenhouse] {company['name']}: HTTP {resp.status_code}")
            return []

        data = resp.json()
        for job in data.get("jobs", []):
            title = job.get("title", "")
            if not is_relevant_role(title):
                continue

            location = job.get("location", {}).get("name", "")
            if not _in_target_location(location):
                continue

            # Use first_published (original post date), NOT updated_at — we only
            # want roles first posted within the last week. Fall back to a
            # per-job lookup if the bulk listing omits the field.
            date_posted = job.get("first_published") or _greenhouse_first_published(
                handle, job.get("id")
            )
            if not is_posted_within_days(date_posted):
                continue

            description = BeautifulSoup(
                job.get("content", ""), "html.parser"
            ).get_text(separator=" ")

            if not requires_english(description):
                continue

            jobs.append({
                "title": title,
                "description": description[:3000],
                "url": job.get("absolute_url", ""),
                "location": location,
                "date_posted": date_posted,
                "spoc_email": extract_spoc_email(description),
                "ats": "Greenhouse"
            })

    except Exception as e:
        print(f"  [Greenhouse] Error scraping {company['name']}: {e}")

    return jobs


def scrape_lever(company: dict) -> list[dict]:
    """
    Lever JSON API: api.lever.co/v0/postings/{handle}
    """
    url = company["careers_url"]

    if company.get("ats_handle"):
        handle = company["ats_handle"]
    elif "lever.co" in url:
        handle = url.rstrip("/").split("/")[-1]
    else:
        handle = company["name"].lower().replace(" ", "").replace("-", "").replace("_", "")

    api_url = f"https://api.lever.co/v0/postings/{handle}?mode=json"
    jobs = []

    try:
        resp = httpx.get(api_url, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            print(f"  [Lever] {company['name']}: HTTP {resp.status_code}")
            return []

        data = resp.json()
        for job in data:
            title = job.get("text", "")
            if not is_relevant_role(title):
                continue

            location = job.get("categories", {}).get("location", "")
            if not _in_target_location(location):
                continue

            # createdAt is a Unix timestamp in milliseconds
            created_ms = job.get("createdAt")
            if created_ms:
                date_posted = datetime.fromtimestamp(
                    created_ms / 1000, tz=timezone.utc
                ).isoformat()
            else:
                date_posted = ""

            if not is_posted_within_days(date_posted):
                continue

            description = " ".join(
                lst.get("content", "") for lst in job.get("lists", [])
            )
            description = (description + " " + job.get("descriptionPlain", "")).strip()

            if not requires_english(description):
                continue

            jobs.append({
                "title": title,
                "description": description[:3000],
                "url": job.get("hostedUrl", ""),
                "location": location,
                "date_posted": date_posted,
                "spoc_email": extract_spoc_email(description),
                "ats": "Lever"
            })

    except Exception as e:
        print(f"  [Lever] Error scraping {company['name']}: {e}")

    return jobs


def scrape_workday_playwright(company: dict) -> list[dict]:
    """
    Legacy Workday scraper via Playwright — used only as a fallback when the
    company's Workday tenant URL (myworkdayjobs.com) isn't known, so the CXS
    JSON API can't be used. Descriptions are placeholders.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(f"  [Workday] Playwright not installed — skipping {company['name']}")
        return []

    jobs = []
    base_url = company["careers_url"]
    seen_titles: set[str] = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            # domcontentloaded is faster and avoids hanging on pages with continuous polling
            page.goto(base_url, timeout=30000, wait_until="domcontentloaded")

            for term in DATA_SEARCH_TERMS:
                search_selectors = [
                    'input[data-automation-id="searchBox"]',
                    'input[placeholder*="Search"]',
                    'input[type="search"]',
                    '#searchBox'
                ]
                filled = False
                for sel in search_selectors:
                    try:
                        page.fill(sel, term, timeout=3000)
                        page.keyboard.press("Enter")
                        page.wait_for_timeout(2000)
                        filled = True
                        break
                    except Exception:
                        continue

                if not filled:
                    continue

                job_elements = page.query_selector_all(
                    '[data-automation-id="jobTitle"], .css-19uc56f, h3 a'
                )
                for el in job_elements[:10]:
                    title = el.text_content().strip()
                    if not title or title in seen_titles:
                        continue
                    if not is_relevant_role(title):
                        continue

                    href = el.get_attribute("href") or ""
                    if href and not href.startswith("http"):
                        href = urljoin(base_url, href)

                    seen_titles.add(title)
                    jobs.append({
                        "title": title,
                        "description": f"Role at {company['name']}. Visit {href or base_url} for full description.",
                        "url": href or base_url,
                        "location": "",
                        "date_posted": "",
                        "spoc_email": "",
                        "ats": "Workday"
                    })

        except Exception as e:
            print(f"  [Workday] Error scraping {company['name']}: {e}")
        finally:
            browser.close()

    return jobs


def scrape_proprietary(company: dict) -> list[dict]:
    """
    Fallback for proprietary portals — HTML scraping via BeautifulSoup.
    """
    base_url = company["careers_url"]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    keyword_pattern = re.compile(
        r"(data engineer|data analyst|data scientist|data science"
        r"|analytics engineer|data platform|business intelligence"
        r"|ml engineer|machine learning|bi engineer|bi developer|analytics"
        # German data roles + internship / working-student terms (EN + DE)
        r"|datenanalyst|datenanalyse|dateningenieur"
        r"|werkstudent|praktikum|praktikant|internship|\bintern\b)",
        re.IGNORECASE
    )

    def extract_jobs_from_soup(soup: BeautifulSoup, page_url: str) -> list[dict]:
        """Extract job links from parsed HTML — handles nested title elements."""
        found = []
        for link in soup.find_all("a"):
            # get_text() captures text from all descendant elements
            text = link.get_text(strip=True)
            if not keyword_pattern.search(text):
                continue
            if not is_relevant_role(text):
                continue
            href = link.get("href", "")
            if href and not href.startswith("http"):
                href = urljoin(page_url, href)
            found.append({
                "title": text,
                "description": f"Role at {company['name']}. Visit {href or page_url} for full description.",
                "url": href or page_url,
                "location": "",
                "date_posted": "",
                "spoc_email": "",
                "ats": "Proprietary"
            })
        return found

    jobs = []
    try:
        resp = httpx.get(base_url, timeout=20, follow_redirects=True, headers=headers)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            jobs = extract_jobs_from_soup(soup, base_url)

        if not jobs:
            for search_url in [
                f"{base_url}?q=data+engineer",
                f"{base_url}/search?term=data",
                f"{base_url}?keyword=data+analyst",
            ]:
                try:
                    sresp = httpx.get(search_url, timeout=10, follow_redirects=True, headers=headers)
                    if sresp.status_code == 200:
                        ssoup = BeautifulSoup(sresp.text, "html.parser")
                        jobs = extract_jobs_from_soup(ssoup, search_url)
                        if jobs:
                            break
                except Exception:
                    continue

    except Exception as e:
        print(f"  [Proprietary] Error scraping {company['name']}: {e}")

    return jobs


def _parse_talentbrew_date(date_str: str) -> str:
    """Convert a 'Jun 4, 2026' style date to ISO; '' if unparseable.

    Tolerant of collapsed/extra whitespace (e.g. Amazon's 'May  6, 2026').
    """
    date_str = " ".join((date_str or "").split())
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d.%m.%Y"):
        try:
            dt = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue
    return ""


def scrape_radancy(company: dict) -> list[dict]:
    """
    Radancy / TalentBrew career sites (e.g. jobs.bmwgroup.com).

    Results are rendered server-side as HTML at /search-jobs/{keyword}, so a
    plain HTTP GET + BeautifulSoup is enough — no browser required. Each result
    row carries title, location and posting date; the detail page is fetched
    best-effort for the full description (used for scoring + SPOC email).
    """
    base_url = company["careers_url"].rstrip("/")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    jobs: list[dict] = []
    seen_ids: set[str] = set()

    def parse_results(html: str) -> None:
        soup = BeautifulSoup(html, "html.parser")
        container = soup.find(id="search-results-list") or soup
        for link in container.find_all("a", href=True):
            href = link["href"]
            if "/job/" not in href:
                continue

            job_id = href.rstrip("/").split("/")[-1]
            if job_id in seen_ids:
                continue

            title_el = link.find(["h2", "h3", "h4"])
            title = (title_el.get_text(strip=True) if title_el
                     else link.get_text(" ", strip=True))
            if not title or not is_relevant_role(title):
                continue

            loc_el = link.find(class_=re.compile(r"location", re.I))
            location = loc_el.get_text(" ", strip=True) if loc_el else ""
            if not _in_target_location(location):
                continue

            date_el = link.find(class_=re.compile(r"date", re.I))
            date_posted = _parse_talentbrew_date(date_el.get_text() if date_el else "")
            if not is_posted_within_days(date_posted):
                continue

            seen_ids.add(job_id)
            full_url = urljoin(base_url, href)

            # Best-effort: fetch the detail page for a real description + email.
            description = f"Role at {company['name']}. Visit {full_url} for full description."
            spoc_email = ""
            try:
                dresp = httpx.get(full_url, timeout=15, follow_redirects=True, headers=headers)
                if dresp.status_code == 200:
                    dsoup = BeautifulSoup(dresp.text, "html.parser")
                    desc_el = dsoup.find(class_=re.compile(r"job-description|ats-description", re.I))
                    text = (desc_el or dsoup).get_text(separator=" ", strip=True)
                    if text:
                        description = text[:3000]
                        spoc_email = extract_spoc_email(text)
            except Exception:
                pass

            jobs.append({
                "title": title,
                "description": description,
                "url": full_url,
                "location": location,
                "date_posted": date_posted,
                "spoc_email": spoc_email,
                "ats": "Radancy",
            })

    for term in DATA_SEARCH_TERMS:
        search_url = (f"{base_url}/search-jobs/{quote(term)}"
                      f"?q={quote(term)}&sortColumn=referencedate&sortDirection=desc")
        try:
            resp = httpx.get(search_url, timeout=20, follow_redirects=True, headers=headers)
            if resp.status_code == 200:
                parse_results(resp.text)
        except Exception as e:
            print(f"  [Radancy] Error searching '{term}' for {company['name']}: {e}")
        time.sleep(0.5)  # polite delay between search terms

    return jobs


def scrape_ashby(company: dict) -> list[dict]:
    """
    Ashby job board API: api.ashbyhq.com/posting-api/job-board/{handle}
    Structured JSON, like Greenhouse. Used by Snowflake, UiPath, and many
    modern scale-ups. The handle may be set explicitly via `ats_handle`,
    otherwise it's derived from the URL or company name.
    """
    url = company["careers_url"]
    handle = company.get("ats_handle") or (
        url.rstrip("/").split("/")[-1] if "ashbyhq.com" in url
        else company["name"].lower().replace(" ", "").replace("-", "").replace("_", "")
    )

    api_url = f"https://api.ashbyhq.com/posting-api/job-board/{handle}"
    jobs = []

    try:
        resp = httpx.get(api_url, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            print(f"  [Ashby] {company['name']}: HTTP {resp.status_code}")
            return []

        for job in resp.json().get("jobs", []):
            if not job.get("isListed", True):
                continue

            title = job.get("title", "").strip()
            if not is_relevant_role(title):
                continue

            addr = (job.get("address") or {}).get("postalAddress", {})
            location = job.get("location", "") or ""
            loc_check = " ".join(filter(None, [
                location, addr.get("addressLocality", ""), addr.get("addressCountry", "")
            ]))
            if not _in_target_location(loc_check):
                continue

            date_posted = job.get("publishedAt", "") or ""
            if not is_posted_within_days(date_posted):
                continue

            description = job.get("descriptionPlain") or BeautifulSoup(
                job.get("descriptionHtml", ""), "html.parser"
            ).get_text(separator=" ")

            if not requires_english(description):
                continue

            jobs.append({
                "title": title,
                "description": description[:3000],
                "url": job.get("jobUrl", ""),
                "location": location or addr.get("addressLocality", ""),
                "date_posted": date_posted,
                "spoc_email": extract_spoc_email(description),
                "ats": "Ashby",
            })

    except Exception as e:
        print(f"  [Ashby] Error scraping {company['name']}: {e}")

    return jobs


def _derive_handle(company: dict) -> str:
    """Explicit ats_handle if set, else company name lowercased without separators."""
    return company.get("ats_handle") or (
        company["name"].lower().replace(" ", "").replace("-", "").replace("_", "")
    )


def scrape_smartrecruiters(company: dict) -> list[dict]:
    """
    SmartRecruiters public API: api.smartrecruiters.com/v1/companies/{id}/postings
    The list omits the description, so the detail endpoint is fetched per relevant
    posting. Used by About You and many European employers.
    """
    handle = _derive_handle(company)
    base = f"https://api.smartrecruiters.com/v1/companies/{handle}/postings"
    jobs = []

    # Paginate — some employers (e.g. ServiceNow ~470) far exceed one page of 100.
    postings = []
    try:
        for offset in range(0, 1000, 100):
            resp = httpx.get(f"{base}?limit=100&offset={offset}", timeout=15,
                             follow_redirects=True)
            if resp.status_code != 200:
                if offset == 0:
                    print(f"  [SmartRecruiters] {company['name']}: HTTP {resp.status_code}")
                break
            page = resp.json().get("content", [])
            postings.extend(page)
            if len(page) < 100:
                break
    except Exception as e:
        print(f"  [SmartRecruiters] Error listing {company['name']}: {e}")
        return []

    try:
        for p in postings:
            title = p.get("name", "")
            if not is_relevant_role(title):
                continue

            loc = p.get("location", {}) or {}
            location = loc.get("fullLocation") or " ".join(filter(None, [
                loc.get("city", ""), loc.get("country", "")
            ]))
            if loc.get("remote"):
                location = (location + " Remote").strip()
            if not _in_target_location(location):
                continue

            date_posted = p.get("releasedDate", "") or ""
            if not is_posted_within_days(date_posted):
                continue

            # Detail call for the full description + canonical URL.
            description = f"Role at {company['name']}."
            job_url = ""
            spoc_email = ""
            try:
                d = httpx.get(f"{base}/{p['id']}", timeout=15, follow_redirects=True).json()
                job_url = d.get("postingUrl") or d.get("applyUrl", "")
                sections = (d.get("jobAd") or {}).get("sections") or {}
                html = " ".join(
                    (sections.get(k) or {}).get("text", "")
                    for k in ("companyDescription", "jobDescription",
                              "qualifications", "additionalInformation")
                )
                text = BeautifulSoup(html, "html.parser").get_text(separator=" ").strip()
                if text:
                    description = text[:3000]
                    spoc_email = extract_spoc_email(text)
            except Exception:
                pass

            jobs.append({
                "title": title,
                "description": description,
                "url": job_url,
                "location": loc.get("fullLocation", location),
                "date_posted": date_posted,
                "spoc_email": spoc_email,
                "ats": "SmartRecruiters",
            })

    except Exception as e:
        print(f"  [SmartRecruiters] Error scraping {company['name']}: {e}")

    return jobs


def scrape_recruitee(company: dict) -> list[dict]:
    """
    Recruitee public API: {handle}.recruitee.com/api/offers/
    Structured JSON with description inline. Used by Adjust and others.
    """
    handle = _derive_handle(company)
    api_url = f"https://{handle}.recruitee.com/api/offers/"
    jobs = []

    try:
        resp = httpx.get(api_url, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            print(f"  [Recruitee] {company['name']}: HTTP {resp.status_code}")
            return []

        for o in resp.json().get("offers", []):
            title = o.get("title", "")
            if not is_relevant_role(title):
                continue

            location = o.get("location") or " ".join(filter(None, [
                o.get("city", ""), o.get("country", "")
            ]))
            if o.get("remote"):
                location = (location + " Remote").strip()
            if not _in_target_location(location):
                continue

            # published_at looks like "2026-06-02 09:50:05 UTC"
            raw = (o.get("published_at") or o.get("created_at") or "").replace(" UTC", "")
            date_posted = ""
            try:
                date_posted = datetime.strptime(
                    raw, "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                pass
            if not is_posted_within_days(date_posted):
                continue

            description = BeautifulSoup(
                o.get("description", "") or "", "html.parser"
            ).get_text(separator=" ").strip()

            jobs.append({
                "title": title,
                "description": description[:3000] or f"Role at {company['name']}.",
                "url": o.get("careers_url", ""),
                "location": location,
                "date_posted": date_posted,
                "spoc_email": extract_spoc_email(description),
                "ats": "Recruitee",
            })

    except Exception as e:
        print(f"  [Recruitee] Error scraping {company['name']}: {e}")

    return jobs


def _parse_workday_url(company: dict):
    """Return (tenant, datacenter, site) from a myworkdayjobs.com URL, else None.

    Reads `workday_url` if present, otherwise `careers_url`. Example:
    https://accenture.wd103.myworkdayjobs.com/en-US/AccentureCareers
      -> ("accenture", "wd103", "AccentureCareers")
    """
    url = company.get("workday_url") or company.get("careers_url", "")
    if "myworkdayjobs.com" not in url:
        return None
    from urllib.parse import urlparse
    host = urlparse(url).netloc
    parts = host.split(".")
    if len(parts) < 3:
        return None
    tenant, dc = parts[0], parts[1]
    site = None
    for seg in [s for s in urlparse(url).path.split("/") if s]:
        if re.fullmatch(r"[a-z]{2}-[A-Za-z]{2}", seg):  # skip locale e.g. en-US
            continue
        site = seg
        break
    if not site:
        return None
    return tenant, dc, site


def _workday_posted_days(posted_on: str) -> int:
    """Parse Workday's relative 'Posted X Days Ago' string into a day count."""
    s = (posted_on or "").lower()
    if "today" in s:
        return 0
    if "yesterday" in s:
        return 1
    m = re.search(r"(\d+)\+?\s*day", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\+?\s*month", s)
    if m:
        return int(m.group(1)) * 30
    return 999  # unknown — don't pre-filter; rely on the detail startDate


def scrape_workday_cxs(company: dict, tenant: str, dc: str, site: str) -> list[dict]:
    """
    Workday via its uniform CXS JSON API (no browser needed):
      POST .../wday/cxs/{tenant}/{site}/jobs   -> listing
      GET  .../wday/cxs/{tenant}/{site}{path}  -> detail (real date + description)
    """
    base = f"https://{tenant}.{dc}.myworkdayjobs.com/wday/cxs/{tenant}/{site}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    jobs = []
    seen: set = set()
    detail_budget = 30  # cap detail calls per company

    for term in ["data engineer", "data analyst", "data scientist",
                 "business intelligence", "machine learning"]:
        try:
            resp = httpx.post(f"{base}/jobs", timeout=15, headers=headers,
                              json={"appliedFacets": {}, "limit": 20,
                                    "offset": 0, "searchText": term})
            if resp.status_code != 200:
                continue

            for jp in resp.json().get("jobPostings", []):
                path = jp.get("externalPath", "")
                if not path or path in seen:
                    continue
                title = jp.get("title", "")
                if not is_relevant_role(title):
                    continue
                # Coarse freshness pre-filter to avoid wasting detail calls.
                if _workday_posted_days(jp.get("postedOn", "")) > 14:
                    continue

                # Cheap location pre-filter: skip concrete out-of-region locations
                # without spending a detail call. "N Locations" is ambiguous, so
                # let those through to the detail check.
                loc_text = jp.get("locationsText", "")
                if (loc_text and not re.search(r"\d+\s+Locations", loc_text)
                        and not _in_target_location(loc_text)):
                    continue
                seen.add(path)

                if detail_budget <= 0:
                    continue
                detail_budget -= 1

                location = loc_text
                date_posted = ""
                description = f"Role at {company['name']}."
                job_url = f"https://{tenant}.{dc}.myworkdayjobs.com/{site}{path}"
                spoc_email = ""
                try:
                    d = httpx.get(f"{base}{path}", timeout=15, headers=headers).json()
                    info = d.get("jobPostingInfo", {}) or {}
                    date_posted = info.get("startDate", "") or ""
                    location = info.get("location", location) or location
                    job_url = info.get("externalUrl", job_url) or job_url
                    text = BeautifulSoup(
                        info.get("jobDescription", "") or "", "html.parser"
                    ).get_text(separator=" ").strip()
                    if text:
                        description = text[:3000]
                        spoc_email = extract_spoc_email(text)
                except Exception:
                    pass

                if not _in_target_location(location):
                    continue
                if date_posted and not is_posted_within_days(date_posted):
                    continue

                jobs.append({
                    "title": title,
                    "description": description,
                    "url": job_url,
                    "location": location,
                    "date_posted": date_posted,
                    "spoc_email": spoc_email,
                    "ats": "Workday",
                })
        except Exception as e:
            print(f"  [Workday] Error ({company['name']} / '{term}'): {e}")

    return jobs


def scrape_workday(company: dict) -> list[dict]:
    """Dispatch: use the CXS JSON API when the Workday tenant URL is known,
    otherwise fall back to the legacy Playwright scraper."""
    parsed = _parse_workday_url(company)
    if parsed:
        return scrape_workday_cxs(company, *parsed)
    return scrape_workday_playwright(company)


# Amazon offices within/near the target region (ISO3 codes used by amazon.jobs).
AMAZON_COUNTRIES = ["DEU", "LUX", "NLD", "AUT", "CHE", "FRA", "BEL"]


def scrape_amazon(company: dict) -> list[dict]:
    """
    Amazon jobs API: www.amazon.jobs/en/search.json (public JSON, no key).
    Queried per target-region country. Amazon is English-first, so the literal
    English-language check is skipped here (most descriptions are English by
    default and rarely contain the word "English").
    """
    base = "https://www.amazon.jobs/en/search.json"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    jobs = []
    seen: set = set()

    for country in AMAZON_COUNTRIES:
        for term in ["data", "machine learning", "business intelligence"]:
            try:
                resp = httpx.get(base, params={
                    "base_query": term, "result_limit": 50,
                    "country": country, "sort": "recent",
                }, timeout=15, follow_redirects=True, headers=headers)
                if resp.status_code != 200:
                    continue

                for j in resp.json().get("jobs", []):
                    jid = j.get("id") or j.get("job_path")
                    if not jid or jid in seen:
                        continue

                    title = j.get("title", "")
                    if not is_relevant_role(title):
                        continue

                    location = j.get("normalized_location") or j.get("location", "")
                    if not _in_target_location(location):
                        continue

                    date_posted = _parse_talentbrew_date(j.get("posted_date", ""))
                    if not is_posted_within_days(date_posted):
                        continue

                    seen.add(jid)
                    raw_desc = " ".join(filter(None, [
                        j.get("description", ""),
                        j.get("basic_qualifications", ""),
                        j.get("preferred_qualifications", ""),
                    ]))
                    description = BeautifulSoup(
                        raw_desc, "html.parser"
                    ).get_text(separator=" ").strip()

                    job_path = j.get("job_path", "")
                    url = (f"https://www.amazon.jobs{job_path}" if job_path
                           else j.get("url_next_step", ""))

                    jobs.append({
                        "title": title,
                        "description": description[:3000] or f"Role at {company['name']}.",
                        "url": url,
                        "location": location,
                        "date_posted": date_posted,
                        "spoc_email": extract_spoc_email(description),
                        "ats": "Amazon",
                    })
            except Exception as e:
                print(f"  [Amazon] Error ({country}/{term}): {e}")
            time.sleep(0.3)

    return jobs


def scrape_company(company: dict) -> list[dict]:
    """Main entry point. Routes to the right scraper based on ATS."""
    ats = company.get("ats", "Proprietary")
    time.sleep(1)  # Polite delay between requests

    if ats == "Greenhouse":
        return scrape_greenhouse(company)
    elif ats == "Lever":
        return scrape_lever(company)
    elif ats == "Ashby":
        return scrape_ashby(company)
    elif ats == "SmartRecruiters":
        return scrape_smartrecruiters(company)
    elif ats == "Recruitee":
        return scrape_recruitee(company)
    elif ats == "Workday":
        return scrape_workday(company)
    elif ats == "Radancy":
        return scrape_radancy(company)
    elif ats == "Amazon":
        return scrape_amazon(company)
    else:
        return scrape_proprietary(company)

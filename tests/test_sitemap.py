"""
Sitemap scraper helpers (BMW-style careers sites): language guess, slug filters, date parsing.
Offline.   Run:  python tests/test_sitemap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scraper  # noqa: E402

DE_TEXT = ("Unser Team der Einsatzänderungssteuerung München entwickelt datengetriebene Lösungen für die "
           "Automobilproduktion der BMW Group. Du unterstützt Projekte zur Datenanalyse und wirkst eng mit den "
           "Fachbereichen zusammen, um neue Nutzungspotenziale zu identifizieren. Wir bieten dir flexible "
           "Arbeitszeiten und ein starkes Team. Deine Aufgaben sind die Analyse von Daten und die Erstellung "
           "von Dashboards. Du bringst Erfahrung und Kenntnisse in Python mit.")
EN_TEXT = ("Our team develops data driven solutions for the automotive production of the BMW Group. You will "
           "support projects on data analysis and work closely with the departments to identify new potential. "
           "We offer flexible working hours and a strong team. Your tasks are the analysis of data and the "
           "creation of dashboards. You bring experience and skills in Python and you are ready to work with us.")


def test_language_guess():
    assert scraper.detect_language(DE_TEXT) == "de"
    assert scraper.detect_language(EN_TEXT) == "en"
    assert scraper.detect_language("too short") == "unknown"


def test_data_role_titles_only():
    assert scraper._data_role_title("München Werkstudent Data Science (wmx) BY 80809")
    assert scraper._data_role_title("München Praktikant Machine Learning für Fahrzeugeffizienz")
    assert scraper._data_role_title("Berlin Werkstudent Reporting und Datenanalyse")
    assert not scraper._data_role_title("München Azubi zum Mechatroniker (wmx) Werk 19 Plätze BY 80809")
    assert not scraper._data_role_title("München Praktikant Events Kommunikation (wmx)")
    assert not scraper._data_role_title("Munich Senior Director of Data Strategy")        # too senior
    assert not scraper._data_role_title("Munich Metal Settlement Specialist")             # 'etl' inside words


def test_germany_detection_from_slug():
    assert scraper._slug_in_germany("München Werkstudent Data Science (wmx) BY 80809")
    assert scraper._slug_in_germany("Leipzig Praktikant Data Analytics SN 04349")
    assert not scraper._slug_in_germany("Spartanburg Data Intern (Spring 2027) South Carolina")
    assert not scraper._slug_in_germany("Woodcliff Lake Lead Data Scientist NJ 07677 USA".replace("NJ", "NJ"))


def test_posting_date_parsing_german_and_english_formats():
    assert scraper._parse_job_page_date("Startdatum der Ausschreibung:\n03.09.26\nStellenstandort")[:10] == "2026-09-03"
    assert scraper._parse_job_page_date("Posting Start Date:\n9/3/26\nJob Location")[:10] == "2026-09-03"
    assert scraper._parse_job_page_date("no date here") == ""


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

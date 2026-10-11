"""
Language rule: keep a posting when English is needed (or the posting is in English) and German is
at most optional; drop it when German is required or mentioned without being called optional.
Offline.   Run:  python tests/test_language.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scraper  # noqa: E402

EN = " Lorem ipsum filler about the role and the team we are building together every day. " * 4
DE = (" Wir suchen Dich für unser Team in München und entwickeln gemeinsam datengetriebene Lösungen für die "
      "Produktion mit Freude an der Arbeit und dem Austausch mit den Kollegen. ") * 3

# (sentence, expected ok, padding that sets the posting language)
CASES = [
    ("Fluent English required; German is a plus.", True, EN),
    ("Business fluent English is a must. German language skills are nice to have.", True, EN),
    ("Fluent in English and German is required.", False, EN),
    ("Excellent command of English. Deutschkenntnisse sind von Vorteil.", True, DE),
    ("Sehr gute Deutsch- und Englischkenntnisse erforderlich.", False, DE),
    ("Verhandlungssichere Deutschkenntnisse sowie gute Englischkenntnisse.", False, DE),
    ("Verhandlungssicheres Englisch, Deutschkenntnisse von Vorteil.", True, DE),
    ("Fließend Englisch in Wort und Schrift. Deutschkenntnisse wünschenswert.", True, DE),
    ("Gute Deutsch- und Englischkenntnisse", False, DE),
    ("German language skills are not required.", True, EN),
    ("No German required, English is our working language.", True, EN),
    ("You speak German (B2) and English.", False, EN),
    ("We work with German customers and the German market.", True, EN),
    ("Our team speaks English daily.", True, EN),
    ("Good knowledge of German is a plus, fluent English required.", True, EN),
    ("English (required) and German (nice to have).", True, EN),
    ("Deutschkenntnisse erforderlich, Englisch von Vorteil.", False, DE),
    ("German is a must, English is a plus.", False, EN),
    ("Fluent English (C1) is required. German (A2) is welcome but not necessary.", True, EN),
    ("Du sprichst fließend Deutsch und gutes Englisch.", False, DE),
    ("Gute Englischkenntnisse, Deutsch ist nicht erforderlich.", True, DE),
    ("Knowledge of German is an advantage but English is the team language.", True, EN),
    ("German is required but English is a plus.", False, EN),
    ("Strong analytical skills. Experience with German tax law is helpful.", True, EN),
    ("You are based in Germany. Python and SQL are required.", True, EN),
]


def test_language_cases():
    wrong = []
    for sentence, want, pad in CASES:
        got = scraper.assess_language(sentence + pad)
        if got["ok"] != want:
            wrong.append((sentence, want, got["note"]))
    assert not wrong, wrong


def test_german_posting_without_any_statement_is_dropped():
    assert not scraper.assess_language(DE)["ok"]


def test_english_posting_that_never_mentions_german_passes():
    r = scraper.assess_language("We are building data pipelines with Python, SQL and Airflow." + EN)
    assert r["ok"] and r["german"] == "none"


def test_short_or_placeholder_text_is_not_judged():
    assert scraper.assess_language("Role at Acme. Visit https://x for full description.")["ok"]
    assert scraper.assess_language("")["ok"]


def test_company_names_are_not_language_requirements():
    r = scraper.assess_language("You will work with Deutsche Bank and Deutsche Telekom in Deutschland on data projects." + EN)
    assert r["ok"] and r["german"] == "none"


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

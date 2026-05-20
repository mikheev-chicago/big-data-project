#!/usr/bin/env python3
"""
Phase 0 — Fed Chair speech scraper.
Fetches all speeches by Bernanke, Yellen, and Powell from federalreserve.gov (2008-2025).
Saves each speech as a .txt file and writes a manifest to data/processed/speeches.csv.

Usage:
    python src/scrape_speeches.py

Output:
    data/raw/speeches/YYYYMMDD_<chair>_<title>.txt  — full speech text
    data/processed/speeches.csv                      — one row per speech
"""

import re
import csv
import time
import logging
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "speeches"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "https://www.federalreserve.gov"
YEARS = range(2008, 2026)   # 2008 through 2025 inclusive
RATE_LIMIT = 1.2            # seconds between HTTP requests (be polite)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Fed Chair registry ────────────────────────────────────────────────────────
# "start" and "end" = the dates each person served as Chair.
# Speeches given before/after their chairmanship are excluded even if their
# name appears (e.g. Powell was a Governor before 2018).
CHAIRS = [
    {
        "key":     "bernanke",
        "pattern": r"bernanke",
        "start":   datetime(2006, 2,  1),
        "end":     datetime(2014, 2,  3),
        "name":    "Ben S. Bernanke",
    },
    {
        "key":     "yellen",
        "pattern": r"yellen",
        "start":   datetime(2014, 2,  3),
        "end":     datetime(2018, 2,  3),
        "name":    "Janet L. Yellen",
    },
    {
        "key":     "powell",
        "pattern": r"powell",
        "start":   datetime(2018, 2,  5),
        "end":     datetime(2025, 12, 31),
        "name":    "Jerome H. Powell",
    },
]


def identify_chair(speaker_text: str, speech_date: datetime):
    """
    Returns (canonical_name, chair_key) if the speaker was serving as Chair
    on speech_date. Returns (None, None) otherwise.

    We check the name AND the date range so we don't accidentally pick up
    speeches Powell gave as a Governor before Feb 2018.
    """
    lower = speaker_text.lower()
    for chair in CHAIRS:
        if re.search(chair["pattern"], lower):
            if chair["start"] <= speech_date <= chair["end"]:
                return chair["name"], chair["key"]
    return None, None


def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (academic research; "
            "BUSN 20800 final project; "
            "contact: v.mikheev9@gmail.com)"
        )
    })
    return s


def fetch(session, url, retries=3):
    """GET a URL with retries and exponential back-off."""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            wait = 2 ** attempt
            log.warning(f"Attempt {attempt+1} failed ({url}): {exc}. Retrying in {wait}s")
            time.sleep(wait)
    log.error(f"All {retries} attempts failed: {url}")
    return None


def parse_listing_date(text: str):
    """Parse 'M/D/YYYY' (the format the Fed uses on listing pages)."""
    try:
        return datetime.strptime(text.strip(), "%m/%d/%Y")
    except ValueError:
        return None


def list_speeches_for_year(session, year: int):
    """
    Fetch the annual speech index page and return a list of dicts:
        {date, speaker, title, url}

    The Fed uses two URL patterns:
        Modern: /newsevents/speech/{year}-speeches.htm
        Legacy: /newsevents/speech/{year}speech.htm
    Both return the same HTML structure, so we try the modern one first.
    """
    for pattern in [f"{year}-speeches.htm", f"{year}speech.htm"]:
        url = f"{BASE_URL}/newsevents/speech/{pattern}"
        resp = fetch(session, url)
        time.sleep(RATE_LIMIT)
        if resp and resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Quick sanity check: does the page actually have a speech list?
            if soup.select_one("div.row.eventlist"):
                log.info(f"  Loaded listing from {url}")
                return _parse_listing_page(soup)

    log.error(f"  Could not load listing page for {year}")
    return []


def _parse_listing_page(soup):
    """
    Parse the standard Fed speech listing HTML.

    Structure (same across all years):
        <div class="row eventlist">
          <div class="col-xs-12 col-sm-8 col-md-8">
            <div class="row">                           ← one per speech
              <div class="eventlist__time">
                <time>11/14/2024</time>
              </div>
              <div class="eventlist__event">
                <p><a href="/newsevents/speech/powell...htm"><em>Title</em></a></p>
                <p class="news__speaker">Chair Jerome H. Powell</p>
                <p>At Some Venue, City, State</p>
              </div>
            </div>
          </div>
        </div>
    """
    entries = []
    rows = soup.select("div.row.eventlist div.col-xs-12 div.row")

    for row in rows:
        time_el    = row.select_one("div.eventlist__time time")
        link_el    = row.select_one("div.eventlist__event p a")
        speaker_el = row.select_one("div.eventlist__event p.news__speaker")

        if not (time_el and link_el):
            continue

        date = parse_listing_date(time_el.get_text(strip=True))
        if not date:
            continue

        speaker = speaker_el.get_text(strip=True) if speaker_el else ""
        title   = link_el.get_text(strip=True)
        href    = link_el.get("href", "")
        url     = (BASE_URL + href) if href.startswith("/") else href

        entries.append({"date": date, "speaker": speaker, "title": title, "url": url})

    return entries


def fetch_speech_text(session, url: str) -> str:
    """
    Fetch a single speech page and return the body text.
    The Fed wraps speech content in <div id="article">.
    Falls back to collecting all long <p> tags if that's missing.
    """
    resp = fetch(session, url)
    time.sleep(RATE_LIMIT)
    if not resp:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    article = soup.select_one("div#article")
    if article:
        # Strip footnotes and share/video widgets inside the article
        for noise in article.select("div.footnotes, .icon-share, script, style"):
            noise.decompose()
        return article.get_text("\n", strip=True)

    # Fallback: grab all substantial paragraphs from the page
    return "\n\n".join(
        p.get_text(strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(strip=True)) > 80
    )


def safe_filename(date: datetime, chair_key: str, title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title)
    slug = re.sub(r"\s+", "_", slug.strip())[:60]
    return f"{date.strftime('%Y%m%d')}_{chair_key}_{slug}.txt"


def main():
    session = make_session()
    records  = []
    skipped  = 0

    for year in YEARS:
        log.info(f"── {year} " + "─" * 50)
        entries = list_speeches_for_year(session, year)
        log.info(f"  {len(entries)} speeches on listing page")

        year_kept = 0
        for entry in entries:
            chair_name, chair_key = identify_chair(entry["speaker"], entry["date"])
            if not chair_name:
                continue    # not a Chair speech — skip

            log.info(
                f"  + {entry['date'].date()}  {chair_name}  \"{entry['title'][:55]}\""
            )

            text = fetch_speech_text(session, entry["url"])
            word_count = len(text.split())

            if word_count < 50:
                log.warning(f"    Skipping — only {word_count} words: {entry['url']}")
                skipped += 1
                continue

            fname = safe_filename(entry["date"], chair_key, entry["title"])
            (RAW_DIR / fname).write_text(text, encoding="utf-8")

            records.append({
                "date":       entry["date"].strftime("%Y-%m-%d"),
                "year":       entry["date"].year,
                "chair":      chair_name,
                "chair_key":  chair_key,
                "title":      entry["title"],
                "url":        entry["url"],
                "word_count": word_count,
                "filename":   fname,
            })
            year_kept += 1

        log.info(f"  Kept {year_kept} chair speech(es) for {year}")

    # ── Write CSV manifest ────────────────────────────────────────────────────
    if records:
        out = PROCESSED_DIR / "speeches.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        log.info(f"\nDone. {len(records)} speeches saved → {out}")
        log.info(f"Skipped {skipped} (too short or empty).")
    else:
        log.warning("No speeches collected. Check network or selector logic.")


if __name__ == "__main__":
    main()

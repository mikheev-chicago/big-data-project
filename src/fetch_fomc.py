#!/usr/bin/env python3
"""
Phase 0 — FOMC meeting calendar fetcher.

Scrapes historical FOMC meeting announcement dates from federalreserve.gov
(2008–2025) and saves them to data/processed/fomc_calendar.csv.

We use the announcement date (the last day of each meeting) since that is when
the policy statement is released and markets move. This date is extracted from
FOMC statement links embedded in each year's archive page — links follow the
pattern /newsevents/pressreleases/monetary{YYYYMMDD}*.htm.

Usage:
    python src/fetch_fomc.py

Output:
    data/processed/fomc_calendar.csv   — columns: date, year
"""

import re
import time
import logging
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL   = "https://www.federalreserve.gov"
RATE_LIMIT = 1.2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

YEARS = range(2008, 2026)

# FOMC statement links encode the announcement date: monetary{YYYYMMDD}a*.htm
_LINK_DATE_RE = re.compile(r"/monetary(\d{8})[a-z]", re.IGNORECASE)

# Text fallback: "January 28-29" or "March 15" — captures end day of multi-day meetings
_TEXT_DATE_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august"
    r"|september|october|november|december)"
    r"\s+(\d{1,2})(?:\s*[–\-]\s*(\d{1,2}))?",
    re.IGNORECASE,
)
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


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


def fetch_page(session, url, retries=3):
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


def dates_from_links(soup, year):
    """Primary method: extract announcement dates from embedded statement links."""
    dates = set()
    for a in soup.find_all("a", href=True):
        m = _LINK_DATE_RE.search(a["href"])
        if m:
            try:
                d = datetime.strptime(m.group(1), "%Y%m%d").date()
                if d.year == year:
                    dates.add(d)
            except ValueError:
                pass
    return dates


def dates_from_text(soup, year):
    """Fallback: parse meeting dates from page text when link method finds nothing."""
    dates = set()
    for m in _TEXT_DATE_RE.finditer(soup.get_text(" ")):
        month = _MONTHS[m.group(1).lower()]
        # Multi-day meetings: take the end day (announcement day)
        day = int(m.group(3)) if m.group(3) else int(m.group(2))
        try:
            from datetime import date
            dates.add(date(year, month, day))
        except ValueError:
            pass
    return dates


def fetch_year(session, year):
    """
    Try the year-specific historical archive first; fall back to the main
    calendar page (which covers recent and upcoming meetings).
    """
    urls = [
        f"{BASE_URL}/monetarypolicy/fomchistorical{year}.htm",
        f"{BASE_URL}/monetarypolicy/fomccalendars.htm",
    ]

    for url in urls:
        resp = fetch_page(session, url)
        time.sleep(RATE_LIMIT)
        if not resp:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        dates = dates_from_links(soup, year)
        if dates:
            log.info(f"  {year}: {len(dates)} meeting(s) via link parsing  [{url.split('/')[-1]}]")
            return sorted(dates)

        dates = dates_from_text(soup, year)
        if dates:
            log.info(f"  {year}: {len(dates)} meeting(s) via text parsing (fallback)")
            return sorted(dates)

    log.warning(f"  {year}: no dates found")
    return []


def main():
    session = make_session()
    records = []

    for year in YEARS:
        log.info(f"── {year}")
        for d in fetch_year(session, year):
            records.append({"date": d.isoformat(), "year": year})

    if not records:
        log.error("No FOMC dates collected — check network or page structure.")
        return

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    out_path = PROCESSED_DIR / "fomc_calendar.csv"
    df.to_csv(out_path, index=False)
    log.info(f"\nDone. {len(df)} FOMC meeting dates → {out_path}")
    log.info(f"Coverage: {df['date'].iloc[0]}  to  {df['date'].iloc[-1]}")


if __name__ == "__main__":
    main()

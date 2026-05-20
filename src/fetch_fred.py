#!/usr/bin/env python3
"""
Phase 0 — FRED macro data fetcher.

Pulls five series and aligns a macro snapshot to each speech date:

  DGS2             2-year Treasury yield         (daily)    → outcome variable
  DFF              Effective federal funds rate   (daily)    → LLM scoring context
  PCEPILFE         Core PCE price index          (monthly)  → compute YoY % change
  UNRATE           Unemployment rate             (monthly)  → LLM scoring context
  A191RL1Q225SBEA  Real GDP growth rate          (quarterly, annualized SAAR %)

Alignment rule: for each speech date, use the most recently *released* observation
(last observation date <= speech date). This avoids lookahead bias — we never
attach data the market couldn't have known at the time.

Usage:
    export FRED_API_KEY=your_key_here
    python src/fetch_fred.py

Requires speeches.csv in data/processed/ (run scrape_speeches.py first).

Output:
    data/raw/fred/{SERIES_ID}.csv      — raw FRED series
    data/processed/macro_context.csv   — one row per speech with aligned macro snapshot
"""

import os
import sys
import logging
from pathlib import Path

import pandas as pd
from fredapi import Fred

BASE_DIR      = Path(__file__).resolve().parent.parent
RAW_FRED_DIR  = BASE_DIR / "data" / "raw" / "fred"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_FRED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Pull one extra year so PCE YoY calculations are valid starting Jan 2008
START_DATE = "2007-01-01"
END_DATE   = "2025-12-31"

SERIES = {
    "DGS2":            "2-year Treasury yield (daily, %)",
    "DFF":             "Effective federal funds rate (daily, %)",
    "PCEPILFE":        "Core PCE price index (monthly, level — YoY % derived below)",
    "UNRATE":          "Unemployment rate (monthly, %)",
    "A191RL1Q225SBEA": "Real GDP growth rate (quarterly, annualized SAAR %)",
}


def pull_series(fred: Fred, series_id: str, desc: str) -> pd.Series:
    log.info(f"Pulling {series_id}: {desc}")
    s = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
    s.name = series_id
    df = s.dropna().reset_index()
    df.columns = ["date", "value"]
    out = RAW_FRED_DIR / f"{series_id}.csv"
    df.to_csv(out, index=False)
    log.info(f"  {len(df):,} observations → {out.name}")
    return s


def align_snapshot(speeches: pd.DataFrame, raw: dict) -> pd.DataFrame:
    """
    For each speech date find the most recently available value of each macro
    series using a backwards merge (last observation date <= speech date).
    Returns a DataFrame with one row per speech date (deduplicated for efficiency,
    then merged back so each speech row gets its own snapshot).
    """
    pce_yoy = raw["PCEPILFE"].pct_change(12) * 100
    pce_yoy.name = "PCE_YOY"

    lookup = {
        "DGS2":       (raw["DGS2"],  0),   # yield on speech date
        "DGS2_prev":  (raw["DGS2"], -1),   # yield on prior trading day (for computing day-of change)
        "DFF":        (raw["DFF"],   0),
        "PCE_YOY":    (pce_yoy,      0),
        "UNRATE":     (raw["UNRATE"],0),
        "GDP_GROWTH": (raw["A191RL1Q225SBEA"], 0),
    }

    speech_dates = speeches[["date"]].copy()
    speech_dates["date"] = pd.to_datetime(speech_dates["date"])
    unique_dates = speech_dates.drop_duplicates().sort_values("date").reset_index(drop=True)

    result = unique_dates.copy()

    for col, (s, day_offset) in lookup.items():
        ref = s.dropna().reset_index()
        ref.columns = ["date", col]
        ref["date"] = pd.to_datetime(ref["date"])
        if col == "GDP_GROWTH":
            ref["date"] = ref["date"] + pd.DateOffset(months=3, days=30)
        ref = ref.sort_values("date")
        # For DGS2_prev, look up against speech_date - 1 day so we get the
        # last available yield strictly before the speech date.
        lookup_dates = unique_dates.copy()
        if day_offset == -1:
            lookup_dates = lookup_dates.copy()
            lookup_dates["date"] = lookup_dates["date"] - pd.Timedelta(days=1)
        merged = pd.merge_asof(lookup_dates, ref, on="date", direction="backward")
        result[col] = merged[col].values

    return result


def main():
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        sys.exit(
            "Error: FRED_API_KEY environment variable not set.\n"
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then run:  export FRED_API_KEY=your_key_here"
        )

    fred = Fred(api_key=api_key)

    raw = {}
    for series_id, desc in SERIES.items():
        raw[series_id] = pull_series(fred, series_id, desc)

    speeches_path = PROCESSED_DIR / "speeches.csv"
    if not speeches_path.exists():
        log.warning("speeches.csv not found — skipping alignment step.")
        log.warning("Re-run fetch_fred.py after scrape_speeches.py finishes to produce macro_context.csv.")
        return

    speeches = pd.read_csv(speeches_path)
    log.info(f"\nAligning macro snapshots to {len(speeches)} speeches...")

    snapshot = align_snapshot(speeches, raw)

    speeches["date"] = pd.to_datetime(speeches["date"])
    out = speeches.merge(snapshot, on="date", how="left")

    out_path = PROCESSED_DIR / "macro_context.csv"
    out.to_csv(out_path, index=False)
    log.info(f"Saved → {out_path}  ({len(out)} rows, {out.shape[1]} columns)")

    macro_cols = ["DGS2", "DFF", "PCE_YOY", "UNRATE", "GDP_GROWTH"]
    missing = out[macro_cols].isna().sum()
    if missing.any():
        log.warning("Missing values after alignment:\n" + str(missing[missing > 0]))
    else:
        log.info("All macro columns fully populated — alignment complete.")


if __name__ == "__main__":
    main()

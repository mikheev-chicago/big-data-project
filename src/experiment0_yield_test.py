#!/usr/bin/env python3
"""
Experiment 0: Score, rank, and yield-test Fed Chair speeches

Steps:
  1. Remove evening speeches (delivered after market close)
  2. Score remaining speeches with Loughran-McDonald lexicon
  3. Rank within each regime and bin into thirds (Dovish / Neutral / Hawkish)
  4. For Hawkish and Dovish speeches, test whether same-day 2yr yield moved
     in the predicted direction (Hawkish → Up, Dovish → Down)
  5. Save one rankings CSV per regime + one per-speech yield results CSV

Usage:
    python3 src/experiment0_yield_test.py
"""

import sys
from pathlib import Path

import pandas as pd

try:
    import pysentiment2 as ps
except ImportError:
    sys.exit("Missing dependency — run: pip install pysentiment2")

ROOT       = Path(__file__).resolve().parents[1]
PROCESSED  = ROOT / "data" / "processed"
SPEECH_DIR = ROOT / "data" / "raw" / "speeches"

# Delivered at dinner/gala events after market close — removed before ranking
# so they don't affect the thirds split
EVENING_SPEECHES = {
    "20100408_bernanke_Economic_Policy_Lessons_from_History.txt",
    "20131119_bernanke_Communication_and_Monetary_Policy.txt",
    "20190228_powell_Recent_Economic_Developments_and_Longer-Term_Challenges.txt",
}

CHAIRS = [
    ("bernanke", "Bernanke", "Chair A", "chair_a"),
    ("yellen",   "Yellen",   "Chair B", "chair_b"),
    ("powell",   "Powell",   "Chair C", "chair_c"),
]


def lm_hawk_score(text: str, lm) -> float:
    """(LM Positive − Negative) / stemmed-token count."""
    tokens = lm.tokenize(text)
    if not tokens:
        return 0.0
    s = lm.get_score(tokens)
    return (int(s["Positive"]) - int(s["Negative"])) / len(tokens)


def assign_buckets(series: pd.Series) -> pd.Series:
    """Equal-thirds bucketing by rank. Ties get average rank."""
    n     = len(series)
    ranks = series.rank(method="average")
    out   = pd.Series("Neutral", index=series.index, dtype=object)
    out[ranks <= n / 3]    = "Dovish"
    out[ranks > 2 * n / 3] = "Hawkish"
    return out


def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    speeches = pd.read_csv(PROCESSED / "speeches_monetary_policy.csv", parse_dates=["date"])
    macro    = pd.read_csv(PROCESSED / "macro_context.csv",            parse_dates=["date"])

    speeches = speeches.merge(
        macro[["filename", "DGS2", "DGS2_prev"]],
        on="filename", how="left"
    )
    speeches["yield_change"] = speeches["DGS2"] - speeches["DGS2_prev"]

    # ── Remove evening speeches ────────────────────────────────────────────────
    before = len(speeches)
    speeches = speeches[~speeches["filename"].isin(EVENING_SPEECHES)].copy()
    print(f"Removed {before - len(speeches)} evening speeches → {len(speeches)} remaining")

    # ── Score ──────────────────────────────────────────────────────────────────
    print("Scoring with Loughran-McDonald lexicon…")
    lm     = ps.LM()
    scores = []
    for _, row in speeches.iterrows():
        fpath = SPEECH_DIR / row["filename"]
        text  = fpath.read_text(errors="ignore") if fpath.exists() else ""
        scores.append(lm_hawk_score(text, lm))
    speeches["lm_score"] = scores

    # ── Per-regime: rank, bucket, save, yield test ─────────────────────────────
    all_yield_rows = []

    for key, label, anon_label, anon_key in CHAIRS:
        sub = speeches[speeches["chair_key"] == key].copy()
        sub = sub.sort_values("date").reset_index(drop=True)

        sub["rank"]   = sub["lm_score"].rank(method="average").astype(int)
        sub["bucket"] = assign_buckets(sub["lm_score"])
        sub["chair"]  = anon_label
        sub["filename"] = sub["filename"].str.replace(f"_{key}_", f"_{anon_key}_", regex=False)

        # Save rankings file
        rankings_path = PROCESSED / f"{anon_key}_rankings.csv"
        sub[["date", "chair", "title", "filename", "lm_score", "rank", "bucket"]].to_csv(
            rankings_path, index=False
        )

        # Yield test — Hawkish and Dovish only
        test = sub[sub["bucket"].isin(["Hawkish", "Dovish"])].copy()
        test["yield_dir"] = test["yield_change"].apply(
            lambda x: "Up" if x > 0 else ("Down" if x < 0 else "Flat")
        )
        test["correct"] = (
            ((test["bucket"] == "Hawkish") & (test["yield_dir"] == "Up")) |
            ((test["bucket"] == "Dovish")  & (test["yield_dir"] == "Down"))
        )
        test["regime"] = anon_label
        all_yield_rows.append(test)

        # Print summary
        hits  = int(test["correct"].sum())
        total = len(test[test["yield_change"].notna()])
        print(f"\n{label} → {anon_label}  ({len(sub)} speeches after removing evening)")
        print(f"  Dovish {(sub['bucket']=='Dovish').sum()}  |  Neutral {(sub['bucket']=='Neutral').sum()}  |  Hawkish {(sub['bucket']=='Hawkish').sum()}")
        print(f"  Yield test (Hawkish + Dovish only): {hits}/{total}  ({hits/total:.1%})")

    # ── Save per-speech yield results ──────────────────────────────────────────
    results = pd.concat(all_yield_rows, ignore_index=True)
    out_cols = ["date", "regime", "title", "bucket", "lm_score", "rank",
                "yield_change", "yield_dir", "correct"]
    results_path = PROCESSED / "experiment0_yield_results.csv"
    results[out_cols].to_csv(results_path, index=False)

    # ── Overall ────────────────────────────────────────────────────────────────
    testable = results[results["yield_change"].notna()]
    total_hits  = int(testable["correct"].sum())
    total_total = len(testable)
    print(f"\n{'─'*45}")
    print(f"OVERALL: {total_hits}/{total_total}  ({total_hits/total_total:.1%})")
    print(f"{'─'*45}")
    print(f"\nPer-speech results → {results_path}")


if __name__ == "__main__":
    main()

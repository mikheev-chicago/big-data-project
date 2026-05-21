#!/usr/bin/env python3
"""
Phase 2 — LLM pairwise hawkishness scoring.

Four stages:
  1. Sentence filter     — Haiku classifies sentences as monetary-policy relevant
  2. Speech repr.        — aggregate filtered damped_lemmas per speech + attach macro
  3. Round-robin tournament — Sonnet compares every pair within each regime
  4. TrueSkill           — aggregate pair outcomes into 0-100 hawkishness scores

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python src/pairwise_scoring.py

Outputs (per regime + combined):
    data/processed/filtered_sentences_{chair}.csv
    data/processed/pairwise_results_{chair}.csv
    data/processed/speech_scores_phase2_{chair}.csv
    data/processed/speech_scores_phase2.csv
"""

import asyncio
import json
import logging
import os
import random
import sys
from itertools import combinations
from pathlib import Path

import anthropic
import pandas as pd
import trueskill

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

FILTER_MODEL   = "claude-haiku-4-5-20251001"
COMPARE_MODEL  = "claude-sonnet-4-6"
BATCH_SIZE     = 50
MAX_CONCURRENT = 40

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

FILTER_SYSTEM = """\
You are a classifier for Federal Reserve speech fragments.

Return a JSON array of the 0-based index numbers of any fragment that discusses:
monetary policy stance, interest rates, inflation, employment, unemployment,
economic growth, GDP, or the Fed's dual mandate (price stability and maximum employment).

Exclude fragments about: bank supervision, financial regulation, capital requirements,
consumer protection, community banking, stress tests, or other non-monetary topics.

Return ONLY a valid JSON array of integers. Example: [0, 2, 5]
If no fragments qualify, return: []"""

COMPARE_SYSTEM = """\
You are evaluating anonymized Federal Reserve speeches to determine which signals \
a more hawkish monetary policy stance.

Hawkish = more concerned about inflation, more willing to raise or hold rates elevated.
Dovish = more concerned about supporting employment and growth, more willing to cut \
or hold rates low.

You will see key terms extracted from two speeches plus the economic conditions at \
the time each was delivered. Judge which speech signals a more hawkish stance given \
its economic context.

Reply with only 'A' or 'B'. Nothing else."""


# ── Stage helpers ──────────────────────────────────────────────────────────────

def generate_pairs(filenames: list[str]) -> list[tuple[str, str]]:
    """
    Return all N*(N-1)/2 unordered pairs for a round-robin tournament.
    Each pair is randomly assigned A/B position to eliminate position bias.
    """
    pairs = list(combinations(filenames, 2))
    result = []
    for a, b in pairs:
        if random.random() < 0.5:
            a, b = b, a
        result.append((a, b))
    random.shuffle(result)
    return result


def build_speech_representations(
    filtered_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Aggregate filtered sentences into one representation dict per speech.

    Groups by filename, joins damped_lemmas with spaces, then inner-joins
    macro context (DFF, PCE_YOY, UNRATE, GDP_GROWTH) from macro_df.
    Speeches with no macro row are excluded.

    Returns a dict keyed by filename. Each value has keys:
        damped_lemmas, date, title, DFF, PCE_YOY, UNRATE, GDP_GROWTH
    """
    agg = (
        filtered_df
        .groupby("filename", as_index=False)
        .agg(
            date         =("date",         "first"),
            title        =("title",        "first"),
            damped_lemmas=("damped_lemmas", lambda x: " ".join(x.dropna())),
        )
    )
    merged = agg.merge(
        macro_df[["filename", "DFF", "PCE_YOY", "UNRATE", "GDP_GROWTH"]],
        on="filename",
        how="inner",
    )
    return {
        row["filename"]: {
            "damped_lemmas": row["damped_lemmas"],
            "date":          str(row["date"]),
            "title":         row["title"],
            "DFF":           float(row["DFF"]),
            "PCE_YOY":       float(row["PCE_YOY"]),
            "UNRATE":        float(row["UNRATE"]),
            "GDP_GROWTH":    float(row["GDP_GROWTH"]),
        }
        for _, row in merged.iterrows()
    }


def compute_trueskill(
    results_df: pd.DataFrame,
    filenames: list[str],
) -> pd.DataFrame:
    """
    Feed pairwise outcomes into TrueSkill and return per-speech scores.

    draw_probability=0 because the LLM always picks A or B.
    Rows where winner is null/NaN are skipped.
    hawkishness_phase2 is min-max normalized to [0, 100] within this regime.

    Returns a DataFrame with columns:
        filename, trueskill_mu, trueskill_sigma, hawkishness_phase2
    """
    env = trueskill.TrueSkill(draw_probability=0.0)
    ratings = {f: env.create_rating() for f in filenames}

    for _, row in results_df.iterrows():
        winner = row["winner"]
        if winner is None or (isinstance(winner, float) and pd.isna(winner)):
            continue
        a, b = row["filename_a"], row["filename_b"]
        if winner == a:
            ratings[a], ratings[b] = env.rate_1vs1(ratings[a], ratings[b])
        else:
            ratings[b], ratings[a] = env.rate_1vs1(ratings[b], ratings[a])

    rows = [
        {"filename": f, "trueskill_mu": r.mu, "trueskill_sigma": r.sigma}
        for f, r in ratings.items()
    ]
    df = pd.DataFrame(rows)

    mu_min = df["trueskill_mu"].min()
    mu_max = df["trueskill_mu"].max()
    if mu_max > mu_min:
        df["hawkishness_phase2"] = (
            (df["trueskill_mu"] - mu_min) / (mu_max - mu_min) * 100
        )
    else:
        df["hawkishness_phase2"] = 50.0

    return df

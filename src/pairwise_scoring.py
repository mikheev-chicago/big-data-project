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


async def filter_sentences_batch(
    sentences: list[str],
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
) -> list[int]:
    """
    Classify a batch of sentences. Returns 0-based indices of those that
    discuss monetary policy, inflation, employment, or economic growth.
    Falls back to including all indices if the API fails or returns bad JSON.
    """
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.messages.create(
                    model=FILTER_MODEL,
                    max_tokens=256,
                    system=FILTER_SYSTEM,
                    messages=[{"role": "user", "content": numbered}],
                )
                indices = json.loads(resp.content[0].text.strip())
                return [i for i in indices if 0 <= i < len(sentences)]
            except (json.JSONDecodeError, ValueError):
                if attempt == 2:
                    return list(range(len(sentences)))
                await asyncio.sleep(2 ** attempt)
            except anthropic.APIError:
                if attempt == 2:
                    return list(range(len(sentences)))
                await asyncio.sleep(2 ** attempt)
    return list(range(len(sentences)))


async def filter_all_sentences(
    df: pd.DataFrame,
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
) -> pd.DataFrame:
    """
    Run the sentence filter across the full DataFrame in batches of BATCH_SIZE.
    Dispatches all batches concurrently (semaphore limits to MAX_CONCURRENT calls).
    Returns a filtered DataFrame with only monetary-policy relevant rows.
    """
    sentences = df["text"].tolist()
    batches = [sentences[i:i + BATCH_SIZE] for i in range(0, len(sentences), BATCH_SIZE)]
    batch_starts = list(range(0, len(sentences), BATCH_SIZE))

    tasks = [filter_sentences_batch(batch, client, semaphore) for batch in batches]
    batch_results = await asyncio.gather(*tasks)

    keep_positions: list[int] = []
    for start, indices in zip(batch_starts, batch_results):
        keep_positions.extend(start + i for i in indices)

    return df.iloc[sorted(keep_positions)].reset_index(drop=True)


def _format_speech_block(label: str, rep: dict) -> str:
    return (
        f"SPEECH {label}\n"
        f"Economic conditions: {rep['date']} | "
        f"Fed funds rate: {rep['DFF']:.2f}% | "
        f"Core PCE: {rep['PCE_YOY']:.1f}% YoY | "
        f"Unemployment: {rep['UNRATE']:.1f}% | "
        f"GDP growth: {rep['GDP_GROWTH']:.1f}% annualized\n"
        f"Key terms: {rep['damped_lemmas']}"
    )


async def compare_pair(
    filename_a: str,
    filename_b: str,
    repr_a: dict,
    repr_b: dict,
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str, str | None]:
    """
    Compare two speeches. Returns (filename_a, filename_b, winner_filename).
    winner_filename is None if the LLM response is not 'A' or 'B' after two attempts.
    """
    user_msg = (
        _format_speech_block("A", repr_a)
        + "\n\n"
        + _format_speech_block("B", repr_b)
        + "\n\nWhich speech signals a more hawkish stance given its economic context? Reply A or B."
    )
    async with semaphore:
        for attempt in range(2):
            try:
                resp = await client.messages.create(
                    model=COMPARE_MODEL,
                    max_tokens=8,
                    system=COMPARE_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                choice = resp.content[0].text.strip().upper()
                if choice == "A":
                    return filename_a, filename_b, filename_a
                if choice == "B":
                    return filename_a, filename_b, filename_b
            except anthropic.APIError:
                if attempt == 0:
                    await asyncio.sleep(2)
    return filename_a, filename_b, None


async def run_tournament(
    pairs: list[tuple[str, str]],
    representations: dict[str, dict],
    client: anthropic.AsyncAnthropic,
    results_path: Path,
    semaphore: asyncio.Semaphore,
) -> pd.DataFrame:
    """
    Run all pairwise comparisons concurrently (semaphore limits to MAX_CONCURRENT).
    Saves results every 200 comparisons for crash-safety.
    Resumes from existing results file if present — skips already-done pairs.
    Uses frozenset for done-pair lookup so (a,b) and (b,a) are treated as the same pair.
    """
    results: list[dict] = []

    if results_path.exists():
        existing = pd.read_csv(results_path)
        done = {
            frozenset([row["filename_a"], row["filename_b"]])
            for _, row in existing.iterrows()
        }
        results = existing.to_dict("records")
        pairs = [(a, b) for a, b in pairs if frozenset([a, b]) not in done]
        log.info(f"  Resuming: {len(results)} done, {len(pairs)} remaining")

    tasks = [
        compare_pair(a, b, representations[a], representations[b], client, semaphore)
        for a, b in pairs
    ]

    completed = 0
    for coro in asyncio.as_completed(tasks):
        fa, fb, winner = await coro
        results.append({"filename_a": fa, "filename_b": fb, "winner": winner})
        completed += 1
        if completed % 200 == 0:
            pd.DataFrame(results).to_csv(results_path, index=False)
            log.info(f"  Progress: {completed}/{len(pairs)} comparisons")

    pd.DataFrame(results).to_csv(results_path, index=False)
    return pd.DataFrame(results)

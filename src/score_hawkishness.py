#!/usr/bin/env python3
"""
Hawkishness scoring using the Apel & Blix Grimaldi (2014) central bank tone lexicon.

Each speech fragment is scored by summing the hawk (+1) / dove (-1) polarity of
matched lemmas and dividing by total lemmas in the fragment (length normalization).

Fragment scores are aggregated to speech level via length-weighted mean, then ranked
within each regime (Bernanke / Yellen / Powell) into equal thirds:
    bottom third  → Dovish
    middle third  → Neutral
    top third     → Hawkish

Yield direction test (excluding evening speeches and Neutral):
    Hawkish speech → 2yr yield should rise (DGS2 close > DGS2_prev)
    Dovish speech  → 2yr yield should fall (DGS2 close < DGS2_prev)

Outputs:
    data/processed/speech_scores_{chair}.csv  — per-speech scores, labels, yield data
    data/processed/yield_results_abg.csv      — accuracy summary per regime + overall
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# ── Apel & Blix Grimaldi (2014) hawk / dove lexicon ──────────────────────────
# +1 = hawkish signal, -1 = dovish signal
# Lemma forms used (lowercase, base form as spaCy would produce)

HAWK_DOVE: dict[str, int] = {
    # ── Hawkish: tightening / strength / above-target ─────────────────────
    "tighten":        +1,
    "tightening":     +1,   # kept as insurance; spaCy usually lemmatizes to tighten
    "hike":           +1,
    "raise":          +1,
    "rate":            0,   # ambiguous alone — skip; but "raise rate" captured separately
    "increase":       +1,
    "overheat":       +1,
    "overheating":    +1,
    "overshoot":      +1,
    "above":          +1,
    "exceed":         +1,
    "exceeds":        +1,
    "firming":        +1,
    "firm":           +1,   # "labor market firm" / "inflation firming"
    "vigilant":       +1,
    "resilient":      +1,
    "elevated":       +1,
    "taper":          +1,
    "tapering":       +1,
    "unwind":         +1,
    "normalize":      +1,
    "normalization":  +1,
    "liftoff":        +1,
    "restrictive":    +1,
    "restriction":    +1,
    "upside":         +1,   # "upside risk to inflation"
    "upward":         +1,
    "robust":         +1,
    "strong":         +1,
    "strengthen":     +1,
    "strength":       +1,
    "tight":          +1,
    "surge":          +1,
    "accelerate":     +1,
    "acceleration":   +1,
    "persistent":     +1,   # "inflation persistent"
    "entrenched":     +1,   # "inflation entrenched"
    "overrun":        +1,
    "inertia":        +1,   # inflation expectations unanchored
    "hawkish":        +1,

    # ── Dovish: easing / weakness / below-target ──────────────────────────
    "ease":           -1,
    "easing":         -1,
    "accommodative":  -1,
    "accommodate":    -1,
    "accommodation":  -1,
    "stimulus":       -1,
    "stimulative":    -1,
    "patient":        -1,
    "patience":       -1,
    "gradual":        -1,
    "gradually":      -1,
    "slow":           -1,
    "sluggish":       -1,
    "weak":           -1,
    "weaken":         -1,
    "weakness":       -1,
    "deteriorate":    -1,
    "deterioration":  -1,
    "downside":       -1,   # "downside risk"
    "downward":       -1,
    "below":          -1,
    "shortfall":      -1,
    "undershoot":     -1,
    "subdued":        -1,
    "muted":          -1,
    "slack":          -1,
    "headwind":       -1,
    "headwinds":      -1,
    "fragile":        -1,
    "uncertainty":    -1,
    "uncertain":      -1,
    "cut":            -1,
    "reduce":         -1,
    "reduction":      -1,
    "low":            -1,   # "inflation low", "growth low"
    "lower":          -1,
    "decelerate":     -1,
    "deceleration":   -1,
    "soft":           -1,
    "soften":         -1,
    "moderate":       -1,
    "moderation":     -1,
    "moderately":     -1,
    "support":        -1,   # "support economic growth" = dovish intent
    "supportive":     -1,
    "dovish":         -1,
    "lax":            -1,
    "laxity":         -1,
}

# Remove the zero-weight placeholder entry (kept for documentation clarity)
HAWK_DOVE = {k: v for k, v in HAWK_DOVE.items() if v != 0}

# Speeches delivered after market close — excluded from yield test
# (yield change for these reflects noise, not the speech)
EVENING_SPEECHES = {
    "20100408_bernanke_Economic_Outlook_and_Fiscal_Challenges.txt",
    "20131119_bernanke_The_Federal_Reserve_Forty_Years_after_the_Reform.txt",
    "20190228_powell_Monetary_Policy_Patience_and_the_Economic_Outlook.txt",
}


def score_fragment(lemma_str: str) -> tuple[float, int]:
    """
    Score a single fragment.
    Returns (normalized_score, n_lemmas).
        normalized_score = sum of polarities / n_lemmas  (0 if no lemmas)
    """
    if not isinstance(lemma_str, str) or not lemma_str.strip():
        return 0.0, 0
    lemmas = lemma_str.split()
    n = len(lemmas)
    if n == 0:
        return 0.0, 0
    polarity_sum = sum(HAWK_DOVE.get(lem, 0) for lem in lemmas)
    return polarity_sum / n, n


def score_speeches(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate fragment scores to speech level via length-weighted mean.
    Returns a DataFrame with one row per speech:
        filename, date, title, raw_score, n_lemmas_total
    """
    scores = sentences_df["lemmas"].apply(score_fragment)
    sentences_df = sentences_df.copy()
    sentences_df["frag_score"] = scores.apply(lambda x: x[0])
    sentences_df["frag_n"]     = scores.apply(lambda x: x[1])

    def weighted_mean(g):
        g = g[g["frag_n"] > 0]
        if g.empty:
            return pd.Series({"raw_score": 0.0, "n_lemmas_total": 0})
        total_n = g["frag_n"].sum()
        w_score = (g["frag_score"] * g["frag_n"]).sum() / total_n
        return pd.Series({"raw_score": w_score, "n_lemmas_total": int(total_n)})

    speech_scores = (
        sentences_df
        .groupby(["filename", "date", "title"], as_index=False)
        .apply(weighted_mean, include_groups=False)
        .reset_index(drop=True)
    )
    return speech_scores


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank speeches within regime and assign Dovish / Neutral / Hawkish
    using equal thirds (33rd and 67th percentile cutoffs on raw_score).
    """
    df = df.copy()
    p33 = df["raw_score"].quantile(1/3)
    p67 = df["raw_score"].quantile(2/3)

    def label(score):
        if score <= p33:
            return "Dovish"
        elif score <= p67:
            return "Neutral"
        else:
            return "Hawkish"

    df["label"] = df["raw_score"].apply(label)
    df["rank"]  = df["raw_score"].rank(method="average").astype(int)
    return df


def run_yield_test(df: pd.DataFrame, macro: pd.DataFrame, chair: str) -> dict:
    """
    Merge in DGS2 / DGS2_prev, exclude evening speeches and Neutral,
    compute yield direction accuracy.
    """
    merged = df.merge(
        macro[["filename", "DGS2", "DGS2_prev"]],
        on="filename",
        how="left"
    )
    merged["yield_change"] = merged["DGS2"] - merged["DGS2_prev"]

    # Exclude evening speeches
    merged = merged[~merged["filename"].isin(EVENING_SPEECHES)]

    # Exclude speeches with missing yield data
    merged = merged.dropna(subset=["DGS2", "DGS2_prev"])

    # Only Hawkish / Dovish for directional test
    directional = merged[merged["label"] != "Neutral"].copy()

    if directional.empty:
        return {"chair": chair, "n": 0, "correct": 0, "accuracy": float("nan")}

    correct = (
        ((directional["label"] == "Hawkish") & (directional["yield_change"] > 0)) |
        ((directional["label"] == "Dovish")  & (directional["yield_change"] < 0))
    ).sum()

    n = len(directional)
    return {
        "chair":    chair,
        "n":        n,
        "correct":  int(correct),
        "accuracy": round(correct / n * 100, 1),
    }


def main():
    macro = pd.read_csv(PROCESSED / "macro_context.csv")

    all_results = []
    all_scores  = []

    for chair in ["bernanke", "yellen", "powell"]:
        print(f"\n── {chair} ──────────────────────────────────────")

        sentences = pd.read_csv(PROCESSED / f"sentences_{chair}.csv")

        # Score + aggregate to speech level
        speech_scores = score_speeches(sentences)

        # Attach labels
        speech_scores = assign_labels(speech_scores)

        # Quick lexicon coverage check
        all_lemmas = " ".join(sentences["lemmas"].dropna()).split()
        found = {k for k in HAWK_DOVE if k in set(all_lemmas)}
        missing = set(HAWK_DOVE) - found
        print(f"  Lexicon coverage: {len(found)}/{len(HAWK_DOVE)} terms found in corpus")
        if missing:
            print(f"  Not found: {', '.join(sorted(missing))}")

        # Print distribution
        counts = speech_scores["label"].value_counts()
        print(f"  Label distribution: {dict(counts)}")

        # Score range
        lo = speech_scores["raw_score"].min()
        hi = speech_scores["raw_score"].max()
        print(f"  Score range: [{lo:.5f}, {hi:.5f}]")

        # Top 3 hawkish / dovish for spot-check
        top_hawk = speech_scores.nlargest(3, "raw_score")[["date", "title", "raw_score"]]
        top_dove = speech_scores.nsmallest(3, "raw_score")[["date", "title", "raw_score"]]
        print("  Top 3 hawkish:")
        for _, r in top_hawk.iterrows():
            print(f"    {r['date']}  {r['title'][:60]}  ({r['raw_score']:+.5f})")
        print("  Top 3 dovish:")
        for _, r in top_dove.iterrows():
            print(f"    {r['date']}  {r['title'][:60]}  ({r['raw_score']:+.5f})")

        # Yield test
        result = run_yield_test(speech_scores, macro, chair)
        all_results.append(result)
        print(f"  Yield test: {result['correct']}/{result['n']} correct → {result['accuracy']}%")

        # Save speech scores
        out_path = PROCESSED / f"speech_scores_{chair}.csv"
        speech_scores.to_csv(out_path, index=False)
        print(f"  Saved → {out_path.name}")

        all_scores.append(speech_scores.assign(chair=chair))

    # Overall accuracy (pool Hawkish/Dovish speeches across all regimes)
    combined = pd.concat(all_scores, ignore_index=True)
    macro_all = macro[["filename", "DGS2", "DGS2_prev"]].copy()
    combined = combined.merge(macro_all, on="filename", how="left")
    combined["yield_change"] = combined["DGS2"] - combined["DGS2_prev"]
    combined = combined[~combined["filename"].isin(EVENING_SPEECHES)]
    combined = combined.dropna(subset=["DGS2", "DGS2_prev"])
    directional_all = combined[combined["label"] != "Neutral"]

    overall_n       = len(directional_all)
    overall_correct = int(
        ((directional_all["label"] == "Hawkish") & (directional_all["yield_change"] > 0) |
         (directional_all["label"] == "Dovish")  & (directional_all["yield_change"] < 0)).sum()
    )
    overall_acc = round(overall_correct / overall_n * 100, 1) if overall_n else float("nan")

    all_results.append({
        "chair":    "ALL",
        "n":        overall_n,
        "correct":  overall_correct,
        "accuracy": overall_acc,
    })

    print(f"\n── Overall ──────────────────────────────────────────")
    print(f"  {overall_correct}/{overall_n} correct → {overall_acc}%")

    # Save summary
    results_df = pd.DataFrame(all_results)
    out_path = PROCESSED / "yield_results_abg.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved yield results → {out_path.name}")


if __name__ == "__main__":
    main()

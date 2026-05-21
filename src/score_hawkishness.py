#!/usr/bin/env python3
"""
Hawkishness scoring using the Apel & Blix Grimaldi (2014) central bank tone lexicon.

Three sub-scores are computed per speech:
    hawk_density  = hawk term hits / total lemmas  (higher = more hawkish)
    dove_density  = dove term hits / total lemmas  (higher = more dovish)
    net_score     = hawk_density - dove_density    (higher = more hawkish)

Each sub-score is standardized (z-score) using the 2008–2020 training window
within each regime. The primary hawkishness measure combines all three:

    hawkishness = (hawk_z - dove_z + net_z) / 3

dove_z is subtracted because high dove density is dovish (opposite direction).
Individual z-scores are retained in the output for robustness checks.

Speeches are ranked within regime into equal thirds:
    bottom third → Dovish  |  middle → Neutral  |  top → Hawkish

Yield direction test (evening speeches and Neutral excluded):
    Hawkish → 2yr yield should rise  |  Dovish → should fall

Outputs:
    data/processed/speech_scores_{chair}.csv  — per-speech scores + yield data
    data/processed/yield_results_abg.csv      — accuracy summary per regime + overall
"""

from pathlib import Path

import pandas as pd
import numpy as np

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

TRAIN_CUTOFF = "2020-12-31"   # z-score mean/std computed from speeches up to this date

# ── Apel & Blix Grimaldi (2014) hawk / dove lexicon ──────────────────────────
# Keys are lemma forms (lowercase base form as spaCy produces).
# +1 = hawkish signal, -1 = dovish signal.

HAWK_DOVE: dict[str, int] = {
    # Hawkish: tightening / strength / above-target
    "tighten":        +1,
    "tightening":     +1,
    "hike":           +1,
    "raise":          +1,
    "increase":       +1,
    "overheat":       +1,
    "overheating":    +1,
    "overshoot":      +1,
    "above":          +1,
    "exceed":         +1,
    "exceeds":        +1,
    "firming":        +1,
    "firm":           +1,
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
    "upside":         +1,
    "upward":         +1,
    "robust":         +1,
    "strong":         +1,
    "strengthen":     +1,
    "strength":       +1,
    "tight":          +1,
    "surge":          +1,
    "accelerate":     +1,
    "acceleration":   +1,
    "persistent":     +1,
    "entrenched":     +1,
    "overrun":        +1,
    "inertia":        +1,
    "hawkish":        +1,
    # Dovish: easing / weakness / below-target
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
    "downside":       -1,
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
    "low":            -1,
    "lower":          -1,
    "decelerate":     -1,
    "deceleration":   -1,
    "soft":           -1,
    "soften":         -1,
    "moderate":       -1,
    "moderation":     -1,
    "moderately":     -1,
    "support":        -1,
    "supportive":     -1,
    "dovish":         -1,
    "lax":            -1,
    "laxity":         -1,
}

HAWK_TERMS = {k for k, v in HAWK_DOVE.items() if v == +1}
DOVE_TERMS = {k for k, v in HAWK_DOVE.items() if v == -1}

# Speeches delivered after market close — excluded from yield test
EVENING_SPEECHES = {
    "20100408_bernanke_Economic_Outlook_and_Fiscal_Challenges.txt",
    "20131119_bernanke_The_Federal_Reserve_Forty_Years_after_the_Reform.txt",
    "20190228_powell_Monetary_Policy_Patience_and_the_Economic_Outlook.txt",
}


# ── Fragment scoring ──────────────────────────────────────────────────────────

def score_fragment(lemma_str: str) -> tuple[int, int, int]:
    """
    Return (hawk_hits, dove_hits, n_lemmas) for one fragment.
    """
    if not isinstance(lemma_str, str) or not lemma_str.strip():
        return 0, 0, 0
    lemmas = lemma_str.split()
    n = len(lemmas)
    hawk = sum(1 for lem in lemmas if lem in HAWK_TERMS)
    dove = sum(1 for lem in lemmas if lem in DOVE_TERMS)
    return hawk, dove, n


# ── Speech-level aggregation ──────────────────────────────────────────────────

def score_speeches(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate fragment counts to speech level.
    Densities are computed over total lemmas across all fragments in the speech,
    which is equivalent to a length-weighted average of fragment-level densities.

    Returns one row per speech with columns:
        filename, date, title, hawk_density, dove_density, net_score, n_lemmas_total
    """
    parsed = sentences_df["lemmas"].apply(score_fragment)
    tmp = sentences_df[["filename", "date", "title"]].copy()
    tmp["hawk_hits"] = parsed.apply(lambda x: x[0])
    tmp["dove_hits"] = parsed.apply(lambda x: x[1])
    tmp["n_lemmas"]  = parsed.apply(lambda x: x[2])

    agg = tmp.groupby(["filename", "date", "title"], as_index=False).agg(
        hawk_hits     = ("hawk_hits", "sum"),
        dove_hits     = ("dove_hits", "sum"),
        n_lemmas_total= ("n_lemmas",  "sum"),
    )

    # Guard against zero-lemma speeches (shouldn't happen, but be safe)
    denom = agg["n_lemmas_total"].replace(0, np.nan)
    agg["hawk_density"] = agg["hawk_hits"] / denom
    agg["dove_density"] = agg["dove_hits"] / denom
    agg["net_score"]    = agg["hawk_density"] - agg["dove_density"]

    return agg.drop(columns=["hawk_hits", "dove_hits"])


# ── Z-score standardization ───────────────────────────────────────────────────

def add_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize hawk_density, dove_density, and net_score using the
    2008–2020 training window (mean and std computed on training speeches only,
    then applied to all speeches to avoid lookahead bias).

    Adds columns: hawk_z, dove_z, net_z, hawkishness
        hawkishness = (hawk_z - dove_z + net_z) / 3
        (dove_z is subtracted: high dove density is dovish, opposite direction)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    train = df[df["date"] <= TRAIN_CUTOFF]

    for col, z_col in [("hawk_density", "hawk_z"),
                       ("dove_density", "dove_z"),
                       ("net_score",    "net_z")]:
        mu  = train[col].mean()
        sig = train[col].std(ddof=1)
        if sig == 0 or pd.isna(sig):
            df[z_col] = 0.0
        else:
            df[z_col] = (df[col] - mu) / sig

    df["hawkishness"] = (df["hawk_z"] - df["dove_z"] + df["net_z"]) / 3
    return df


# ── Labelling ─────────────────────────────────────────────────────────────────

def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank speeches by primary hawkishness measure within regime.
    Equal-thirds bucketing: bottom → Dovish, middle → Neutral, top → Hawkish.
    """
    df = df.copy()
    p33 = df["hawkishness"].quantile(1 / 3)
    p67 = df["hawkishness"].quantile(2 / 3)

    def label(score):
        if score <= p33:
            return "Dovish"
        elif score <= p67:
            return "Neutral"
        else:
            return "Hawkish"

    df["label"] = df["hawkishness"].apply(label)
    df["rank"]  = df["hawkishness"].rank(method="average").astype(int)
    return df


# ── Yield direction test ──────────────────────────────────────────────────────

def run_yield_test(df: pd.DataFrame, macro: pd.DataFrame, chair: str) -> dict:
    merged = df.merge(macro[["filename", "DGS2", "DGS2_prev"]], on="filename", how="left")
    merged["yield_change"] = merged["DGS2"] - merged["DGS2_prev"]
    merged = merged[~merged["filename"].isin(EVENING_SPEECHES)]
    merged = merged.dropna(subset=["DGS2", "DGS2_prev"])

    directional = merged[merged["label"] != "Neutral"].copy()
    if directional.empty:
        return {"chair": chair, "n": 0, "correct": 0, "accuracy": float("nan")}

    correct = int((
        ((directional["label"] == "Hawkish") & (directional["yield_change"] > 0)) |
        ((directional["label"] == "Dovish")  & (directional["yield_change"] < 0))
    ).sum())
    n = len(directional)
    return {"chair": chair, "n": n, "correct": correct, "accuracy": round(correct / n * 100, 1)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    macro = pd.read_csv(PROCESSED / "macro_context.csv")

    all_results = []
    all_scored  = []

    for chair in ["bernanke", "yellen", "powell"]:
        print(f"\n── {chair} ──────────────────────────────────────")

        sentences = pd.read_csv(PROCESSED / f"sentences_{chair}.csv")

        # Score fragments → aggregate to speech level
        df = score_speeches(sentences)

        # Z-score standardize (training window 2008–2020)
        df = add_z_scores(df)
        train_n = (pd.to_datetime(df["date"]) <= TRAIN_CUTOFF).sum()
        print(f"  Training window speeches: {train_n}/{len(df)}")

        # Lexicon coverage check
        all_lemmas = set(" ".join(sentences["lemmas"].dropna()).split())
        found   = {k for k in HAWK_DOVE if k in all_lemmas}
        missing = set(HAWK_DOVE) - found
        print(f"  Lexicon coverage: {len(found)}/{len(HAWK_DOVE)} terms found")
        if missing:
            print(f"  Not found: {', '.join(sorted(missing))}")

        # Assign labels using primary hawkishness measure
        df = assign_labels(df)

        print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
        print(f"  hawkishness range: [{df['hawkishness'].min():.3f}, {df['hawkishness'].max():.3f}]")

        # Spot-check: top 3 hawkish / dovish
        for direction, method in [("hawkish", "nlargest"), ("dovish", "nsmallest")]:
            top = getattr(df, method)(3, "hawkishness")[["date", "title", "hawkishness", "hawk_z", "dove_z", "net_z"]]
            print(f"  Top 3 {direction}:")
            for _, r in top.iterrows():
                print(f"    {str(r['date'])[:10]}  {r['title'][:55]:<55}  "
                      f"H={r['hawkishness']:+.2f}  (hk={r['hawk_z']:+.2f} dv={r['dove_z']:+.2f} net={r['net_z']:+.2f})")

        # Yield test
        result = run_yield_test(df, macro, chair)
        all_results.append(result)
        print(f"  Yield test: {result['correct']}/{result['n']} → {result['accuracy']}%")

        # Save
        out_path = PROCESSED / f"speech_scores_{chair}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path.name}")

        all_scored.append(df.assign(chair=chair))

    # Overall yield test across all regimes
    combined = pd.concat(all_scored, ignore_index=True)
    combined = combined.merge(macro[["filename", "DGS2", "DGS2_prev"]], on="filename", how="left")
    combined["yield_change"] = combined["DGS2"] - combined["DGS2_prev"]
    combined = combined[~combined["filename"].isin(EVENING_SPEECHES)]
    combined = combined.dropna(subset=["DGS2", "DGS2_prev"])
    directional_all = combined[combined["label"] != "Neutral"]

    overall_n       = len(directional_all)
    overall_correct = int((
        ((directional_all["label"] == "Hawkish") & (directional_all["yield_change"] > 0)) |
        ((directional_all["label"] == "Dovish")  & (directional_all["yield_change"] < 0))
    ).sum())
    overall_acc = round(overall_correct / overall_n * 100, 1) if overall_n else float("nan")
    all_results.append({"chair": "ALL", "n": overall_n, "correct": overall_correct, "accuracy": overall_acc})

    print(f"\n── Overall ──────────────────────────────────────────")
    print(f"  {overall_correct}/{overall_n} correct → {overall_acc}%")

    pd.DataFrame(all_results).to_csv(PROCESSED / "yield_results_abg.csv", index=False)
    print(f"\nSaved yield results → yield_results_abg.csv")


if __name__ == "__main__":
    main()

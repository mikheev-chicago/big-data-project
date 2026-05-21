#!/usr/bin/env python3
"""
Step 4: Lemmatization (TD3) and IDF Damping (TD4)

TD3 — Lemmatization:
    Lowercase and lemmatize every sentence fragment with spaCy.
    Remove stopwords, punctuation, spaces, and pure-digit tokens.
    Minimum token length: 2 characters.

TD4 — IDF Damping:
    Document frequency (DF) is computed at the speech level per regime —
    i.e. what fraction of *speeches* contain a given lemma. Using speeches
    (not fragments) as documents is intentional: words like "inflation"
    appear in 30% of fragments but 95% of speeches, so speech-level DF
    correctly identifies them as boilerplate.

    Multipliers:
        DF > 0.90  →  0.0   (pure boilerplate — zero out entirely)
        DF > 0.50  →  0.3   (common but not universal — downweight)
        DF ≤ 0.50  →  1.0   (differentiating — keep at full weight)

    The 0.0/0.3/1.0 multipliers are saved to idf_weights_{chair}.csv for
    use at scoring time. Applied here to produce:
        lemmas        — all content lemmas (stopwords/punct removed)
        damped_lemmas — lemmas with 0.0-weight (boilerplate) terms removed

Output:
    data/processed/sentences_{chair}.csv   — updated with lemmas columns
    data/processed/idf_weights_{chair}.csv — lemma → doc_freq → multiplier

Usage:
    python3 src/lemmatize_and_damp.py
"""

import sys
from pathlib import Path

import pandas as pd

try:
    import spacy
except ImportError:
    sys.exit("pip install spacy && python3 -m spacy download en_core_web_sm")

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

CHAIRS = ["bernanke", "yellen", "powell"]

# IDF damping thresholds and their multipliers
IDF_TIERS = [
    (0.90, 0.0),   # appears in >90% of speeches → boilerplate, zero out
    (0.50, 0.3),   # appears in >50% of speeches → common, downweight
    (0.00, 1.0),   # everything else → differentiating, keep
]


def get_multiplier(doc_freq: float) -> float:
    for threshold, multiplier in IDF_TIERS:
        if doc_freq > threshold:
            return multiplier
    return 1.0


def lemmatize_batch(texts: list[str], nlp) -> list[str]:
    """
    Lemmatize a list of text fragments in batch.
    Returns a list of space-separated lemma strings (one per input text).
    Filters: stopwords, punctuation, spaces, digits, length < 2.
    """
    results = []
    for doc in nlp.pipe(texts, batch_size=256):
        tokens = [
            t.lemma_.lower()
            for t in doc
            if not t.is_stop
            and not t.is_punct
            and not t.is_space
            and not t.like_num
            and len(t.lemma_) >= 2
        ]
        results.append(" ".join(tokens))
    return results


def compute_speech_level_df(df: pd.DataFrame) -> pd.Series:
    """
    Compute per-lemma document frequency at speech level.
    DF = (number of speeches containing the lemma) / (total speeches).
    Returns a Series: lemma → doc_freq.
    """
    speeches = df.groupby("filename")["lemmas"].apply(
        lambda frags: " ".join(frags.dropna())
    )
    n_speeches = len(speeches)

    # Count how many speeches each lemma appears in
    lemma_counts: dict[str, int] = {}
    for speech_text in speeches:
        unique_lemmas = set(speech_text.split())
        for lemma in unique_lemmas:
            lemma_counts[lemma] = lemma_counts.get(lemma, 0) + 1

    doc_freq = pd.Series(lemma_counts, name="doc_freq") / n_speeches
    return doc_freq


def main():
    print("Loading spaCy model (with tagger for lemmatization)…")
    # Parser disabled (not needed); tagger + attribute_ruler + lemmatizer kept
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    for chair in CHAIRS:
        print(f"\n── {chair} ──────────────────────────────────────")

        df = pd.read_csv(PROCESSED / f"sentences_{chair}.csv")
        n_fragments = len(df)
        n_speeches  = df["filename"].nunique()

        # ── TD3: Lemmatize all fragments ──────────────────────────────────────
        print(f"  Lemmatizing {n_fragments} fragments…")
        df["lemmas"] = lemmatize_batch(df["text"].tolist(), nlp)

        # ── TD4: Compute speech-level document frequency ──────────────────────
        print(f"  Computing IDF over {n_speeches} speeches…")
        doc_freq = compute_speech_level_df(df)

        # Assign multipliers
        weights_df = doc_freq.reset_index()
        weights_df.columns = ["lemma", "doc_freq"]
        weights_df["multiplier"] = weights_df["doc_freq"].apply(get_multiplier)

        # Save weights table
        weights_path = PROCESSED / f"idf_weights_{chair}.csv"
        weights_df.sort_values("doc_freq", ascending=False).to_csv(
            weights_path, index=False
        )

        # Build lemma → multiplier lookup
        mult_lookup = dict(zip(weights_df["lemma"], weights_df["multiplier"]))

        # ── Apply damping: produce damped_lemmas ──────────────────────────────
        # damped_lemmas = lemmas with 0.0-weight (boilerplate) terms removed
        # 0.3-weight terms are kept here; multiplier is applied at scoring time
        def damp(lemma_str: str) -> str:
            if not isinstance(lemma_str, str):
                return ""
            return " ".join(
                lem for lem in lemma_str.split()
                if mult_lookup.get(lem, 1.0) > 0.0
            )

        df["damped_lemmas"] = df["lemmas"].apply(damp)

        # ── Save updated sentences CSV ────────────────────────────────────────
        out_path = PROCESSED / f"sentences_{chair}.csv"
        df.to_csv(out_path, index=False)

        # ── Summary ───────────────────────────────────────────────────────────
        zeroed    = (weights_df["multiplier"] == 0.0).sum()
        dampedw   = (weights_df["multiplier"] == 0.3).sum()
        kept      = (weights_df["multiplier"] == 1.0).sum()
        vocab_size = len(weights_df)

        print(f"  Vocab: {vocab_size} unique lemmas")
        print(f"    0.0 (zeroed, DF >90%): {zeroed:>5}  e.g. {_sample(weights_df, 0.0, 5)}")
        print(f"    0.3 (damped, DF >50%): {dampedw:>5}  e.g. {_sample(weights_df, 0.3, 5)}")
        print(f"    1.0 (kept,   DF ≤50%): {kept:>5}  e.g. {_sample(weights_df, 1.0, 5)}")
        print(f"  Saved → {out_path.name}  |  {weights_path.name}")


def _sample(df: pd.DataFrame, mult: float, n: int) -> str:
    """Return n example lemmas at a given multiplier level."""
    subset = df[df["multiplier"] == mult]["lemma"]
    # Sort by doc_freq descending for 0.0/0.3, ascending for 1.0 (most rare = most interesting)
    if mult == 1.0:
        subset = df[df["multiplier"] == mult].sort_values("doc_freq")["lemma"]
    else:
        subset = df[df["multiplier"] == mult].sort_values("doc_freq", ascending=False)["lemma"]
    sample = subset.head(n).tolist()
    return ", ".join(sample)


if __name__ == "__main__":
    main()

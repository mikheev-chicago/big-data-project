#!/usr/bin/env python3
"""
Unsupervised analysis: TF-IDF → PCA (2D) + K-means clustering of Fed speeches.

Uses raw sentence text from sentences_{chair}.csv so it runs without any API dependency.
Prefer filtered_sentences_{chair}.csv when available (monetary-policy sentences only).

Outputs:
    data/processed/unsupervised_topics.csv
        filename, chair_key, date, title, cluster, pc1, pc2

Usage:
    python src/unsupervised.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

N_CLUSTERS  = 5
N_TOP_TERMS = 12

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_speech_docs() -> pd.DataFrame:
    """
    One row per speech: concatenate all sentence text for that speech.
    Prefers filtered_sentences_{chair}.csv; falls back to sentences_{chair}.csv.
    """
    frames = []
    for chair in ["bernanke", "yellen", "powell"]:
        filtered_path = PROCESSED / f"filtered_sentences_{chair}.csv"
        raw_path      = PROCESSED / f"sentences_{chair}.csv"

        if filtered_path.exists():
            df = pd.read_csv(filtered_path)
            log.info(f"  {chair}: using filtered sentences ({len(df)} rows)")
        elif raw_path.exists():
            df = pd.read_csv(raw_path)
            log.info(f"  {chair}: using raw sentences ({len(df)} rows) — no filtered file yet")
        else:
            log.warning(f"  {chair}: no sentence file found, skipping")
            continue

        frames.append(df[["filename", "chair_key", "date", "title", "text"]])

    if not frames:
        sys.exit("Error: no sentence files found in data/processed/")

    all_sents = pd.concat(frames, ignore_index=True)

    docs = (
        all_sents
        .groupby("filename", as_index=False)
        .agg(
            chair_key=("chair_key", "first"),
            date     =("date",      "first"),
            title    =("title",     "first"),
            doc_text =("text",      lambda x: " ".join(x.dropna())),
        )
    )
    return docs


# ── TF-IDF → PCA + K-means ────────────────────────────────────────────────────

def fit_tfidf(docs: pd.DataFrame):
    vectorizer = TfidfVectorizer(
        max_features=500,
        min_df=3,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(docs["doc_text"])
    log.info(f"  TF-IDF: {X.shape[0]} speeches × {X.shape[1]} terms")
    return X, vectorizer.get_feature_names_out()


def fit_pca(X) -> tuple[PCA, np.ndarray]:
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())
    log.info(
        f"  PCA variance explained: "
        f"PC1={pca.explained_variance_ratio_[0]:.1%}, "
        f"PC2={pca.explained_variance_ratio_[1]:.1%}"
    )
    return pca, coords


def fit_kmeans(X, n_clusters: int) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X.toarray())
    return km, labels


def log_top_terms(km: KMeans, feature_names: np.ndarray, labels: np.ndarray) -> None:
    order = km.cluster_centers_.argsort()[:, ::-1]
    log.info("\n  Top terms per cluster:")
    for i in range(km.n_clusters):
        top   = [feature_names[j] for j in order[i, :N_TOP_TERMS]]
        count = (labels == i).sum()
        log.info(f"    Cluster {i} ({count:3d} speeches): {', '.join(top)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading speech documents...")
    docs = load_speech_docs()
    log.info(f"  {len(docs)} speeches total")

    log.info("Fitting TF-IDF...")
    X, feature_names = fit_tfidf(docs)

    log.info("Fitting PCA (2 components)...")
    _, coords = fit_pca(X)

    log.info(f"Fitting K-means (k={N_CLUSTERS})...")
    km, labels = fit_kmeans(X, N_CLUSTERS)
    log_top_terms(km, feature_names, labels)

    docs = docs.copy()
    docs["cluster"] = labels
    docs["pc1"]     = coords[:, 0]
    docs["pc2"]     = coords[:, 1]

    out_cols = ["filename", "chair_key", "date", "title", "cluster", "pc1", "pc2"]
    out_path = PROCESSED / "unsupervised_topics.csv"
    docs[out_cols].to_csv(out_path, index=False)
    log.info(f"\nSaved: {out_path.name}")

    print("\nCluster × Chair distribution:")
    print(
        docs.groupby(["cluster", "chair_key"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )


if __name__ == "__main__":
    main()

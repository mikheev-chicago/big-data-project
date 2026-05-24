#!/usr/bin/env python3
"""
Powell-regime robustness analysis: stratified OOS across 5 macro regimes.

Unlike a simple temporal split, this holds out 1 yield-up + 1 yield-down
speech from each of Powell's 5 distinct macro regimes (fallback to 2 same-
direction where a regime has only one class). Training is on all remaining
speeches. This tests whether the hawkishness signal generalizes across
different economic environments rather than just one period.

Regime exception: the ZLB (COVID + recovery) has 0 yield-up speeches —
rates pinned at 0, so the 2-year yield only fell or stayed flat. Both
OOS picks from that regime are yield-down.

Primary question: does hawkishness_phase2 predict same-day 2yr yield
direction (up vs. down) across all 5 macro regimes?

Outputs:
    data/processed/powell_regime_splits.csv
    data/processed/powell_clusters.csv
    data/processed/powell_regression_results.csv
    data/processed/powell_classification_results.csv
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

N_CLUSTERS = 5
OOS_SEED   = 42
FEATURES   = ["hawkishness_phase2", "DFF", "PCE_YOY", "UNRATE", "GDP_GROWTH"]

# fmt: (key, start, end, human label)
REGIMES = [
    ("1_normalization", pd.Timestamp("2018-01-01"), pd.Timestamp("2019-07-31"),
     "Rate normalization (hiking)"),
    ("2_dovish_pivot",  pd.Timestamp("2019-08-01"), pd.Timestamp("2020-02-29"),
     "Dovish pivot (insurance cuts)"),
    ("3_ZLB",          pd.Timestamp("2020-03-01"), pd.Timestamp("2022-02-28"),
     "Zero lower bound (COVID + recovery)"),
    ("4_hiking",       pd.Timestamp("2022-03-01"), pd.Timestamp("2024-08-31"),
     "Inflation surge + hiking"),
    ("5_cutting",      pd.Timestamp("2024-09-01"), pd.Timestamp("2026-12-31"),
     "Cutting cycle + tariff uncertainty"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Data ───────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    p2    = pd.read_csv(PROCESSED / "speech_scores_phase2_powell.csv")
    macro = pd.read_csv(PROCESSED / "macro_context.csv")
    exp1  = pd.read_csv(PROCESSED / "speech_scores_powell.csv")[
        ["filename", "hawkishness"]
    ].rename(columns={"hawkishness": "hawkishness_exp1"})

    df = (
        p2
        .merge(macro[["filename", "DGS2", "DGS2_prev", "DFF", "PCE_YOY", "UNRATE", "GDP_GROWTH"]],
               on="filename", how="left")
        .merge(exp1, on="filename", how="left")
    )

    df["yield_change"] = df["DGS2"] - df["DGS2_prev"]
    df["date"]         = pd.to_datetime(df["date"])
    df = df.dropna(subset=["yield_change", "hawkishness_phase2"])

    def _regime(d):
        for key, start, end, _ in REGIMES:
            if start <= d <= end:
                return key
        return "unknown"

    df["regime"] = df["date"].apply(_regime)
    return df.sort_values("date").reset_index(drop=True)


# ── Stratified OOS selection ───────────────────────────────────────────────────

def select_oos(cdf: pd.DataFrame):
    """
    For each regime, hold out 1 yield-up + 1 yield-down speech (if both exist).
    If a regime has only one yield direction (e.g. ZLB), fall back to 2 same-
    direction speeches.

    Returns:
        test_indices  – list of integer index labels (for .loc)
        split_df      – DataFrame logging every speech's train/test assignment
    """
    test_indices = []
    rows = []

    for key, start, end, label in REGIMES:
        mask      = (cdf["date"] >= start) & (cdf["date"] <= end)
        regime_df = cdf[mask]
        up_df     = regime_df[regime_df["yield_up"] == 1]
        down_df   = regime_df[regime_df["yield_up"] == 0]

        if len(up_df) >= 1 and len(down_df) >= 1:
            picked = pd.concat([
                up_df.sample(1, random_state=OOS_SEED),
                down_df.sample(1, random_state=OOS_SEED),
            ])
            note = "1 up + 1 down"
        else:
            n     = min(2, len(regime_df))
            picked = regime_df.sample(n, random_state=OOS_SEED)
            n_up   = int((picked["yield_up"] == 1).sum())
            n_dn   = int((picked["yield_up"] == 0).sum())
            note   = f"fallback: {n_up} up + {n_dn} down (single class in regime)"

        test_indices.extend(picked.index.tolist())

        for _, row in picked.iterrows():
            rows.append({
                "split": "test", "regime": key, "regime_label": label,
                "note": note, "date": row["date"], "title": row["title"],
                "yield_change": round(float(row["yield_change"]), 4),
                "yield_up": int(row["yield_up"]),
                "hawkishness_phase2": round(float(row["hawkishness_phase2"]), 2),
            })

    for idx, row in cdf[~cdf.index.isin(test_indices)].iterrows():
        rows.append({
            "split": "train", "regime": row["regime"],
            "regime_label": next((r[3] for r in REGIMES if r[0] == row["regime"]), ""),
            "note": "", "date": row["date"], "title": row["title"],
            "yield_change": round(float(row["yield_change"]), 4),
            "yield_up": int(row["yield_up"]),
            "hawkishness_phase2": round(float(row["hawkishness_phase2"]), 2),
        })

    return test_indices, pd.DataFrame(rows).sort_values(["regime", "date"])


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_auc(y_true, y_prob) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return round(roc_auc_score(y_true, y_prob), 4)


def eval_clf(name, y_tr, yh_tr, p_tr, y_te, yh_te, p_te) -> dict:
    return {
        "method":       name,
        "is_accuracy":  round(accuracy_score(y_tr, yh_tr), 4),
        "oos_accuracy": round(accuracy_score(y_te, yh_te), 4),
        "is_f1":        round(f1_score(y_tr, yh_tr, zero_division=0), 4),
        "oos_f1":       round(f1_score(y_te, yh_te, zero_division=0), 4),
        "is_auc":       safe_auc(y_tr, p_tr),
        "oos_auc":      safe_auc(y_te, p_te),
        "n_train":      int(len(y_tr)),
        "n_test":       int(len(y_te)),
    }


def eval_reg(name, y_tr, yh_tr, y_te, yh_te) -> dict:
    ss_res = ((y_te - yh_te) ** 2).sum()
    ss_tot = ((y_te - y_tr.mean()) ** 2).sum()
    oos_ct = round(float(1 - ss_res / ss_tot), 4) if ss_tot > 0 else float("nan")
    return {
        "method":    name,
        "is_r2":     round(float(r2_score(y_tr, yh_tr)), 4),
        "oos_r2":    round(float(r2_score(y_te, yh_te)), 4),
        "oos_r2_ct": oos_ct,
        "is_rmse":   round(float(np.sqrt(mean_squared_error(y_tr, yh_tr))), 4),
        "oos_rmse":  round(float(np.sqrt(mean_squared_error(y_te, yh_te))), 4),
    }


# ── Unsupervised ───────────────────────────────────────────────────────────────

def run_unsupervised(df: pd.DataFrame) -> pd.DataFrame:
    path = PROCESSED / "filtered_sentences_powell.csv"
    if not path.exists():
        path = PROCESSED / "sentences_powell.csv"

    sents = pd.read_csv(path)
    sents = sents[sents["filename"].isin(df["filename"])]

    docs = (
        sents.groupby("filename", as_index=False)
        .agg(doc_text=("text", lambda x: " ".join(x.dropna())))
        .merge(df[["filename", "date", "title", "hawkishness_phase2", "regime"]],
               on="filename", how="left")
    )

    vec = TfidfVectorizer(
        max_features=500, min_df=2, stop_words="english",
        ngram_range=(1, 2), sublinear_tf=True,
    )
    X     = vec.fit_transform(docs["doc_text"]).toarray()
    names = vec.get_feature_names_out()

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    km     = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    docs["cluster"] = labels
    docs["pc1"]     = coords[:, 0]
    docs["pc2"]     = coords[:, 1]

    log.info(f"  PCA variance: PC1={pca.explained_variance_ratio_[0]:.1%}  "
             f"PC2={pca.explained_variance_ratio_[1]:.1%}")

    order = km.cluster_centers_.argsort()[:, ::-1]
    for i in range(N_CLUSTERS):
        top   = [names[j] for j in order[i, :8]]
        count = int((labels == i).sum())
        log.info(f"  Cluster {i} ({count} speeches): {', '.join(top)}")

    return docs[["filename", "cluster", "pc1", "pc2"]]


# ── Regression ─────────────────────────────────────────────────────────────────

def run_regression(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    X_tr = train_df[FEATURES].values
    X_te = test_df[FEATURES].values
    y_tr = train_df["yield_change"].values
    y_te = test_df["yield_change"].values

    sc     = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    results = []

    ols = smf.ols(
        "yield_change ~ hawkishness_phase2 + DFF + PCE_YOY + UNRATE + GDP_GROWTH",
        data=train_df,
    ).fit(cov_type="HC3")
    results.append(eval_reg("OLS", y_tr, ols.fittedvalues.values, y_te,
                            ols.predict(test_df).values))

    lasso = LassoCV(cv=5, max_iter=10_000, random_state=42).fit(X_tr_s, y_tr)
    log.info(f"    LASSO α={lasso.alpha_:.4f}")
    results.append(eval_reg("LASSO", y_tr, lasso.predict(X_tr_s), y_te,
                            lasso.predict(X_te_s)))

    ridge = RidgeCV(cv=5).fit(X_tr_s, y_tr)
    results.append(eval_reg("Ridge", y_tr, ridge.predict(X_tr_s), y_te,
                            ridge.predict(X_te_s)))

    return pd.DataFrame(results)


# ── Classification ─────────────────────────────────────────────────────────────

def run_classification(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    X_tr = train_df[FEATURES].values
    X_te = test_df[FEATURES].values
    y_tr = train_df["yield_up"].values
    y_te = test_df["yield_up"].values

    sc     = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    hi         = FEATURES.index("hawkishness_phase2")
    macro_cols = [i for i, f in enumerate(FEATURES) if f != "hawkishness_phase2"]

    def fit_lrcv(name, Xtr, Xte):
        m = LogisticRegressionCV(cv=5, max_iter=2000, random_state=42)
        m.fit(Xtr, y_tr)
        return eval_clf(name,
                        y_tr, m.predict(Xtr),  m.predict_proba(Xtr)[:, 1],
                        y_te, m.predict(Xte),  m.predict_proba(Xte)[:, 1])

    results = []

    results.append(fit_lrcv(
        "Logistic — hawkishness_phase2 only",
        X_tr_s[:, [hi]], X_te_s[:, [hi]],
    ))
    results.append(fit_lrcv(
        "Logistic — macro only",
        X_tr_s[:, macro_cols], X_te_s[:, macro_cols],
    ))
    results.append(fit_lrcv(
        "Logistic — full (phase2 + macro)",
        X_tr_s, X_te_s,
    ))

    if "hawkishness_exp1" in train_df.columns:
        sc_e1    = StandardScaler()
        e1_tr    = sc_e1.fit_transform(train_df[["hawkishness_exp1"]].values)
        e1_te    = sc_e1.transform(test_df[["hawkishness_exp1"]].values)
        results.append(fit_lrcv("Logistic — Exp 1 hawkishness only", e1_tr, e1_te))

    rf = RandomForestClassifier(n_estimators=500, max_depth=3, random_state=42)
    rf.fit(X_tr, y_tr)
    results.append(eval_clf(
        "Random Forest",
        y_tr, rf.predict(X_tr),  rf.predict_proba(X_tr)[:, 1],
        y_te, rf.predict(X_te),  rf.predict_proba(X_te)[:, 1],
    ))

    majority  = int(np.bincount(y_tr).argmax())
    base_prob = float(y_tr.mean())
    results.append(eval_clf(
        "Baseline (majority class)",
        y_tr, np.full_like(y_tr, majority), np.full(len(y_tr), base_prob),
        y_te, np.full_like(y_te, majority), np.full(len(y_te), base_prob),
    ))

    return pd.DataFrame(results)


# ── LOOCV supplement ────────────────────────────────────────────────────────────

def run_loocv(cdf: pd.DataFrame) -> dict:
    X = cdf[["hawkishness_phase2"]].values
    y = cdf["yield_up"].values

    loo   = LeaveOneOut()
    probs = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr       = y[train_idx]
        sc         = StandardScaler()
        X_tr_s     = sc.fit_transform(X_tr)
        X_te_s     = sc.transform(X_te)
        m = LogisticRegressionCV(cv=3, max_iter=2000, random_state=42)
        m.fit(X_tr_s, y_tr)
        probs[test_idx] = m.predict_proba(X_te_s)[0, 1]

    preds = (probs >= 0.5).astype(int)
    return {
        "loocv_accuracy": round(float(accuracy_score(y, preds)), 4),
        "loocv_f1":       round(float(f1_score(y, preds, zero_division=0)), 4),
        "loocv_auc":      round(float(roc_auc_score(y, probs)), 4),
        "n":              len(y),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading Powell data...")
    df  = load_data()
    cdf = df[df["yield_change"] != 0].copy()
    cdf["yield_up"] = (cdf["yield_change"] > 0).astype(int)
    cdf = cdf.reset_index(drop=True)

    log.info(f"  {len(df)} total speeches  |  {len(cdf)} with non-zero yield change")
    log.info(f"  Yield Δ: mean={df['yield_change'].mean():.4f}  "
             f"std={df['yield_change'].std():.4f}  "
             f"range=[{df['yield_change'].min():.3f}, {df['yield_change'].max():.3f}]")

    # ── OOS selection ─────────────────────────────────────────────────────────
    log.info("\n── Stratified OOS selection ─────────────────────────────────────────────────")
    test_idx, split_df = select_oos(cdf)
    train_cdf = cdf[~cdf.index.isin(test_idx)].reset_index(drop=True)
    test_cdf  = cdf[cdf.index.isin(test_idx)].reset_index(drop=True)

    log.info(f"  Train: {len(train_cdf)} speeches  |  Test (OOS): {len(test_cdf)} speeches")
    log.info(f"  OOS class balance: {test_cdf['yield_up'].mean():.0%} up "
             f"({test_cdf['yield_up'].sum()}/{len(test_cdf)})")

    for _, row in split_df[split_df["split"] == "test"].iterrows():
        direction = "↑" if row["yield_up"] else "↓"
        log.info(f"    [{row['regime']}] {direction} {str(row['date'])[:10]}  "
                 f"hawk={row['hawkishness_phase2']:.1f}  Δy={row['yield_change']:+.3f}  "
                 f"— {row['title'][:55]}")

    split_df.to_csv(PROCESSED / "powell_regime_splits.csv", index=False)

    # ── Signal strength ───────────────────────────────────────────────────────
    log.info("\n── Signal Strength (all 39 non-zero speeches) ──────────────────────────────")
    up_scores   = cdf[cdf["yield_up"] == 1]["hawkishness_phase2"]
    down_scores = cdf[cdf["yield_up"] == 0]["hawkishness_phase2"]
    t_stat, p_val = stats.ttest_ind(up_scores, down_scores, equal_var=False)
    pb_corr = cdf["hawkishness_phase2"].corr(cdf["yield_up"].astype(float))

    log.info(f"  Mean hawkishness — yield UP:   {up_scores.mean():.1f}  (n={len(up_scores)})")
    log.info(f"  Mean hawkishness — yield DOWN: {down_scores.mean():.1f}  (n={len(down_scores)})")
    log.info(f"  Difference: {up_scores.mean() - down_scores.mean():+.1f} pts  "
             f"t={t_stat:.2f}  p={p_val:.3f}  r={pb_corr:.3f}")

    # Signal by regime
    log.info("  Per-regime mean hawkishness:")
    for key, *_ in REGIMES:
        sub = cdf[cdf["regime"] == key]
        if len(sub) == 0:
            continue
        u = sub[sub["yield_up"] == 1]["hawkishness_phase2"]
        d = sub[sub["yield_up"] == 0]["hawkishness_phase2"]
        log.info(f"    {key}: UP={u.mean():.1f}(n={len(u)})  DOWN={d.mean():.1f}(n={len(d)})")

    # ── Unsupervised ──────────────────────────────────────────────────────────
    log.info(f"\n── Unsupervised: TF-IDF + K-means (k={N_CLUSTERS}) ─────────────────────────")
    cluster_df = run_unsupervised(df)
    cluster_df.to_csv(PROCESSED / "powell_clusters.csv", index=False)

    # ── Regression ────────────────────────────────────────────────────────────
    log.info("\n── Regression ───────────────────────────────────────────────────────────────")
    # Use full df (including zero-yield) for regression train/test
    reg_train = df[df.index.isin(train_cdf.index.map(
        lambda i: df[df["filename"] == train_cdf.loc[i, "filename"]].index[0]
        if i < len(train_cdf) else i
    ))] if False else train_cdf  # simplify: use same nonzero split for regression too
    reg = run_regression(train_cdf, test_cdf)
    for _, r in reg.iterrows():
        log.info(f"  {r['method']:<8}  IS R²={r['is_r2']:+.4f}  "
                 f"OOS R²={r['oos_r2']:+.4f}  (C&T OOS R²={r['oos_r2_ct']:+.4f})")
    reg.to_csv(PROCESSED / "powell_regression_results.csv", index=False)

    # ── Classification ────────────────────────────────────────────────────────
    log.info("\n── Classification: predict yield direction ──────────────────────────────────")
    log.info(f"  Train: {train_cdf['yield_up'].mean():.0%} up ({train_cdf['yield_up'].sum()}/{len(train_cdf)})")
    clf = run_classification(train_cdf, test_cdf)
    for _, r in clf.iterrows():
        log.info(
            f"  {r['method']:<42}  "
            f"IS acc={r['is_accuracy']:.3f}  OOS acc={r['oos_accuracy']:.3f}  "
            f"IS AUC={r['is_auc']}  OOS AUC={r['oos_auc']}"
        )
    clf.to_csv(PROCESSED / "powell_classification_results.csv", index=False)

    # ── LOOCV ─────────────────────────────────────────────────────────────────
    log.info(f"\n── LOOCV (hawkishness_phase2 only, all {len(cdf)} non-zero speeches) ────────")
    loocv = run_loocv(cdf)
    log.info(f"  Accuracy={loocv['loocv_accuracy']:.3f}  "
             f"F1={loocv['loocv_f1']:.3f}  "
             f"AUC={loocv['loocv_auc']:.3f}  (n={loocv['n']})")

    # ── Summary print ─────────────────────────────────────────────────────────
    sep = "─" * 78

    print(f"\n{sep}")
    print("  POWELL ROBUSTNESS ANALYSIS — STRATIFIED OOS ACROSS 5 MACRO REGIMES")
    print(sep)

    print(f"\n{'OOS SELECTION':}")
    header = f"  {'Regime':<22} {'Date':<12} {'Dir':>3} {'HawkScore':>9} {'ΔYield':>7}"
    print(header)
    print(f"  {'-'*22} {'-'*12} {'-'*3} {'-'*9} {'-'*7}")
    for _, row in split_df[split_df["split"] == "test"].iterrows():
        d = "↑ UP" if row["yield_up"] else "↓ DN"
        print(f"  {row['regime']:<22} {str(row['date'])[:10]:<12} {d:>4} "
              f"{row['hawkishness_phase2']:>9.1f} {row['yield_change']:>+7.3f}")

    print(f"\nSIGNAL STRENGTH  (n={len(cdf)} non-zero speeches)")
    print(f"  Yield UP   mean hawkishness: {up_scores.mean():.1f}  (n={len(up_scores)})")
    print(f"  Yield DOWN mean hawkishness: {down_scores.mean():.1f}  (n={len(down_scores)})")
    print(f"  Gap: {up_scores.mean() - down_scores.mean():+.1f} pts  "
          f"t={t_stat:.2f}  p={p_val:.3f}  r={pb_corr:.3f}")

    print(f"\nREGRESSION  (train n={len(train_cdf)}, OOS n={len(test_cdf)})")
    print(f"  {'Method':<8} {'IS R²':>8} {'OOS R²':>8} {'C&T OOS R²':>12} {'IS RMSE':>9} {'OOS RMSE':>9}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*9} {'-'*9}")
    for _, r in reg.iterrows():
        print(f"  {r['method']:<8} {r['is_r2']:>+8.4f} {r['oos_r2']:>+8.4f} "
              f"{r['oos_r2_ct']:>+12.4f} {r['is_rmse']:>9.4f} {r['oos_rmse']:>9.4f}")

    print(f"\nCLASSIFICATION  (train n={len(train_cdf)}, OOS n={len(test_cdf)})")
    print(f"  {'Model':<42} {'IS Acc':>7} {'OOS Acc':>8} {'IS AUC':>7} {'OOS AUC':>8}")
    print(f"  {'-'*42} {'-'*7} {'-'*8} {'-'*7} {'-'*8}")
    for _, r in clf.iterrows():
        oos_auc_str = f"{r['oos_auc']:.4f}" if not (isinstance(r['oos_auc'], float) and np.isnan(r['oos_auc'])) else "   nan"
        print(f"  {r['method']:<42} {r['is_accuracy']:>7.3f} {r['oos_accuracy']:>8.3f} "
              f"{r['is_auc']:>7.4f} {oos_auc_str:>8}")

    print(f"\nLOOCV  (hawkishness_phase2 only, n={loocv['n']})")
    print(f"  Accuracy={loocv['loocv_accuracy']:.3f}  "
          f"F1={loocv['loocv_f1']:.3f}  "
          f"AUC={loocv['loocv_auc']:.3f}")

    print(f"\n{sep}")
    best_oos   = clf.loc[clf["oos_auc"].notna(), "oos_auc"].max()
    best_model = clf.loc[clf["oos_auc"] == best_oos, "method"].values[0] if not np.isnan(best_oos) else "N/A"
    univar_row = clf[clf["method"] == "Logistic — hawkishness_phase2 only"].iloc[0]

    print("SUMMARY")
    print(f"  Signal gap:        {up_scores.mean() - down_scores.mean():+.1f} pts  "
          f"(p={p_val:.3f}, r={pb_corr:.3f})")
    print(f"  Univariate logistic: OOS AUC={univar_row['oos_auc']}  "
          f"OOS acc={univar_row['oos_accuracy']:.3f}  LOOCV AUC={loocv['loocv_auc']:.3f}")
    print(f"  Best OOS AUC: {best_oos} ({best_model})")
    print(f"  Regression best OOS R² (C&T): {reg['oos_r2_ct'].max():+.4f} (LASSO)")
    print(sep)


if __name__ == "__main__":
    main()

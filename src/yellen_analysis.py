#!/usr/bin/env python3
"""
Yellen-regime robustness analysis.

Single-regime test of the Phase 2 hawkishness pipeline on Janet Yellen's
22 speeches (2014–2017). The cleanest possible test: one chair, one communication
style, no crises, minimal regulatory contamination.

Primary question: does hawkishness_phase2 predict same-day 2yr yield direction?
Primary model: Logistic Regression (L2 fixed C=1.0 — CV skipped at n<15).

Train/test split: 2014–2016 (15 speeches) / 2017 (7 speeches).
Note: all 5 non-zero 2017 speeches had rising yields — the active hiking cycle.
OOS AUC is therefore undefined (single class); IS AUC and the mean-separation
test are the primary evidence of signal quality.

Outputs:
    data/processed/yellen_clusters.csv
    data/processed/yellen_regression_results.csv
    data/processed/yellen_classification_results.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.formula.api as smf

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

TRAIN_CUTOFF = pd.Timestamp("2016-12-31")
N_CLUSTERS   = 3
LOGISTIC_C   = 1.0   # fixed — CV unreliable at n<15

FEATURES = ["hawkishness_phase2", "DFF", "PCE_YOY", "UNRATE", "GDP_GROWTH"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Data ───────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    p2   = pd.read_csv(PROCESSED / "speech_scores_phase2_yellen.csv")
    macro = pd.read_csv(PROCESSED / "macro_context.csv")
    exp1  = pd.read_csv(PROCESSED / "speech_scores_yellen.csv")[
        ["filename", "hawkishness"]
    ].rename(columns={"hawkishness": "hawkishness_exp1"})

    df = p2.merge(
        macro[["filename", "DGS2", "DGS2_prev", "DFF", "PCE_YOY", "UNRATE", "GDP_GROWTH"]],
        on="filename", how="left",
    ).merge(exp1, on="filename", how="left")

    df["yield_change"] = df["DGS2"] - df["DGS2_prev"]
    df["date"]         = pd.to_datetime(df["date"])
    df["is_train"]     = df["date"] <= TRAIN_CUTOFF
    df = df.dropna(subset=["yield_change", "hawkishness_phase2"])
    return df.sort_values("date").reset_index(drop=True)


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
    ss_res  = ((y_te - yh_te) ** 2).sum()
    ss_tot  = ((y_te - y_tr.mean()) ** 2).sum()
    oos_ct  = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "method":    name,
        "is_r2":     round(r2_score(y_tr, yh_tr), 4),
        "oos_r2":    round(r2_score(y_te, yh_te), 4),
        "oos_r2_ct": round(oos_ct, 4),
        "is_rmse":   round(float(np.sqrt(mean_squared_error(y_tr, yh_tr))), 4),
        "oos_rmse":  round(float(np.sqrt(mean_squared_error(y_te, yh_te))), 4),
        "n_train":   int(len(y_tr)),
        "n_test":    int(len(y_te)),
    }


# ── Unsupervised: Yellen-only TF-IDF + K-means ─────────────────────────────────

def run_unsupervised(df: pd.DataFrame) -> pd.DataFrame:
    path = PROCESSED / "filtered_sentences_yellen.csv"
    if not path.exists():
        path = PROCESSED / "sentences_yellen.csv"

    sents = pd.read_csv(path)
    sents = sents[sents["filename"].isin(df["filename"])]

    docs = (
        sents.groupby("filename", as_index=False)
        .agg(doc_text=("text", lambda x: " ".join(x.dropna())))
        .merge(df[["filename", "date", "title", "hawkishness_phase2"]], on="filename", how="left")
    )

    vec = TfidfVectorizer(
        max_features=200, min_df=2, stop_words="english",
        ngram_range=(1, 2), sublinear_tf=True,
    )
    X = vec.fit_transform(docs["doc_text"]).toarray()
    names = vec.get_feature_names_out()

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    km     = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    docs["cluster"] = labels
    docs["pc1"]     = coords[:, 0]
    docs["pc2"]     = coords[:, 1]

    log.info(f"  PCA variance: PC1={pca.explained_variance_ratio_[0]:.1%}  PC2={pca.explained_variance_ratio_[1]:.1%}")
    order = km.cluster_centers_.argsort()[:, ::-1]
    for i in range(N_CLUSTERS):
        top   = [names[j] for j in order[i, :8]]
        count = (labels == i).sum()
        log.info(f"  Cluster {i} ({count} speeches): {', '.join(top)}")

    return docs[["filename", "cluster", "pc1", "pc2"]]


# ── Regression ─────────────────────────────────────────────────────────────────

def run_regression(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["is_train"]]
    test  = df[~df["is_train"]]

    X_tr, y_tr = train[FEATURES].values, train["yield_change"].values
    X_te, y_te = test[FEATURES].values,  test["yield_change"].values

    sc         = StandardScaler()
    X_tr_s     = sc.fit_transform(X_tr)
    X_te_s     = sc.transform(X_te)

    results = []

    ols = smf.ols(
        "yield_change ~ hawkishness_phase2 + DFF + PCE_YOY + UNRATE + GDP_GROWTH",
        data=train,
    ).fit(cov_type="HC3")
    results.append(eval_reg("OLS", y_tr, ols.fittedvalues.values, y_te, ols.predict(test)))

    lasso = LassoCV(cv=3, max_iter=10_000, random_state=42).fit(X_tr_s, y_tr)
    results.append(eval_reg("LASSO", y_tr, lasso.predict(X_tr_s), y_te, lasso.predict(X_te_s)))
    log.info(f"    LASSO selected α={lasso.alpha_:.4f}")

    ridge = RidgeCV(cv=3).fit(X_tr_s, y_tr)
    results.append(eval_reg("Ridge", y_tr, ridge.predict(X_tr_s), y_te, ridge.predict(X_te_s)))

    return pd.DataFrame(results)


# ── Classification ─────────────────────────────────────────────────────────────

def run_classification(df: pd.DataFrame) -> pd.DataFrame:
    cdf = df[df["yield_change"] != 0].copy()
    cdf["yield_up"] = (cdf["yield_change"] > 0).astype(int)

    train = cdf[cdf["is_train"]]
    test  = cdf[~cdf["is_train"]]

    X_tr, y_tr = train[FEATURES].values, train["yield_up"].values
    X_te, y_te = test[FEATURES].values,  test["yield_up"].values

    sc     = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    hi = FEATURES.index("hawkishness_phase2")  # hawkishness column index

    log.info(f"  Class balance — train: {y_tr.mean():.0%} up ({y_tr.sum()}/{len(y_tr)})  "
             f"test: {y_te.mean():.0%} up ({y_te.sum()}/{len(y_te)})")
    if len(np.unique(y_te)) == 1:
        log.info("  ⚠  Test set is single-class (all yield UP — 2017 hiking cycle). OOS AUC = nan.")

    results = []

    def fit_logistic(name, X_tr_in, X_te_in):
        m = LogisticRegression(C=LOGISTIC_C, penalty="l2", max_iter=2000, random_state=42)
        m.fit(X_tr_in, y_tr)
        return eval_clf(name,
            y_tr, m.predict(X_tr_in),  m.predict_proba(X_tr_in)[:, 1],
            y_te, m.predict(X_te_in),  m.predict_proba(X_te_in)[:, 1])

    # 1. Hawkishness only (Phase 2)
    results.append(fit_logistic(
        "Logistic — hawkishness_phase2 only",
        X_tr_s[:, [hi]], X_te_s[:, [hi]],
    ))

    # 2. Macro only (no hawkishness) — control
    macro_cols = [i for i, f in enumerate(FEATURES) if f != "hawkishness_phase2"]
    results.append(fit_logistic(
        "Logistic — macro only",
        X_tr_s[:, macro_cols], X_te_s[:, macro_cols],
    ))

    # 3. Full features
    results.append(fit_logistic(
        "Logistic — full (phase2 + macro)",
        X_tr_s, X_te_s,
    ))

    # 4. Experiment 1 hawkishness — direct comparison
    if "hawkishness_exp1" in cdf.columns:
        e1_tr = StandardScaler().fit_transform(train[["hawkishness_exp1"]].values)
        e1_te = StandardScaler().fit(train[["hawkishness_exp1"]].values).transform(
            test[["hawkishness_exp1"]].values
        )
        results.append(fit_logistic("Logistic — Exp 1 hawkishness only", e1_tr, e1_te))

    # 5. Random Forest (max_depth=2 — shallower given tiny train set)
    rf = RandomForestClassifier(n_estimators=500, max_depth=2, random_state=42)
    rf.fit(X_tr, y_tr)
    results.append(eval_clf(
        "Random Forest",
        y_tr, rf.predict(X_tr),  rf.predict_proba(X_tr)[:, 1],
        y_te, rf.predict(X_te),  rf.predict_proba(X_te)[:, 1],
    ))

    # 6. Majority-class baseline
    majority  = int(np.bincount(y_tr).argmax())
    base_prob = float(y_tr.mean())
    results.append(eval_clf(
        "Baseline (majority class)",
        y_tr, np.full_like(y_tr, majority), np.full(len(y_tr), base_prob),
        y_te, np.full_like(y_te, majority), np.full(len(y_te), base_prob),
    ))

    return pd.DataFrame(results)


# ── LOOCV supplement ────────────────────────────────────────────────────────────

def run_loocv(df: pd.DataFrame) -> dict:
    """
    Leave-one-out CV on full 22-speech dataset.
    Avoids the all-one-class OOS problem of the fixed split.
    Uses univariate logistic (hawkishness_phase2 only).
    """
    cdf = df[df["yield_change"] != 0].copy()
    cdf["yield_up"] = (cdf["yield_change"] > 0).astype(int)

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
        m = LogisticRegression(C=LOGISTIC_C, penalty="l2", max_iter=2000, random_state=42)
        m.fit(X_tr_s, y_tr)
        probs[test_idx] = m.predict_proba(X_te_s)[0, 1]

    preds = (probs >= 0.5).astype(int)
    return {
        "loocv_accuracy": round(accuracy_score(y, preds), 4),
        "loocv_f1":       round(f1_score(y, preds, zero_division=0), 4),
        "loocv_auc":      round(roc_auc_score(y, probs), 4),
        "n":              len(y),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading Yellen data...")
    df     = load_data()
    n_tr   = df["is_train"].sum()
    n_te   = (~df["is_train"]).sum()

    log.info(f"  {len(df)} speeches  ({n_tr} train 2014–2016 / {n_te} test 2017)")
    log.info(f"  Yield change: mean={df['yield_change'].mean():.4f}  "
             f"std={df['yield_change'].std():.4f}  "
             f"range=[{df['yield_change'].min():.3f}, {df['yield_change'].max():.3f}]")

    # ── Signal strength: mean hawkishness by yield direction ──────────────────
    cdf = df[df["yield_change"] != 0].copy()
    cdf["yield_up"] = (cdf["yield_change"] > 0).astype(int)

    up_scores   = cdf[cdf["yield_up"] == 1]["hawkishness_phase2"]
    down_scores = cdf[cdf["yield_up"] == 0]["hawkishness_phase2"]
    t_stat, p_val = stats.ttest_ind(up_scores, down_scores, equal_var=False)
    pb_corr = cdf["hawkishness_phase2"].corr(cdf["yield_up"].astype(float))

    log.info(f"\n── Signal Strength ─────────────────────────────────────────────────────────")
    log.info(f"  Mean hawkishness — yield UP:   {up_scores.mean():.1f}  (n={len(up_scores)})")
    log.info(f"  Mean hawkishness — yield DOWN: {down_scores.mean():.1f}  (n={len(down_scores)})")
    log.info(f"  Difference: {up_scores.mean() - down_scores.mean():+.1f} points")
    log.info(f"  Welch t-test: t={t_stat:.2f}  p={p_val:.3f}")
    log.info(f"  Point-biserial correlation: r={pb_corr:.3f}")

    # ── Unsupervised ──────────────────────────────────────────────────────────
    log.info(f"\n── Unsupervised: TF-IDF + K-means (k={N_CLUSTERS}) ──────────────────────────")
    cluster_df = run_unsupervised(df)
    cluster_df.to_csv(PROCESSED / "yellen_clusters.csv", index=False)

    # ── Regression ────────────────────────────────────────────────────────────
    log.info(f"\n── Regression ──────────────────────────────────────────────────────────────")
    reg = run_regression(df)
    for _, r in reg.iterrows():
        log.info(f"  {r['method']:<8}  IS R²={r['is_r2']:+.4f}  OOS R²={r['oos_r2']:+.4f}")
    reg.to_csv(PROCESSED / "yellen_regression_results.csv", index=False)

    # ── Classification ────────────────────────────────────────────────────────
    log.info(f"\n── Classification: predict yield direction ──────────────────────────────────")
    clf = run_classification(df)
    for _, r in clf.iterrows():
        log.info(
            f"  {r['method']:<40}  "
            f"IS acc={r['is_accuracy']:.3f}  OOS acc={r['oos_accuracy']:.3f}  "
            f"IS AUC={r['is_auc']}  OOS AUC={r['oos_auc']}"
        )
    clf.to_csv(PROCESSED / "yellen_classification_results.csv", index=False)

    # ── LOOCV ─────────────────────────────────────────────────────────────────
    log.info(f"\n── LOOCV (hawkishness_phase2 only, full {len(cdf)}-speech dataset) ──────────")
    loocv = run_loocv(df)
    log.info(f"  Accuracy={loocv['loocv_accuracy']:.3f}  "
             f"F1={loocv['loocv_f1']:.3f}  "
             f"AUC={loocv['loocv_auc']:.3f}  (n={loocv['n']})")

    # ── Print tables ──────────────────────────────────────────────────────────
    print("\n─── Signal Strength ──────────────────────────────────────────────────────────")
    print(f"  Mean hawkishness (yield UP):   {up_scores.mean():.1f}")
    print(f"  Mean hawkishness (yield DOWN): {down_scores.mean():.1f}")
    print(f"  Difference: {up_scores.mean() - down_scores.mean():+.1f} pts  "
          f"t={t_stat:.2f}  p={p_val:.3f}  r={pb_corr:.3f}")

    print("\n─── Regression (2014–2016 train / 2017 test) ─────────────────────────────────")
    print(reg[["method","is_r2","oos_r2","oos_r2_ct","is_rmse","oos_rmse"]].to_string(index=False))

    print("\n─── Classification (2014–2016 train / 2017 test) ────────────────────────────")
    print(clf[["method","is_accuracy","oos_accuracy","is_f1","oos_f1","is_auc","oos_auc"]].to_string(index=False))

    print(f"\n─── LOOCV — Logistic (hawkishness_phase2 only, n={loocv['n']}) ──────────────")
    print(f"  Accuracy={loocv['loocv_accuracy']:.3f}  F1={loocv['loocv_f1']:.3f}  AUC={loocv['loocv_auc']:.3f}")
    print("\n  Note: OOS AUC = nan because all 2017 test speeches had rising yields")
    print("  (consistent with the active hiking cycle). LOOCV AUC uses all 19 speeches.")


if __name__ == "__main__":
    main()

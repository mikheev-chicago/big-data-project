#!/usr/bin/env python3
"""
Phase 3 — Supervised learning: regression and classification.

Task A (Regression): predict same-day 2yr yield change (bps)
    Methods: OLS, LASSO (CV), Ridge (CV), Random Forest
    Metrics: IS R², OOS R² (vs test mean), OOS R² (vs train mean, C&T 2008),
             IS RMSE, OOS RMSE

Task B (Classification): predict yield direction (up=1, down=0)
    Methods: Logistic Regression (CV), Random Forest
    Baseline: majority-class classifier
    Metrics: IS/OOS Accuracy, F1, ROC-AUC

Train/test split: speeches ≤ 2020-12-31 (in-sample) vs 2021+ (out-of-sample).
Matches the split used in validation.py.

Usage:
    python src/phase3_supervised.py

Requires:
    data/processed/speech_scores_phase2.csv  — from pairwise_scoring.py
    data/processed/macro_context.csv         — from fetch_fred.py

Outputs:
    data/processed/phase3_regression_results.csv
    data/processed/phase3_classification_results.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

TRAIN_CUTOFF = pd.Timestamp("2020-12-31")

EVENING_SPEECHES = {
    "20100408_bernanke_Economic_Outlook_and_Fiscal_Challenges.txt",
    "20131119_bernanke_The_Federal_Reserve_Forty_Years_after_the_Reform.txt",
    "20190228_powell_Monetary_Policy_Patience_and_the_Economic_Outlook.txt",
}

# Features used in both tasks
FEATURES = [
    "hawkishness_phase2",
    "DFF",
    "PCE_YOY",
    "UNRATE",
    "GDP_GROWTH",
    "chair_powell",
    "chair_yellen",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Merge Phase 2 hawkishness scores with macro context.
    Returns one row per speech with yield_change, features, and is_train flag.
    """
    scores = pd.read_csv(PROCESSED / "speech_scores_phase2.csv")
    macro  = pd.read_csv(PROCESSED / "macro_context.csv")

    macro_cols = ["filename", "DGS2", "DGS2_prev", "DFF", "PCE_YOY", "UNRATE", "GDP_GROWTH"]
    df = scores.merge(macro[macro_cols], on="filename", how="left")

    df["yield_change"] = df["DGS2"] - df["DGS2_prev"]
    df = df[~df["filename"].isin(EVENING_SPEECHES)]
    df = df.dropna(subset=["yield_change", "hawkishness_phase2"])

    df["chair_powell"] = (df["chair_key"] == "powell").astype(int)
    df["chair_yellen"] = (df["chair_key"] == "yellen").astype(int)

    df["date"]     = pd.to_datetime(df["date"])
    df["is_train"] = df["date"] <= TRAIN_CUTOFF

    return df.reset_index(drop=True)


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def eval_regression(
    name: str,
    y_is: np.ndarray, yhat_is: np.ndarray,
    y_oos: np.ndarray, yhat_oos: np.ndarray,
) -> dict:
    train_mean = y_is.mean()
    ss_res = ((y_oos - yhat_oos) ** 2).sum()
    ss_tot_ct = ((y_oos - train_mean) ** 2).sum()
    oos_r2_ct = 1 - ss_res / ss_tot_ct if ss_tot_ct > 0 else float("nan")

    return {
        "method":    name,
        "is_r2":     round(r2_score(y_is, yhat_is), 4),
        "oos_r2":    round(r2_score(y_oos, yhat_oos), 4),
        "oos_r2_ct": round(oos_r2_ct, 4),
        "is_rmse":   round(float(np.sqrt(mean_squared_error(y_is, yhat_is))), 4),
        "oos_rmse":  round(float(np.sqrt(mean_squared_error(y_oos, yhat_oos))), 4),
        "n_train":   len(y_is),
        "n_test":    len(y_oos),
    }


def eval_classification(
    name: str,
    y_is: np.ndarray, yhat_is: np.ndarray, prob_is: np.ndarray,
    y_oos: np.ndarray, yhat_oos: np.ndarray, prob_oos: np.ndarray,
) -> dict:
    oos_auc = (
        round(roc_auc_score(y_oos, prob_oos), 4)
        if len(np.unique(y_oos)) > 1 else float("nan")
    )
    return {
        "method":       name,
        "is_accuracy":  round(accuracy_score(y_is, yhat_is), 4),
        "oos_accuracy": round(accuracy_score(y_oos, yhat_oos), 4),
        "is_f1":        round(f1_score(y_is, yhat_is, zero_division=0), 4),
        "oos_f1":       round(f1_score(y_oos, yhat_oos, zero_division=0), 4),
        "is_auc":       round(roc_auc_score(y_is, prob_is), 4),
        "oos_auc":      oos_auc,
        "n_train":      len(y_is),
        "n_test":       len(y_oos),
    }


# ── Task A: Regression ─────────────────────────────────────────────────────────

def task_a_regression(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["is_train"]].copy()
    test  = df[~df["is_train"]].copy()

    X_train = train[FEATURES].values
    y_train = train["yield_change"].values
    X_test  = test[FEATURES].values
    y_test  = test["yield_change"].values

    results = []

    # OLS with HC3 robust SEs — no scaling needed
    formula = (
        "yield_change ~ hawkishness_phase2 + DFF + PCE_YOY + UNRATE "
        "+ GDP_GROWTH + chair_powell + chair_yellen"
    )
    ols = smf.ols(formula, data=train).fit(cov_type="HC3")
    results.append(eval_regression(
        "OLS",
        y_train, ols.fittedvalues.values,
        y_test,  ols.predict(test),
    ))
    log.info(f"  OLS        IS R²={results[-1]['is_r2']:+.4f}  OOS R²={results[-1]['oos_r2']:+.4f}")

    # Standardize for sklearn linear models
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # LASSO (cross-validated alpha)
    lasso = LassoCV(cv=5, max_iter=10_000, random_state=42).fit(X_train_s, y_train)
    results.append(eval_regression(
        "LASSO",
        y_train, lasso.predict(X_train_s),
        y_test,  lasso.predict(X_test_s),
    ))
    log.info(f"  LASSO      IS R²={results[-1]['is_r2']:+.4f}  OOS R²={results[-1]['oos_r2']:+.4f}  α={lasso.alpha_:.4f}")

    # Ridge (cross-validated alpha)
    ridge = RidgeCV(cv=5).fit(X_train_s, y_train)
    results.append(eval_regression(
        "Ridge",
        y_train, ridge.predict(X_train_s),
        y_test,  ridge.predict(X_test_s),
    ))
    log.info(f"  Ridge      IS R²={results[-1]['is_r2']:+.4f}  OOS R²={results[-1]['oos_r2']:+.4f}  α={ridge.alpha_:.4f}")

    # Random Forest — no scaling, limit depth to curb overfitting on ~100 training pts
    rf = RandomForestRegressor(n_estimators=500, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)
    results.append(eval_regression(
        "RandomForest",
        y_train, rf.predict(X_train),
        y_test,  rf.predict(X_test),
    ))
    log.info(f"  RF         IS R²={results[-1]['is_r2']:+.4f}  OOS R²={results[-1]['oos_r2']:+.4f}")

    out = pd.DataFrame(results)
    log.info(f"\n  Feature importance (RF):")
    for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
        log.info(f"    {feat:<25} {imp:.3f}")

    return out


# ── Task B: Classification ─────────────────────────────────────────────────────

def task_b_classification(df: pd.DataFrame) -> pd.DataFrame:
    # Drop exact zero yield changes — direction is ambiguous
    df = df[df["yield_change"] != 0].copy()
    df["yield_up"] = (df["yield_change"] > 0).astype(int)

    train = df[df["is_train"]].copy()
    test  = df[~df["is_train"]].copy()

    X_train = train[FEATURES].values
    y_train = train["yield_up"].values
    X_test  = test[FEATURES].values
    y_test  = test["yield_up"].values

    log.info(f"  Class balance — train: {y_train.mean():.1%} up  |  test: {y_test.mean():.1%} up")

    results = []

    # Standardize for logistic regression
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Logistic Regression (L2, cross-validated C)
    lr = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    results.append(eval_classification(
        "LogisticRegression",
        y_train, lr.predict(X_train_s), lr.predict_proba(X_train_s)[:, 1],
        y_test,  lr.predict(X_test_s),  lr.predict_proba(X_test_s)[:, 1],
    ))
    log.info(f"  Logistic   IS acc={results[-1]['is_accuracy']:.3f}  OOS acc={results[-1]['oos_accuracy']:.3f}  OOS AUC={results[-1]['oos_auc']}")

    # Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=500, max_depth=4, random_state=42)
    rfc.fit(X_train, y_train)
    results.append(eval_classification(
        "RandomForest",
        y_train, rfc.predict(X_train), rfc.predict_proba(X_train)[:, 1],
        y_test,  rfc.predict(X_test),  rfc.predict_proba(X_test)[:, 1],
    ))
    log.info(f"  RF         IS acc={results[-1]['is_accuracy']:.3f}  OOS acc={results[-1]['oos_accuracy']:.3f}  OOS AUC={results[-1]['oos_auc']}")

    # Majority-class baseline
    majority = int(np.bincount(y_train).argmax())
    base_prob = float(y_train.mean())
    results.append(eval_classification(
        "Baseline (majority class)",
        y_train, np.full_like(y_train, majority), np.full(len(y_train), base_prob),
        y_test,  np.full_like(y_test, majority),  np.full(len(y_test), base_prob),
    ))
    log.info(f"  Baseline   IS acc={results[-1]['is_accuracy']:.3f}  OOS acc={results[-1]['oos_accuracy']:.3f}")

    return pd.DataFrame(results)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    scores_path = PROCESSED / "speech_scores_phase2.csv"
    if not scores_path.exists():
        sys.exit(
            "Error: speech_scores_phase2.csv not found.\n"
            "Run pairwise_scoring.py first to generate Phase 2 scores."
        )

    log.info("Loading data...")
    df = load_data()
    n_train = df["is_train"].sum()
    n_test  = (~df["is_train"]).sum()
    log.info(f"  {len(df)} speeches with yield data  ({n_train} train / {n_test} test)")

    # ── Task A ────────────────────────────────────────────────────────────────
    log.info("\n── Task A: Regression (predict same-day 2yr yield change) ──────────────")
    reg_results = task_a_regression(df)
    reg_path = PROCESSED / "phase3_regression_results.csv"
    reg_results.to_csv(reg_path, index=False)

    print("\nTask A — Regression Results:")
    print(reg_results[["method", "is_r2", "oos_r2", "oos_r2_ct", "is_rmse", "oos_rmse"]].to_string(index=False))

    # ── Task B ────────────────────────────────────────────────────────────────
    log.info("\n── Task B: Classification (predict yield direction) ─────────────────────")
    clf_results = task_b_classification(df)
    clf_path = PROCESSED / "phase3_classification_results.csv"
    clf_results.to_csv(clf_path, index=False)

    print("\nTask B — Classification Results:")
    print(clf_results[["method", "is_accuracy", "oos_accuracy", "is_f1", "oos_f1", "is_auc", "oos_auc"]].to_string(index=False))

    log.info(f"\nSaved:\n  {reg_path.name}\n  {clf_path.name}")


if __name__ == "__main__":
    main()

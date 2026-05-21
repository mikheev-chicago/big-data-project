#!/usr/bin/env python3
"""
Validation: OOS R², time-series sanity check, inter-score correlations.

1. Train/test split at 2021-01-01
   Train: 116 speeches (all Bernanke + all Yellen + Powell 2018-2020)
   Test:  22 speeches  (Powell 2021-2025, all out-of-sample)
   Fit S1-S4 on training data, evaluate on test data.
   Two OOS R² flavours:
     - Test-mean benchmark: SS_tot = Σ(y - mean(y_test))²
     - Train-mean benchmark: SS_tot = Σ(y - mean(y_train))²  [Campbell & Thompson 2008]

2. Time-series sanity check
   Quarterly average hawkishness per regime.
   Expected for Powell: peak Q3 2022 (75bp hike cycle), trough Q2 2020 (COVID response).

3. Inter-score correlations
   Pearson correlations among hawk_z, dove_z, net_z, hawkishness — per regime and pooled.

Outputs:
    data/processed/oos_results.csv           OOS metrics per spec
    data/processed/hawkishness_quarterly.csv quarterly hawkishness by regime
    data/processed/score_correlations.csv    pairwise correlation table
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

SPLIT_DATE = pd.Timestamp("2021-01-01")

EVENING_SPEECHES = {
    "20100408_bernanke_Economic_Outlook_and_Fiscal_Challenges.txt",
    "20131119_bernanke_The_Federal_Reserve_Forty_Years_after_the_Reform.txt",
    "20190228_powell_Monetary_Policy_Patience_and_the_Economic_Outlook.txt",
}

PRE_FOMC_WINDOW = 14
MACRO = "DFF + PCE_YOY + UNRATE + GDP_GROWTH"


# ── Data loading (mirrors regression_analysis.py) ─────────────────────────────

def load_data() -> pd.DataFrame:
    scores = pd.concat([
        pd.read_csv(PROCESSED / f"speech_scores_{c}.csv").assign(chair=c)
        for c in ["bernanke", "yellen", "powell"]
    ], ignore_index=True)

    macro = pd.read_csv(PROCESSED / "macro_context.csv")
    df = scores.merge(
        macro[["filename", "DGS2", "DGS2_prev", "DFF", "PCE_YOY", "UNRATE", "GDP_GROWTH"]],
        on="filename", how="left"
    )
    df["yield_change"] = df["DGS2"] - df["DGS2_prev"]
    df["date"]         = pd.to_datetime(df["date"])
    df = df[~df["filename"].isin(EVENING_SPEECHES)].copy()

    fomc_dates = pd.to_datetime(
        pd.read_csv(PROCESSED / "fomc_calendar.csv")["date"]
    )
    def pre_fomc_flag(d):
        return int(any(0 < (fd - d).days <= PRE_FOMC_WINDOW for fd in fomc_dates))
    df["pre_fomc"]     = df["date"].apply(pre_fomc_flag)
    df["chair_yellen"] = (df["chair"] == "yellen").astype(float)
    df["chair_powell"] = (df["chair"] == "powell").astype(float)

    df = df.dropna(subset=["yield_change", "hawkishness", "hawk_z", "dove_z",
                            "net_z", "DFF", "PCE_YOY", "UNRATE", "GDP_GROWTH"])
    return df.reset_index(drop=True)


# ── OOS R² helpers ────────────────────────────────────────────────────────────

def oos_r2(y_true: pd.Series, y_pred: pd.Series, baseline: float) -> float:
    """
    OOS R² = 1 - MSE_model / MSE_baseline
    baseline is either mean(y_test) or mean(y_train).
    """
    mse_model    = ((y_true - y_pred) ** 2).mean()
    mse_baseline = ((y_true - baseline) ** 2).mean()
    return 1 - mse_model / mse_baseline


# ── Section 1: OOS regression ─────────────────────────────────────────────────

def run_oos_regression(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["date"] < SPLIT_DATE].copy()
    test  = df[df["date"] >= SPLIT_DATE].copy()

    print(f"\n── Train/test split at {SPLIT_DATE.date()} ──────────────────────────")
    print(f"  Training: {len(train)} speeches  ({train['chair'].value_counts().to_dict()})")
    print(f"  Test:     {len(test)} speeches  ({test['chair'].value_counts().to_dict()})")
    print(f"  Train yield_change mean: {train['yield_change'].mean():.4f}")
    print(f"  Test  yield_change mean: {test['yield_change'].mean():.4f}")

    specs = {
        "S1": f"yield_change ~ {MACRO}",
        "S2": f"yield_change ~ hawkishness + {MACRO}",
        "S3": f"yield_change ~ hawkishness * pre_fomc + {MACRO}",
        "S4": f"yield_change ~ hawkishness * pre_fomc + {MACRO} + chair_yellen + chair_powell",
    }

    train_mean = train["yield_change"].mean()
    test_mean  = test["yield_change"].mean()
    rows = []

    print(f"\n{'Spec':<6} {'IS R²':>8} {'OOS R² (vs test mean)':>22} {'OOS R² (vs train mean, C&T)':>28}")
    print("─" * 68)

    for spec_name, formula in specs.items():
        # Fit on training data with cluster-robust SEs
        fitted = smf.ols(formula, data=train).fit(
            cov_type="cluster",
            cov_kwds={"groups": train["date"].astype(str).values}
        )
        is_r2 = fitted.rsquared

        # Predict on test data
        y_pred = fitted.predict(test)
        y_true = test["yield_change"]

        oos_vs_test  = oos_r2(y_true, y_pred, test_mean)
        oos_vs_train = oos_r2(y_true, y_pred, train_mean)   # Campbell & Thompson

        print(f"{spec_name:<6} {is_r2:>8.4f} {oos_vs_test:>22.4f} {oos_vs_train:>28.4f}")

        rows.append({
            "spec":           spec_name,
            "n_train":        len(train),
            "n_test":         len(test),
            "is_r2":          round(is_r2, 6),
            "oos_r2_testmean": round(oos_vs_test, 6),
            "oos_r2_trainmean": round(oos_vs_train, 6),
        })

    print("─" * 68)
    print("C&T = Campbell & Thompson (2008) — uses training mean as naive benchmark")

    return pd.DataFrame(rows)


# ── Section 2: Time-series sanity check ──────────────────────────────────────

def run_time_series_check(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n── Quarterly hawkishness by regime ──────────────────────────────")

    df = df.copy()
    df["quarter"] = df["date"].dt.to_period("Q")

    qt = (
        df.groupby(["chair", "quarter"])
        .agg(
            n         = ("hawkishness", "count"),
            avg_h     = ("hawkishness", "mean"),
            min_h     = ("hawkishness", "min"),
            max_h     = ("hawkishness", "max"),
        )
        .reset_index()
    )
    qt["quarter_str"] = qt["quarter"].astype(str)

    for chair in ["bernanke", "yellen", "powell"]:
        sub = qt[qt["chair"] == chair].sort_values("quarter")
        peak  = sub.loc[sub["avg_h"].idxmax()]
        trough = sub.loc[sub["avg_h"].idxmin()]
        print(f"\n  {chair.upper()}")
        print(f"  {'Quarter':<10} {'N':>4} {'Avg H':>8}  {'Min H':>8}  {'Max H':>8}")
        print(f"  {'─'*48}")
        for _, r in sub.iterrows():
            flag = ""
            if r["quarter_str"] == peak["quarter_str"]:
                flag = " ◀ PEAK"
            elif r["quarter_str"] == trough["quarter_str"]:
                flag = " ◀ TROUGH"
            print(f"  {r['quarter_str']:<10} {r['n']:>4} {r['avg_h']:>+8.3f}  {r['min_h']:>+8.3f}  {r['max_h']:>+8.3f}{flag}")

    # Powell-specific sanity check
    powell_qt = qt[qt["chair"] == "powell"].sort_values("avg_h", ascending=False)
    actual_peak   = powell_qt.iloc[0]
    actual_trough = powell_qt.iloc[-1]
    print(f"\n  Powell sanity check:")
    print(f"    Expected peak   Q3 2022 → actual peak   {actual_peak['quarter_str']} (avg H={actual_peak['avg_h']:+.3f})")
    print(f"    Expected trough Q2 2020 → actual trough {actual_trough['quarter_str']} (avg H={actual_trough['avg_h']:+.3f})")

    q3_2022 = powell_qt[powell_qt["quarter_str"] == "2022Q3"]
    q2_2020 = powell_qt[powell_qt["quarter_str"] == "2020Q2"]
    if not q3_2022.empty:
        rank_peak = (powell_qt["avg_h"] >= q3_2022.iloc[0]["avg_h"]).sum()
        print(f"    Q3 2022 rank (out of {len(powell_qt)} quarters): #{rank_peak} (avg H={q3_2022.iloc[0]['avg_h']:+.3f})")
    else:
        print("    Q3 2022: no speeches in this quarter")
    if not q2_2020.empty:
        rank_trough = (powell_qt["avg_h"] <= q2_2020.iloc[0]["avg_h"]).sum()
        print(f"    Q2 2020 rank (out of {len(powell_qt)} quarters): #{rank_trough} most dovish (avg H={q2_2020.iloc[0]['avg_h']:+.3f})")
    else:
        print("    Q2 2020: no speeches in this quarter")

    return qt[["chair", "quarter_str", "n", "avg_h", "min_h", "max_h"]].rename(
        columns={"quarter_str": "quarter"}
    )


# ── Section 3: Inter-score correlations ──────────────────────────────────────

def run_correlations(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n── Inter-score correlations ─────────────────────────────────────")
    score_cols = ["hawk_z", "dove_z", "net_z", "hawkishness"]

    corr_rows = []
    for group_name, sub in [("bernanke", df[df["chair"] == "bernanke"]),
                             ("yellen",   df[df["chair"] == "yellen"]),
                             ("powell",   df[df["chair"] == "powell"]),
                             ("pooled",   df)]:
        corr_matrix = sub[score_cols].corr(method="pearson")
        print(f"\n  {group_name.upper()}  (n={len(sub)})")
        header = f"  {'':12}" + "".join(f"{c:>12}" for c in score_cols)
        print(header)
        print("  " + "─" * (12 + 12 * len(score_cols)))
        for row_var in score_cols:
            line = f"  {row_var:<12}"
            for col_var in score_cols:
                val = corr_matrix.loc[row_var, col_var]
                line += f"{val:>12.3f}"
            print(line)

        for i, v1 in enumerate(score_cols):
            for v2 in score_cols[i+1:]:
                corr_rows.append({
                    "regime": group_name,
                    "var1":   v1,
                    "var2":   v2,
                    "pearson_r": round(corr_matrix.loc[v1, v2], 4),
                })

    return pd.DataFrame(corr_rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data()

    oos_df  = run_oos_regression(df)
    qt_df   = run_time_series_check(df)
    corr_df = run_correlations(df)

    oos_df.to_csv(PROCESSED / "oos_results.csv", index=False)
    qt_df.to_csv(PROCESSED / "hawkishness_quarterly.csv", index=False)
    corr_df.to_csv(PROCESSED / "score_correlations.csv", index=False)

    print(f"\nSaved:")
    print(f"  data/processed/oos_results.csv")
    print(f"  data/processed/hawkishness_quarterly.csv")
    print(f"  data/processed/score_correlations.csv")


if __name__ == "__main__":
    main()

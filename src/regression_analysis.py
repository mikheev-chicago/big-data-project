#!/usr/bin/env python3
"""
Regression analysis: speech hawkishness → same-day 2yr Treasury yield change

Outcome: yield_change = DGS2(speech day close) − DGS2_prev(prior day close), in pp.

Five specifications, all with cluster-robust SEs clustered by date:
    S1   Macro-only baseline: DFF + PCE_YOY + UNRATE + GDP_GROWTH
    S2   S1 + hawkishness  (primary composite z-score)
    S3   S2 + pre_fomc dummy + hawkishness × pre_fomc interaction
    S4   S3 + chair fixed effects (Bernanke = reference category)
    S5   Robustness: replace composite hawkishness with each sub-score separately
         S5a: hawk_z   S5b: dove_z (expect negative sign)   S5c: net_z

Pre-FOMC window: speech falls within 14 calendar days before an FOMC announcement date.

Outputs:
    data/processed/regression_results.csv   tidy per-coefficient table
    data/processed/regression_summary.csv   per-specification model stats (N, R², etc.)
    Prints formatted regression tables to console.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

EVENING_SPEECHES = {
    "20100408_bernanke_Economic_Outlook_and_Fiscal_Challenges.txt",
    "20131119_bernanke_The_Federal_Reserve_Forty_Years_after_the_Reform.txt",
    "20190228_powell_Monetary_Policy_Patience_and_the_Economic_Outlook.txt",
}

PRE_FOMC_WINDOW = 14   # calendar days before FOMC announcement

STARS = {0.01: "***", 0.05: "**", 0.10: "*"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_analysis_data() -> pd.DataFrame:
    # Speech scores (all three regimes)
    scores = pd.concat([
        pd.read_csv(PROCESSED / f"speech_scores_{c}.csv").assign(chair=c)
        for c in ["bernanke", "yellen", "powell"]
    ], ignore_index=True)

    # Macro data: yield variables + controls
    macro = pd.read_csv(PROCESSED / "macro_context.csv")
    df = scores.merge(
        macro[["filename", "DGS2", "DGS2_prev", "DFF", "PCE_YOY", "UNRATE", "GDP_GROWTH"]],
        on="filename", how="left"
    )
    df["yield_change"] = df["DGS2"] - df["DGS2_prev"]
    df["date"] = pd.to_datetime(df["date"])

    # Remove evening speeches (market had closed before speech)
    df = df[~df["filename"].isin(EVENING_SPEECHES)].copy()

    # Pre-FOMC flag
    fomc_dates = pd.to_datetime(pd.read_csv(PROCESSED / "fomc_calendar.csv")["date"])
    def pre_fomc_flag(speech_date):
        for fd in fomc_dates:
            days_until = (fd - speech_date).days
            if 0 < days_until <= PRE_FOMC_WINDOW:
                return 1
        return 0
    df["pre_fomc"] = df["date"].apply(pre_fomc_flag)

    # Chair dummies (Bernanke = reference)
    df["chair_yellen"] = (df["chair"] == "yellen").astype(float)
    df["chair_powell"] = (df["chair"] == "powell").astype(float)

    # Drop any rows with missing required values
    required = ["yield_change", "hawkishness", "hawk_z", "dove_z", "net_z",
                "DFF", "PCE_YOY", "UNRATE", "GDP_GROWTH"]
    df = df.dropna(subset=required).reset_index(drop=True)
    return df


# ── OLS with cluster-robust SEs ───────────────────────────────────────────────

def fit_cluster(formula: str, df: pd.DataFrame):
    model  = smf.ols(formula, data=df)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["date"].astype(str).values}
    )
    return result


# ── Table formatting helpers ──────────────────────────────────────────────────

def star(p: float) -> str:
    for threshold, s in sorted(STARS.items()):
        if p < threshold:
            return s
    return ""


def fmt_coef(coef: float, pval: float) -> str:
    return f"{coef:+.4f}{star(pval)}"


def fmt_se(se: float) -> str:
    return f"({se:.4f})"


def print_table(results: list, labels: list, var_order: list, title: str):
    col_w = 14
    sep = "─" * (30 + col_w * len(results))
    print(f"\n{title}")
    print(sep)

    # Column headers
    hdr = f"{'':30}" + "".join(f"{lbl:>{col_w}}" for lbl in labels)
    print(hdr)
    print(sep)

    for var in var_order:
        coef_line = f"{var:30}"
        se_line   = f"{'':30}"
        any_found = False
        for res in results:
            if var in res.params:
                coef_line += f"{fmt_coef(res.params[var], res.pvalues[var]):>{col_w}}"
                se_line   += f"{fmt_se(res.bse[var]):>{col_w}}"
                any_found  = True
            else:
                coef_line += f"{'':>{col_w}}"
                se_line   += f"{'':>{col_w}}"
        if any_found:
            print(coef_line)
            print(se_line)

    print(sep)
    for stat_name, getter in [
        ("N",       lambda r: str(int(r.nobs))),
        ("R²",      lambda r: f"{r.rsquared:.4f}"),
        ("Adj. R²", lambda r: f"{r.rsquared_adj:.4f}"),
    ]:
        row = f"{stat_name:30}" + "".join(f"{getter(r):>{col_w}}" for r in results)
        print(row)
    print(sep)
    print("Significance: * p<0.10  ** p<0.05  *** p<0.01")
    print("Cluster-robust SEs by date in parentheses")


# ── Tidy output helpers ───────────────────────────────────────────────────────

def results_to_tidy(results: list, spec_names: list) -> pd.DataFrame:
    rows = []
    for spec, res in zip(spec_names, results):
        for var in res.params.index:
            rows.append({
                "spec":    spec,
                "variable": var,
                "coef":    res.params[var],
                "se":      res.bse[var],
                "tstat":   res.tvalues[var],
                "pvalue":  res.pvalues[var],
                "stars":   star(res.pvalues[var]),
            })
    return pd.DataFrame(rows)


def summary_to_df(results: list, spec_names: list) -> pd.DataFrame:
    rows = []
    for spec, res in zip(spec_names, results):
        rows.append({
            "spec":        spec,
            "n":           int(res.nobs),
            "r2":          round(res.rsquared, 6),
            "r2_adj":      round(res.rsquared_adj, 6),
            "f_pvalue":    round(res.f_pvalue, 6) if hasattr(res, "f_pvalue") else None,
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_analysis_data()
    n_total = len(df)

    pre_fomc_n = df["pre_fomc"].sum()
    chair_dist = df["chair"].value_counts().to_dict()
    print(f"Analysis sample: {n_total} speeches")
    print(f"  Chair breakdown: {chair_dist}")
    print(f"  Pre-FOMC speeches (within {PRE_FOMC_WINDOW} days): {pre_fomc_n}")
    print(f"  Yield change: mean={df['yield_change'].mean():.4f} std={df['yield_change'].std():.4f} "
          f"range=[{df['yield_change'].min():.3f}, {df['yield_change'].max():.3f}]")

    MACRO = "DFF + PCE_YOY + UNRATE + GDP_GROWTH"

    # ── S1–S4: main specifications ────────────────────────────────────────────
    r1 = fit_cluster(f"yield_change ~ {MACRO}", df)
    r2 = fit_cluster(f"yield_change ~ hawkishness + {MACRO}", df)
    r3 = fit_cluster(f"yield_change ~ hawkishness * pre_fomc + {MACRO}", df)
    r4 = fit_cluster(f"yield_change ~ hawkishness * pre_fomc + {MACRO} + chair_yellen + chair_powell", df)

    main_results = [r1, r2, r3, r4]
    main_labels  = ["S1 Macro", "S2 +Hawk", "S3 +FOMC×Hawk", "S4 +Chair FE"]
    main_vars    = [
        "hawkishness",
        "hawkishness:pre_fomc",
        "pre_fomc",
        "DFF",
        "PCE_YOY",
        "UNRATE",
        "GDP_GROWTH",
        "chair_yellen",
        "chair_powell",
        "Intercept",
    ]

    print_table(main_results, main_labels, main_vars,
                "Dependent variable: yield_change (2yr Treasury, pp)")

    # ── S5: sub-score robustness ──────────────────────────────────────────────
    r5a = fit_cluster(f"yield_change ~ hawk_z + {MACRO}", df)
    r5b = fit_cluster(f"yield_change ~ dove_z + {MACRO}", df)
    r5c = fit_cluster(f"yield_change ~ net_z  + {MACRO}", df)

    rob_results = [r5a, r5b, r5c]
    rob_labels  = ["S5a hawk_z", "S5b dove_z", "S5c net_z"]
    rob_vars    = ["hawk_z", "dove_z", "net_z",
                   "DFF", "PCE_YOY", "UNRATE", "GDP_GROWTH", "Intercept"]

    print_table(rob_results, rob_labels, rob_vars,
                "S5 Sub-score robustness (macro controls included, not shown for brevity)")

    # ── Key finding summary ───────────────────────────────────────────────────
    hawk_coef = r2.params.get("hawkishness", float("nan"))
    hawk_pval = r2.pvalues.get("hawkishness", float("nan"))
    int_coef  = r3.params.get("hawkishness:pre_fomc", float("nan"))
    int_pval  = r3.pvalues.get("hawkishness:pre_fomc", float("nan"))
    print(f"\nKey results:")
    print(f"  S2 hawkishness coef: {hawk_coef:+.4f}  p={hawk_pval:.3f}{star(hawk_pval)}")
    print(f"  S3 hawkishness×pre_fomc: {int_coef:+.4f}  p={int_pval:.3f}{star(int_pval)}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    all_results   = main_results + rob_results
    all_spec_names = ["S1", "S2", "S3", "S4", "S5a", "S5b", "S5c"]

    tidy_df   = results_to_tidy(all_results, all_spec_names)
    summary_df = summary_to_df(all_results, all_spec_names)

    tidy_df.to_csv(PROCESSED / "regression_results.csv", index=False)
    summary_df.to_csv(PROCESSED / "regression_summary.csv", index=False)

    print(f"\nSaved:")
    print(f"  data/processed/regression_results.csv  ({len(tidy_df)} coefficient rows)")
    print(f"  data/processed/regression_summary.csv  ({len(summary_df)} specs)")
    print(f"\nModel summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

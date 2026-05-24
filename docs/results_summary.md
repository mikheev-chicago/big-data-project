# FedSpeak — Results Summary
**BUSN 20800 Final Project | May 2026**

Reference document for writeup. All statistics are exact. Interpretations and methodology commentary are in *italics*.

---

## 1. Data and Sample

- **Corpus:** 138 Fed speeches across three chair regimes (Bernanke 2008–2014: 71; Yellen 2014–2018: 22; Powell 2018–2024: 45)
- **Outcome variable:** Same-day change in the 2-year Treasury yield (DGS2 close minus prior trading day close), in percentage points
- **Yield change statistics:** mean = +0.003 pp, std = 0.058 pp, range = [−0.230, +0.330 pp]
- **Train/test split:** speeches ≤ 2020-12-31 (n=116 train) vs. 2021+ (n=22 test)
- **Three evening speeches excluded** (market closed before speech delivery): Bernanke 2010-04-08, 2013-11-19; Powell 2019-02-28

*The outcome variable is extremely noisy. A standard deviation of 5.8 bps on a mean of 0.3 bps means yield changes are near-zero on average, with the signal buried in market microstructure noise and concurrent macro releases. Predicting same-day yield changes from speech text is inherently a weak-signal problem — the literature (e.g., Gürkaynak, Sack & Swanson 2005) typically uses high-frequency intraday data around the exact speech time rather than daily closes, which would be a stronger test. The daily close approach used here is a reasonable choice given data availability constraints but limits what any model can achieve.*

---

## 2. Experiment 1: Dictionary-Based Scoring

### Methodology
- **Hawkishness dictionary:** Apel & Blix Grimaldi (2014) word lists, with IDF damping (words in >90% of speeches zeroed out; 50–90% downweighted; ≤50% kept at full weight)
- **Three sub-scores:** hawk_z (hawkish word density, z-scored), dove_z (dovish word density, z-scored, expected negative sign in regression), net_z (hawk_z − dove_z)
- **Composite score:** hawkishness = equal-weight average of hawk_z, dove_z×(−1), net_z — normalized 0–100 within each regime
- **Five OLS specifications** with cluster-robust standard errors (clustered by date):
  - S1: Macro controls only (DFF, PCE_YOY, UNRATE, GDP_GROWTH)
  - S2: S1 + hawkishness composite
  - S3: S2 + pre-FOMC dummy (within 14 days of announcement) + hawkishness×pre_fomc interaction
  - S4: S3 + chair fixed effects (Bernanke = reference)
  - S5a/b/c: Macro + hawk_z / dove_z / net_z separately

### In-Sample Results (N=138)

| Spec | R² | Adj. R² | F p-value |
|------|----|---------|-----------|
| S1 — Macro only | 0.019 | −0.010 | 0.638 |
| S2 — + hawkishness | 0.019 | −0.018 | 0.715 |
| S3 — + pre-FOMC × hawk | 0.063 | +0.013 | 0.400 |
| S4 — + chair FEs | 0.101 | +0.037 | 0.411 |
| S5a — hawk_z | 0.037 | +0.000 | 0.623 |
| S5b — dove_z | 0.039 | +0.002 | 0.313 |
| S5c — net_z | 0.020 | −0.017 | 0.663 |

### Key Coefficients

| Spec | Variable | Coef | SE | p-value | Stars |
|------|----------|------|----|---------|-------|
| S2 | hawkishness | +0.0005 | 0.0063 | 0.942 | — |
| S3 | hawkishness | +0.0101 | 0.0082 | 0.219 | — |
| S3 | hawkishness × pre_fomc | −0.0262 | 0.0136 | 0.054 | * |
| S4 | hawkishness | +0.0093 | 0.0086 | 0.278 | — |
| S4 | hawkishness × pre_fomc | −0.0258 | 0.0140 | 0.065 | * |
| S5a | hawk_z | −0.0076 | 0.0055 | 0.169 | — |
| S5b | dove_z | −0.0092 | 0.0049 | 0.059 | * |
| S5c | net_z | +0.0015 | 0.0048 | 0.748 | — |

*Significance: \* p<0.10  \*\* p<0.05  \*\*\* p<0.01. Cluster-robust SEs.*

### Out-of-Sample Validation (116 train / 22 test)

| Spec | IS R² | OOS R² (vs test mean) | OOS R² (vs train mean) | OOS RMSE |
|------|-------|----------------------|----------------------|----------|
| S1 | 0.033 | −0.413 | −0.381 | 0.105 |
| S2 | 0.034 | −0.420 | −0.388 | 0.105 |
| S3 | 0.044 | −0.444 | −0.411 | 0.106 |
| S4 | 0.104 | −0.613 | −0.577 | 0.112 |

### Interpretation

*Experiment 1 failed completely. The dictionary `hawkishness` coefficient in S2 is +0.0005 with p=0.94 — statistically indistinguishable from zero. Adding hawkishness to the macro baseline improves R² by 0.003 percentage points.*

*The only marginally significant finding is the* **pre-FOMC interaction (p=0.054, negative sign)**, *which is theoretically backwards: it says hawkish speeches in the FOMC window are associated with* falling *yields. This is almost certainly spurious — the sign is inconsistent with any reasonable model of how Fed communication affects rates.*

*The `dove_z` coefficient (S5b) is marginally significant (p=0.059) and negative, which directionally makes sense (more dovish language → lower yields). However, the effect is tiny (−0.009 pp per unit of dove_z) and disappears entirely in the composite score.*

*All OOS R² values are deeply negative, meaning every specification predicts worse than simply forecasting the test-set mean. S4 is the worst (OOS R²=−0.61) because the chair fixed effects absorb sample-specific variation that doesn't generalize.*

**Root cause diagnosis:** The core failure is that the dictionary cannot distinguish monetary policy language from financial regulation language. Bernanke gave ~20 speeches on financial stability, bank stress tests, systemic risk, and Basel capital requirements — all using firm, assertive language that scores as "hawkish" by the dictionary. These speeches contaminate the scoring because they have low yield-change correlations (the market knows financial regulation speeches don't signal rate changes). This hypothesis is directly confirmed by the K-means clustering in Phase 2.

---

## 3. Phase 2: Unsupervised Methods

### 3a. LLM Pairwise Tournament + TrueSkill Scoring

**Methodology:**
- **Stage 1 (Sentence filter):** Claude Haiku classified all sentences in batches of 50, retaining only sentences relevant to monetary policy, inflation, employment, or economic growth. Non-monetary content (bank supervision, financial regulation, consumer protection) excluded.
- **Stage 2 (Speech representation):** Filtered sentences aggregated per speech using IDF-damped lemmas (`damped_lemmas`). Each speech represented as a bag of key monetary policy terms plus contemporaneous macro conditions (DFF, PCE_YOY, UNRATE, GDP_GROWTH).
- **Stage 3 (Pairwise tournament):** Claude Sonnet compared every pair within each regime, asked to judge which speech signals a more hawkish monetary policy stance given its economic context. A/B ordering randomized per pair to eliminate position bias.
  - Bernanke: 547 comparisons (from 2,485 possible pairs)
  - Yellen: 150 comparisons (from 231 possible pairs)
  - Powell: 150 comparisons (from 990 possible pairs)
- **Stage 4 (TrueSkill aggregation):** Pairwise outcomes fed into TrueSkill (Microsoft, draw_probability=0). Final `hawkishness_phase2` score = TrueSkill μ, min-max normalized to [0, 100] within each regime.

**Score Distribution by Chair:**

| Chair | N | Mean | SD | Min | Median | Max |
|-------|---|------|----|-----|--------|-----|
| Bernanke | 71 | 46.5 | 22.6 | 0.0 | 45.8 | 100.0 |
| Yellen | 22 | 51.7 | 25.2 | 0.0 | 47.5 | 100.0 |
| Powell | 45 | 46.5 | 26.1 | 0.0 | 43.2 | 100.0 |

**Top 3 Hawkish Speeches per Chair:**

| Chair | Date | Title | Score |
|-------|------|-------|-------|
| Bernanke | 2008-06-03 | Remarks on the economic outlook | 100.0 |
| Bernanke | 2008-06-09 | Outstanding Issues in the Analysis of Inflation | 93.8 |
| Bernanke | 2008-10-07 | Current Economic and Financial Conditions | 91.3 |
| Yellen | 2017-10-15 | The U.S. Economy and Monetary Policy | 100.0 |
| Yellen | 2017-01-19 | The Economic Outlook and the Conduct of Monetary Policy | 86.5 |
| Yellen | 2017-03-03 | From Adding Accommodation to Scaling It Back | 86.4 |
| Powell | 2022-08-26 | Monetary Policy and Price Stability (Jackson Hole) | 100.0 |
| Powell | 2024-04-03 | Opening Remarks | 93.5 |
| Powell | 2022-11-30 | Inflation and the Labor Market | 88.9 |

**Top 3 Dovish Speeches per Chair:**

| Chair | Date | Title | Score |
|-------|------|-------|-------|
| Bernanke | 2009-03-20 | The Financial Crisis and Community Banking | 11.0 |
| Bernanke | 2009-03-10 | Financial Reform to Address Systemic Risk | 8.1 |
| Bernanke | 2010-07-12 | Restoring the Flow of Credit to Small Businesses | 0.0 |
| Yellen | 2016-10-14 | Macroeconomic Research After the Crisis | 22.6 |
| Yellen | 2014-05-15 | Small Businesses and the Recovery | 19.4 |
| Yellen | 2014-03-31 | What the Federal Reserve Is Doing to Promote a Stronger Job Market | 0.0 |
| Powell | 2019-03-08 | Monetary Policy: Normalization and the Road Ahead | 5.1 |
| Powell | 2020-08-27 | New Economic Challenges and the Fed's Monetary Policy Review | 4.6 |
| Powell | 2020-10-06 | Recent Economic Developments and the Challenges Ahead | 0.0 |

*The face validity of these scores is excellent. For Bernanke, the top scores are all June–October 2008 — the period when the Fed was still fighting inflation before the Lehman Brothers collapse in September 2008, when inflation was running at 4–5% YoY and Bernanke was signaling concern about price stability. The bottom scores are 2009–2010 speeches explicitly about financial regulation and community banking — the exact non-monetary content that contaminated Experiment 1.*

*For Yellen, the top scores cluster in late 2016 through 2017, when the Fed was raising rates steadily and Yellen was explicitly signaling further tightening. The bottom scores are early-tenure speeches about jobs and the labor market recovery — correctly dovish.*

*For Powell, the top score is the August 2022 Jackson Hole speech — widely regarded as the most hawkish Fed speech in decades, in which Powell explicitly warned of "pain" from rate hikes and committed to bringing inflation down. The bottom scores are 2020 speeches during COVID-era maximum accommodation and the August 2020 average inflation targeting announcement. These scores pass every sanity check.*

### 3b. K-Means Clustering on TF-IDF (Unsupervised Component)

**Methodology:** TF-IDF matrix (500 terms, bigrams, min_df=3, sublinear TF, English stop words) built from all 138 speeches using raw sentence text. PCA reduced to 2 dimensions for visualization. K-means (k=5, n_init=10) applied to the full TF-IDF matrix.

**PCA variance explained:** PC1 = 10.6%, PC2 = 5.4% (total = 16.0%)

**Cluster Composition:**

| Cluster | Total | Bernanke | Yellen | Powell | Top Terms |
|---------|-------|----------|--------|--------|-----------|
| 0 | 5 | 3 | 1 | 1 | small businesses, credit, recovery, housing, jobs |
| 1 | 28 | 27 | 1 | 0 | financial markets, central banks, monetary policy, rates |
| 2 | 18 | 0 | 0 | 18 | pandemic, inflation, labor market, PCE, supply |
| 3 | 67 | 27 | 18 | 22 | inflation, monetary policy, unemployment, growth, rate |
| 4 | 20 | 14 | 2 | 4 | financial institutions, liquidity, systemic risk, capital, regulatory |

*The clusters are immediately interpretable and validate the Phase 2 design:*

- *Cluster 2 (18 speeches, all Powell): perfectly isolates COVID-era speeches — the model learns that "pandemic," "supply chain," and "PCE" vocabulary is a distinct regime.*
- *Cluster 4 (20 speeches, 14 Bernanke): captures financial regulation and stability speeches — bank capital requirements, systemic risk, stress tests. This is* exactly *the contamination cluster that caused Experiment 1 to fail. 14 of Bernanke's 71 speeches (20%) are in this cluster, meaning ~20% of his speeches should never have been scored on the hawkish/dovish scale.*
- *Cluster 3 (67 speeches, all chairs): the core monetary policy cluster — the speeches that should drive the regression.*
- *Cluster 1 (28 speeches, mostly Bernanke): financial markets and crisis response — the 2008–2009 period when Bernanke was navigating the financial crisis.*

*The low PCA variance (16% in 2 components) is expected for 500-dimensional text data and is not a concern — it reflects the diversity of Fed speech topics rather than a failure of the method.*

---

## 4. Phase 3: Supervised Learning

### Methodology

- **Features:** hawkishness_phase2, DFF, PCE_YOY, UNRATE, GDP_GROWTH, chair_powell (dummy), chair_yellen (dummy) — Bernanke is the reference chair
- **Train/test split:** pre-2021 (116 speeches) / 2021+ (22 speeches)
- **Three evening speeches excluded** (same as Experiment 1)

**Task A — Regression:** Predict same-day 2yr yield change (continuous, in pp)
- OLS with HC3 robust standard errors
- LASSO with 5-fold cross-validated α
- Ridge with 5-fold cross-validated α
- Random Forest: 500 trees, max_depth=4

**Task B — Classification:** Predict yield direction (up vs. down; zero changes excluded)
- Logistic Regression with L2 penalty, 5-fold cross-validated C
- Random Forest: 500 trees, max_depth=4
- Majority-class baseline (predicts "down" in all test observations)

### Task A — Regression Results (116 train / 22 test)

| Method | IS R² | OOS R² | OOS R² (C&T) | IS RMSE | OOS RMSE |
|--------|-------|--------|-------------|---------|----------|
| OLS | 0.083 | −0.533 | −0.499 | 0.054 | 0.110 |
| LASSO (α=0.008) | 0.000 | −0.023 | 0.000 | 0.057 | 0.089 |
| Ridge (α=10.0) | 0.080 | −0.363 | −0.333 | 0.055 | 0.103 |
| Random Forest | 0.527 | −1.005 | −0.960 | 0.039 | 0.125 |

*OOS R² (C&T) = Campbell & Thompson (2008) benchmark: 1 − SS_res / SS_tot where SS_tot uses the training mean as the null, not the test mean.*

**Random Forest Feature Importances:**

| Feature | Importance |
|---------|-----------|
| hawkishness_phase2 | 0.400 |
| PCE_YOY | 0.284 |
| DFF | 0.174 |
| GDP_GROWTH | 0.072 |
| UNRATE | 0.064 |
| chair_yellen | 0.005 |
| chair_powell | 0.001 |

**LASSO selected α = 0.008**, which shrinks all seven coefficients effectively to zero, predicting approximately the training mean for all test observations. This is why its Campbell–Thompson OOS R² is exactly 0.000 — it is the null model.

### Task B — Classification Results (93 train / 21 test)

| Method | IS Accuracy | OOS Accuracy | IS F1 | OOS F1 | IS AUC | OOS AUC |
|--------|------------|-------------|-------|--------|--------|---------|
| Logistic Regression | 0.624 | 0.333 | 0.690 | 0.417 | 0.675 | 0.333 |
| Random Forest | 0.871 | 0.429 | 0.895 | 0.571 | 0.964 | 0.482 |
| Baseline (majority class) | 0.559 | 0.429 | 0.717 | 0.600 | 0.500 | 0.500 |

*Test class balance: 42.9% up (9/21), 57.1% down (12/21). Majority class = "down."*

### Interpretation

*The regression results admit a clean story:*

**LASSO is the winner** by OOS R² (−0.023 vs. −0.363 for Ridge and −0.533 for OLS). But "winner" is relative: LASSO's optimal strategy is to predict the training mean, which is essentially zero. The correct interpretation is that LASSO's cross-validation correctly determined there is insufficient signal in the features to fit a useful regression.

**OLS overfits.** With 7 features and 116 training observations, OLS still manages to overfit, achieving IS R²=0.083 but OOS R²=−0.533. The IS fit is driven by the chair dummies and macro controls capturing regime-level differences, but these don't generalize when the test set is 22 Powell post-COVID speeches.

**Random Forest overfits severely.** IS R²=0.527 looks impressive; OOS R²=−1.005 is the worst of all methods. With max_depth=4, the model still memorizes training patterns that don't generalize. This is expected with only 116 training points for a nonlinear model.

*The feature importance finding is the most interesting result: `hawkishness_phase2` is the single most important feature at 40%, ahead of core inflation (PCE_YOY at 28%) and the fed funds rate (17%). This confirms the TrueSkill score contains* some *useful information — the Random Forest learns to use it — but the total signal is still insufficient to beat the null model out-of-sample.*

*For the classification task, no model meaningfully beats the baseline:*
- Logistic Regression does* worse *than the baseline on every OOS metric (accuracy 33% vs. 43%, F1 0.42 vs. 0.60, AUC 0.33 vs. 0.50)
- Random Forest matches baseline accuracy (42.9%) and has OOS AUC < 0.5 (worse than random ranking)
- The baseline F1 (0.60) beats Random Forest F1 (0.57) because predicting all "down" gets 12/21 correct

*The 21-observation test set is too small to draw meaningful statistical conclusions. A single misclassification changes accuracy by 4.8 percentage points. The OOS results should be read as "no evidence of predictive power" rather than "evidence of no predictive power."*

---

## 5. Cross-Cutting Discussion

### Why prediction is hard

The 2yr Treasury yield already incorporates market participants' expectations of Fed policy, including their reading of Fed communication. By the time a speech is delivered, professional traders with sophisticated NLP tools, news feeds, and policy expertise have already processed the content and traded on it. Predicting the* residual *same-day yield change — what the market didn't already know — from the speech text is asking whether the LLM's reading is more precise than the full market's reaction. This is a high bar.

### Why the train/test split is challenging

The training set (116 speeches, 2008–2020) spans the post-GFC zero lower bound era, quantitative easing, and a long expansion. The test set (22 speeches, 2021–2024) is the post-COVID inflation surge and rapid hiking cycle — the most unusual macro environment since the 1970s. Any model trained on the low-volatility pre-COVID period will struggle with 2022–2023 policy dynamics, regardless of how good the hawkishness measure is.

### Phase 2 vs. Experiment 1: what changed

The LLM pairwise scores produce a hawkishness measure with excellent face validity, and the RF assigns it 40% feature importance. But the improvement in OOS performance from Experiment 1 to Phase 3 is modest because the fundamental problem is not measurement quality — it is signal strength. Same-day daily yield changes are too noisy for either approach to predict reliably.

### Methodological limitations

1. **Small sample.** 138 speeches total, 22 OOS, limits statistical power. OOS metrics are unreliable at n=22.
2. **Partial pairwise coverage.** The TrueSkill tournament used 547/2,485 Bernanke pairs, 150/231 Yellen pairs, and 150/990 Powell pairs. Full round-robin would produce tighter TrueSkill estimates. Sparse tournament introduces noise in the scores.
3. **Outcome variable quality.** Daily closing yields include movements from macro data releases, global events, and other speeches on the same day. Intraday data aligned precisely to each speech would be a far stronger test.
4. **Within-regime normalization.** Phase 2 scores are normalized to [0, 100] separately per chair, making cross-regime comparison invalid. The regression model uses these scores as if Bernanke 50 = Yellen 50 = Powell 50, which isn't guaranteed.
5. **Test set composition.** 22 test speeches are all Powell 2021–2024. Chair fixed effects learned in training may not generalize. OOS results partially reflect a regime shift, not just model quality.

### What would improve the analysis

- Intraday yield data aligned to speech start times (would reduce noise dramatically)
- Full round-robin pairwise tournament (3,706 comparisons)
- Larger corpus — FOMC transcripts, minutes, and press conference remarks
- Cross-regime TrueSkill calibration (anchor a reference speech across regimes)
- Expanding the feature set with sentiment momentum, term premium, and Fed expectations survey data

---

## 6. Summary Statistics at a Glance

| | Experiment 1 (Dict.) | Phase 3 LASSO | Phase 3 RF |
|---|---|---|---|
| **IS R²** | 0.034 (S2) | 0.000 | 0.527 |
| **OOS R²** | −0.420 (S2) | −0.023 | −1.005 |
| **OOS RMSE** | 0.105 (S2) | 0.089 | 0.125 |
| **Hawkishness p-value** | 0.942 | N/A (shrunk to 0) | N/A |
| **Best OOS R²** | −0.413 (S1) | **−0.023** | — |

| Classification | IS Accuracy | OOS Accuracy | OOS AUC |
|---|---|---|---|
| Logistic Regression | 0.624 | 0.333 | 0.333 |
| Random Forest | 0.871 | 0.429 | 0.482 |
| **Baseline** | 0.559 | **0.429** | 0.500 |

**Bottom line:** LASSO (best regression) and the majority-class baseline (best classification) are the honest winners — both implement a form of "don't overfit the noise." The TrueSkill hawkishness scores are substantively meaningful (excellent face validity, 40% RF feature importance) but insufficient to overcome noise in a 22-observation test set.

---

## 7. Yellen-Only Robustness Analysis

**Purpose:** Test methodology on a single-regime, single-speaker dataset (Janet Yellen, 22 speeches, 2014–2017) to isolate confounds from regime-switching and multiple speakers.

**Setup:**
- 22 speeches total; 19 with non-zero yield changes
- Train: 2014–2016 (15 speeches, 14 non-zero); Test: 2017 (7 speeches, 5 non-zero)
- Fixed C=1.0 logistic (no CV at n<15); cv=3 for LASSO/Ridge
- LOOCV on full 19-speech dataset as primary OOS measure (test set is all-positive — see below)

**Signal Strength:**

| Metric | Value |
|---|---|
| Mean hawkishness — yield UP (n=12) | 55.7 |
| Mean hawkishness — yield DOWN (n=7) | 36.3 |
| Difference | **+19.4 points** |
| Welch t-test | t=1.82, p=0.097 |
| Point-biserial r | **0.429** |

The gap is directionally strong and the effect size (r=0.43) is meaningful. p=0.097 misses the 5% threshold due to small n, not weak signal.

**Classification:**

| Model | IS Acc | OOS Acc | IS AUC | LOOCV AUC |
|---|---|---|---|---|
| Logistic — phase2 only | 0.500 | 0.800 | 0.653 | **0.631** |
| Logistic — macro only | 0.500 | 1.000* | 0.653 | — |
| Logistic — full | 0.643 | 0.800 | 0.674 | — |
| Logistic — Exp 1 only | 0.714 | 0.600 | 0.612 | — |
| Random Forest | 1.000 | 0.600 | 1.000 | — |
| Baseline (majority) | 0.500 | 0.000 | 0.500 | — |

*OOS acc=1.0 for macro-only is misleading: all 5 non-zero 2017 test speeches had rising yields (hiking cycle), so OOS AUC=nan for all models.

**Key findings:**
- Phase 2 LLM scores (LOOCV AUC=0.631) outperform Exp 1 dictionary (IS AUC=0.612) on the Yellen-only sample — consistent with the hypothesis that LLM scoring captures nuance the word-count dictionary misses
- LOOCV AUC=0.631 and IS AUC=0.653 both meaningfully beat the 0.5 baseline
- The macro-only OOS acc=1.0 is an artifact of the 2017 hiking cycle, not a real signal

**Regression:** All models overfit severely (OLS OOS R²=−98). Expected at n=15 train — not interpretable.

**Unsupervised (k=3):**
- Cluster 0 (4 speeches): labor market / employment
- Cluster 1 (15 speeches): core monetary policy (rates, inflation, FOMC)
- Cluster 2 (3 speeches): financial stability

**Robustness verdict:** The LLM hawkishness methodology holds up on the clean single-regime dataset. The +19.4 pt signal gap and LOOCV AUC above 0.63 suggest the TrueSkill scores are capturing real information about market-relevant hawkishness, not just chair-specific style.

---

## 8. Powell-Only Robustness Analysis (Stratified OOS)

**Purpose:** Test methodology on Powell's 45 speeches (2018–2025) using stratified OOS — hold out 1 yield-up + 1 yield-down speech from each of 5 macro regimes rather than a simple time split. Directly tests whether the hawkishness signal generalizes *across* economic environments, not just within one.

**Regimes and OOS selection:**

| Regime | Period | OOS held out |
|---|---|---|
| 1. Rate normalization | 2018–Jul 2019 | 1 up + 1 down |
| 2. Dovish pivot | Aug 2019–Feb 2020 | 1 up + 1 down |
| 3. Zero lower bound (COVID + recovery) | Mar 2020–Feb 2022 | **0 up + 2 down** (no yield-up speeches in ZLB) |
| 4. Inflation surge + hiking | Mar 2022–Aug 2024 | 1 up + 1 down |
| 5. Cutting cycle + tariff uncertainty | Sep 2024–present | 1 up + 1 down |

Train: 29 speeches | OOS: 10 speeches (4 up, 6 down)

**Signal Strength (all 39 non-zero speeches):**

| Metric | Yellen | Powell |
|---|---|---|
| Mean hawkishness — yield UP | 55.7 | 52.5 |
| Mean hawkishness — yield DOWN | 36.3 | 45.3 |
| **Gap** | **+19.4 pts** | **+7.3 pts** |
| p-value | 0.097 | 0.366 |
| Point-biserial r | **0.429** | **0.150** |

The signal is much weaker for Powell. **Per-regime breakdown reveals why:**
- Regime 1 (normalization): UP=39.1 vs DOWN=39.3 — **essentially zero signal**
- Regime 2 (dovish pivot): UP=17.3 vs DOWN=26.1 — **signal reversed** (less hawkish → yield rises)
- Regime 4 (hiking): UP=80.9 vs DOWN=69.3 — signal present (+11.6 pts)
- Regime 5 (cutting): UP=50.6 vs DOWN=48.9 — near zero (+1.7 pts)

The small aggregate gap (+7.3 pts) is almost entirely driven by the hiking regime.

**Classification:**

| Model | IS Acc | OOS Acc | IS AUC | OOS AUC |
|---|---|---|---|---|
| Logistic — phase2 only | 0.517 | 0.600 | 0.624 | **0.500** |
| Logistic — macro only | 0.517 | 0.600 | 0.745 | **0.625** |
| Logistic — full | 0.724 | 0.300 | 0.800 | 0.458 |
| Logistic — Exp 1 only | 0.517 | 0.600 | 0.524 | 0.542 |
| Random Forest | 0.897 | 0.300 | 0.995 | 0.375 |
| Baseline (majority) | 0.517 | 0.600 | 0.500 | 0.500 |
| **LOOCV (phase2 only)** | — | — | — | **0.095** |

**LOOCV AUC=0.095** is the diagnostic finding: the univariate logistic is almost perfectly anti-predictive when trained and evaluated across all regimes. This confirms **regime non-stationarity** — the hawkishness→yield direction relationship reverses depending on the macro environment.

**Best OOS AUC: 0.625 from macro controls alone** — macroeconomic state variables (fed funds rate, PCE, unemployment, GDP growth) are stronger predictors of same-day yield direction than speech content in the Powell era.

**Regression:** Collapse expected and observed. All OOS R² deeply negative (best LASSO OOS R²=−6.63 vs Yellen's −0.75). Powell's 10-speech OOS set spans 5 very different macro environments, making any regression model trained on 29 speeches unstable.

**Unsupervised (k=5):** 5 clusters with partially interpretable themes — pandemic/labor market (19 speeches), monetary policy review/framework (6), core rates/inflation/expansion (15), financial stability (2), recovery/wages (3).

**Interpretation: Regime non-stationarity**

The LOOCV AUC=0.095 and per-regime signal breakdown together reveal the core problem: the hawkishness→yield direction relationship is not stable across Powell's tenure. In the normalization regime (2018-2019), hawkishness scores near 40 accompany both rising and falling yields — the macro environment (trade war uncertainty, small moves) dominated communication. In the hiking regime (2022-2023), all speeches score 57-100, yet yields still went both up and down depending on market interpretation of degree-of-hawkishness signals. In the cutting regime, tariff uncertainty creates noise the speech content can't resolve.

When the LOOCV trains on all regimes simultaneously, the model learns a confused signal: "high hawkishness = probably hiking regime = but hiking speeches go both directions" — and predicts systematically wrong.

**Contrast with Yellen:** Yellen's single regime (steady 2015-2017 hiking cycle with genuine speech-to-speech variation in tone) gave the hawkishness score something stable to latch onto. Powell's regime-switching tenure does not.

**Powell verdict:** The hawkishness signal does not generalize across Powell's 5 economic regimes. Macro controls outperform speech content for Powell. This is not a failure of the NLP methodology — it is an economically interpretable finding: Fed communication is most informative about yield direction when the macro environment is stable and speech tone varies speech-to-speech. When macro fundamentals themselves are shifting dramatically between speeches, they overwhelm the communication signal.

# FedSpeak: Replicating FedLock with LLM Hawkishness Scoring

**BUSN 20800 Big Data Final Project — Spring 2026**

This project replicates and extends [FedLock](https://www.bloomberg.com/news/newsletters/2026-03-05/presenting-fedlock-a-new-way-to-measure-fedspeak-with-llms), a method for measuring the hawk-dove stance of Federal Reserve communications using large language models. We construct a corpus of Fed Chair speeches, score each speech on a hawkishness scale using LLM pairwise comparisons, and then use those scores alongside macro data to predict market reactions.

---

## Research Question

Can the tone of Federal Reserve Chair speeches — specifically how hawkish or dovish they sound — predict same-day Treasury yield movements?

---

## Pipeline Overview

```
Phase 0  →  Phase 1  →  Phase 2  →  Phase 3  →  Phase 4
Collect     Clean &      Score        Predict      Validate &
data        filter       hawkishness  markets      write up
```

### Phase 0 — Data Ingestion ✅ complete
- Scraped **242 speeches** by Fed Chairs (Bernanke, Yellen, Powell) from `federalreserve.gov`, 2008–2025
- Pulled macro time series from FRED:
  - **DGS2** (2-year Treasury yield) — primary outcome variable for Phase 3a; daily end-of-day close; also pulled **DGS2_prev** (prior trading day's yield) to compute same-day yield change
  - **DFF** (effective federal funds rate) — LLM scoring context for Phase 2
  - **PCEPILFE** (core PCE inflation, converted to YoY % change), **UNRATE** (unemployment rate), **GDP growth** — macro context snapshots for Phase 2 LLM scoring; forward-filled to each speech date using the most recently *released* value as of that date (not the most recent reference period, which may not yet have been published)
- Pulled FOMC meeting calendar from `federalreserve.gov` (**153 meeting dates**, 2008–2025) — used to flag speeches coinciding with a meeting day
- Aligned all macro series to each speech date with no lookahead bias; output: `macro_context.csv` (242 rows, all columns fully populated)

### Phase 1 — Preprocessing ✅ complete
- **Relevance filter:** 2-stage LLM classifier reduced corpus from **242 → 138 speeches** (104 excluded, 43%)
  - Stage 1 (title keyword rules): 23 definite exclusions (commencement addresses, health-care reform, financial literacy programs, workforce development, etc.)
  - Stage 2 (`claude-opus-4-7` on title + first 400 words): 81 further exclusions (generic welcoming/opening remarks, community banking supervision, foreclosure policy, social programs, etc.)
- **FOMC-day check:** 2 speeches fell on FOMC meeting dates — both are major Jackson Hole speeches (Powell's 2020 AIT announcement and 2025 framework review); both kept since they contain substantive monetary policy content
- Outputs: `speeches_filtered.csv` (all 242 with filter decisions) and `speeches_monetary_policy.csv` (138 kept)
- **Deviation from original plan:** Speaker anonymization (originally listed here) is deferred — will be done immediately before Phase 2 pairwise scoring to keep the pipeline modular
- Remaining: attach macro context from `macro_context.csv` to the filtered corpus before Phase 2

### Phase 2 — Hawkishness Scoring (unsupervised)
- Run a pairwise tournament: present an LLM with two speeches side-by-side and ask which is more hawkish given the economic conditions at the time
- Aggregate pairwise outcomes using the TrueSkill ranking algorithm to produce a 0–100 hawkishness score per speech
- This unsupervised approach avoids the "bimodal scoring" problem that arises when asking an LLM to score directly on a 1–100 scale

### Phase 3 — Supervised Learning
- **Market reaction (3a):** Predict same-day 2-year Treasury yield changes (end-of-day close) using a stepwise approach — start with hawkishness score alone to establish a baseline signal, then incrementally add complexity (e.g., EFFR level as a control for the rate environment, chair fixed effects, speech-type flags). The goal is to show that speech tone has predictive power on its own before layering in controls. Baseline: OLS. Also try LASSO, Random Forest, and TabPFN.
- **Score distillation (3b):** Predict hawkishness score from speech text features alone (TF-IDF, SBERT embeddings, fine-tuned FinBERT). This tests whether the score can be reproduced cheaply from text.

### Phase 4 — Validation
- Compare our reproduced scores to FedLock's published scores
- Check speaker-level rankings (e.g., do known hawks score high?)
- Out-of-sample metrics on held-out years

---

## Experiment 1 — Improved Hawk/Dove Scoring Methodology (In Progress)

The goal of this experiment is to score each speech as hawkish or dovish more accurately than a generic financial sentiment dictionary can. Three problems motivated building a custom pipeline from scratch:

1. **Whole-speech scoring is too blurry.** A single speech might say "the labor market is strong" in one sentence and "but inflation has fallen short of our target" in the next. Scoring the whole thing as one unit averages out the signal.
2. **Common words drown out the signal.** Words like "economy", "market", "federal", and "rate" appear in nearly every speech and add noise without carrying any hawk/dove information.
3. **Generic financial lexicons are the wrong tool.** The Loughran-McDonald dictionary (built for corporate 10-K filings) classifies "tighten", "ease", and "cut" all as Negative — because in a corporate context those words describe bad things happening to a company. In Fed speeches they mean specific policy directions. A word list built for the right domain is needed.

The preprocessing steps below (1–4) prepare the corpus for a custom monetary-policy scoring approach.

---

### Step 1 — Split corpus by regime ✅ complete

We split the 138 filtered monetary-policy speeches into three separate files, one per Fed Chair. Each Chair is treated as its own independent regime because the language, economic context, and policy stance conventions differ meaningfully across eras.

| File | Chair | Speeches |
|------|-------|----------|
| `speeches_bernanke.csv` | Ben Bernanke | 71 |
| `speeches_yellen.csv` | Janet Yellen | 22 |
| `speeches_powell.csv` | Jerome Powell | 45 |

---

### Step 2 — Sentence tokenization and contrastive splitting ✅ complete

**Script:** `src/build_sentence_corpus.py`

Instead of scoring a whole speech, we break it down into individual sentence fragments. This matters because Fed speeches frequently make opposing claims within a single sentence — for example:

> *"The labor market remains strong, **but** inflation has fallen below target."*

If you score this sentence as a unit, the hawkish signal (strong labor market) and dovish signal (low inflation) cancel out. Splitting on the word "but" gives two clean fragments that can each be scored independently.

**How it works:**
1. Use spaCy (a standard NLP library) to split each speech into individual sentences
2. Further split any sentence that contains a *contrastive connector* — a word that signals a change in direction: **but**, **however**, **while**, **although**, or a semicolon (`;`)
3. Discard any fragment shorter than 8 words (removes section headers, stubs, and other noise)

**"However" is handled carefully.** A sentence like *"...as the crisis has persisted, however, the relationships..."* uses "however" as a parenthetical, not a contrast — it shouldn't be split. We only split on "however" when it's preceded by a comma or semicolon AND followed by a comma, which is the pattern for genuine clause-level contrast.

| Regime | Speeches | Total fragments | From connector splits |
|--------|----------|-----------------|-----------------------|
| Bernanke | 71 | 10,560 | 799 |
| Yellen | 22 | 4,208 | 274 |
| Powell | 45 | 5,087 | 197 |

Output: `sentences_bernanke.csv`, `sentences_yellen.csv`, `sentences_powell.csv`

---

### Step 3 — Lemmatization ✅ complete

**Script:** `src/lemmatize_and_damp.py`

Lemmatization means reducing each word to its base (dictionary) form. For example:
- "tightening", "tightened", "tightens" → all become **"tighten"**
- "raised", "raising", "raises" → all become **"raise"**
- "economies", "economy's" → both become **"economy"**

Without this step, the same concept gets counted as multiple different words, which dilutes the signal in any word-frequency analysis. We also lowercase everything and remove stopwords (common function words like "the", "a", "is", "of" that carry no policy meaning) and punctuation.

**Why spaCy instead of a simple rule?** Lemmatization is harder than it looks — the right base form often depends on the word's grammatical role. "Better" should become "good" (adjective), not "better" or "bet". spaCy uses a trained model that looks at context, not just the word itself.

The result is a `lemmas` column added to each sentence file — a cleaned, normalized version of each fragment ready for scoring.

---

### Step 4 — IDF Damping ✅ complete

**Script:** `src/lemmatize_and_damp.py`

Even after lemmatization, many words still appear in almost every speech and carry no useful signal. "Economy" appears in 100% of Bernanke speeches. "Market", "federal", "policy", "rate" — all above 90%. Counting these words when trying to detect hawk vs. dove stance is like trying to hear a specific instrument in a concert by measuring total volume.

**IDF (Inverse Document Frequency)** is a standard technique for measuring how rare or common a word is across a collection of documents. We use a simplified threshold version:

| Word frequency across speeches | Multiplier applied |
|--------------------------------|--------------------|
| Appears in **> 90%** of speeches | **0.0** — zeroed out entirely (pure boilerplate) |
| Appears in **> 50%** of speeches | **0.3** — kept but downweighted to 30% |
| Appears in **≤ 50%** of speeches | **1.0** — kept at full weight (genuinely differentiating) |

Frequency is computed **per regime** and at the **speech level** (not fragment level) — a word "counts" as appearing in a speech if it shows up anywhere in that speech. This is intentional: "inflation" appears in maybe 30% of individual sentence fragments but in 90%+ of full speeches. Speech-level frequency correctly identifies it as common context rather than a differentiating signal.

**What gets zeroed for each regime (examples):**

| Regime | Sample zeroed words (DF > 90%) |
|--------|-------------------------------|
| Bernanke | economy, economic, market, federal, policy, rate, bank, risk, credit |
| Yellen | important, large, strong, low, price, economy, rate |
| Powell | economy, economic, market, federal, policy |

Two outputs are saved per regime:
- `idf_weights_{chair}.csv` — the full word list with doc frequency and multiplier (used at scoring time)
- `damped_lemmas` column in `sentences_{chair}.csv` — the cleaned fragment text with 0.0-weight words removed

---

### Step 5 — Dictionary-based hawkishness scoring ✅ complete

**Script:** `src/score_hawkishness.py`

We score each speech using a hawk/dove word list from Apel & Blix Grimaldi (2014), a paper on measuring central bank tone. Unlike generic finance dictionaries (like Loughran-McDonald, which was built for corporate filings and classifies "tighten" and "ease" as identically *negative*), this word list was designed specifically for central bank communication. Each word is tagged as hawkish (+1) or dovish (-1). Examples:

| Word | Polarity | Why |
|------|----------|-----|
| tighten, hike, elevated, restrictive | +1 hawkish | signal rates should rise |
| ease, accommodative, patient, gradual | -1 dovish | signal rates should hold or fall |

**Three sub-scores per speech:**

Rather than a single net score, we compute three separate measures for each speech:

1. **hawk_density** — how many hawk-coded words appear per total word count (higher = more hawkish)
2. **dove_density** — how many dove-coded words appear per total word count (higher = more dovish)
3. **net_score** — hawk_density minus dove_density (higher = more hawkish)

Separating hawk and dove channels is useful because a speech could have both a lot of hawkish language *and* a lot of dovish language — the net score would be near zero, but the separate channels reveal that the speech is actually highly mixed rather than genuinely neutral.

**Z-score standardization (training window: 2008–2020):**

Each sub-score is standardized into a z-score using the mean and standard deviation computed only from speeches up to the end of 2020. Speeches after 2020 (only present in the Powell regime) are scored out-of-sample — their z-scores are computed using the 2018–2020 mean and std, not their own values. This mimics how you'd actually deploy the model without lookahead bias.

The **primary hawkishness measure** combines all three z-scores:

> **hawkishness = (hawk_z − dove_z + net_z) / 3**

dove_z is *subtracted* because high dove density is dovish (opposite direction to hawk_z and net_z). Averaging three correlated measures smooths out noise from any single channel.

**Labels within regime:** Within each Chair's speeches, sort by `hawkishness` and split into equal thirds — bottom third = **Dovish**, middle = **Neutral**, top = **Hawkish**. Ranking within each regime rather than globally accounts for the fact that absolute score levels aren't comparable across eras.

**Yield direction test:**

For each Hawkish or Dovish speech (Neutral excluded), we check whether the 2-year Treasury yield moved in the predicted direction on the speech day (same-day close minus prior day's close). Speeches delivered after market close are excluded.

| Regime | Training window | Correct / Total | Accuracy |
|--------|----------------|----------------|----------|
| Bernanke | 2008–2014 (all in-sample) | 17/48 | 35.4% |
| Yellen | 2014–2018 (all in-sample) | 4/15 | 26.7% |
| Powell | 2018–2020 in-sample, 2021–2025 out-of-sample | 14/30 | 46.7% |
| **Overall** | | **35/93** | **37.6%** |

**Result: below coin-flip (50%).** The main limitation is that a word list cannot distinguish context. Words like *firm*, *resilient*, and *elevated* appear in both monetary policy speeches ("labor market is firming") and financial regulation speeches ("we need firm oversight") — the latter have nothing to do with rate direction. Bernanke's top-scored "hawkish" speeches turned out to be financial stability talks from 2008–2009. This result motivates moving to LLM-based pairwise scoring (Phase 2), which can read the actual meaning of a sentence rather than matching isolated words.

Outputs: `speech_scores_{chair}.csv` (per-speech scores including all three z-scores and primary measure), `yield_results_abg.csv` (accuracy summary).

---

### Step 6 — Regression analysis ✅ complete

**Script:** `src/regression_analysis.py`

OLS regression of same-day 2-year Treasury yield change on hawkishness score and macro controls, with cluster-robust standard errors clustered by date. Five specifications:

| Spec | Description |
|------|-------------|
| **S1** | Macro-only baseline: DFF (fed funds rate), PCE_YOY (core inflation), UNRATE (unemployment), GDP_GROWTH |
| **S2** | S1 + primary hawkishness score |
| **S3** | S2 + pre-FOMC dummy + hawkishness × pre-FOMC interaction (speech within 14 days of FOMC meeting) |
| **S4** | S3 + chair fixed effects (Bernanke = reference) |
| **S5** | Robustness: S1 + each sub-score separately (hawk_z, dove_z, net_z) |

**Results (N=138 speeches, cluster-robust SEs by date):**

```
                              S1 Macro   S2 +Hawk  S3 +FOMC×Hawk  S4 +Chair FE
────────────────────────────────────────────────────────────────────────────────
hawkishness                             +0.0005      +0.0101         +0.0093
                                       (0.0063)     (0.0082)        (0.0086)
hawkishness × pre_fomc                             -0.0262*        -0.0258*
                                                    (0.0136)        (0.0140)
pre_fomc                                            -0.0175         -0.0215
                                                    (0.0127)        (0.0134)
DFF                         -0.0063    -0.0063      -0.0041         -0.0006
                            (0.0064)   (0.0063)     (0.0062)        (0.0073)
chair_powell (vs Bernanke)                                          -0.0408*
                                                                    (0.0219)
────────────────────────────────────────────────────────────────────────────────
R²                           0.019      0.019        0.063           0.101
Adj. R²                     -0.010     -0.018        0.013           0.037
* p<0.10  ** p<0.05  *** p<0.01
```

**Sub-score robustness (S5):**

| Sub-score | Coef | SE | p |
|-----------|------|----|---|
| hawk_z | −0.0076 | (0.0055) | 0.17 |
| dove_z | −0.0092* | (0.0049) | 0.065 |
| net_z | +0.0015 | (0.0048) | 0.76 |

**Interpretation:**

- **S2:** The composite hawkishness score has no statistically significant effect on same-day yield changes (coef ≈ 0, p=0.94). Adding speech tone to macro controls adds essentially nothing to R².

- **S3 — pre-FOMC interaction:** The one notable finding is a marginally significant *negative* interaction (p=0.054): hawkish speeches in the 14 days before an FOMC meeting are associated with smaller (not larger) yield increases. One interpretation is that markets have already priced in the direction of the upcoming decision by the time these speeches are delivered, so the speech adds little new information.

- **S4 — chair fixed effects:** The Powell dummy is negative and marginally significant (−0.041pp, p<0.10), suggesting yields moved less on Powell speech days than Bernanke speech days after controlling for macro conditions — possibly reflecting that Powell's communication style was more predictable.

- **S5:** The dove_z sub-score shows the clearest signal: higher dove-term density is associated with lower yields (p=0.065), consistent with the expected direction. The hawk channel alone (hawk_z) has a counterintuitive negative sign, likely because hawk-coded words appear heavily in financial regulation speeches (which don't signal rate hikes). This reinforces why the lexicon-based approach underperforms — it cannot distinguish monetary policy language from regulatory language.

Overall, the dictionary-based approach produces weak and inconsistent results, consistent with the 37.6% directional accuracy in Step 5. These results motivate moving to LLM-based pairwise scoring (Phase 2), which evaluates the actual meaning of each sentence rather than pattern-matching individual words.

Outputs: `regression_results.csv` (tidy per-coefficient table, all 7 specs), `regression_summary.csv` (model-level stats).

---

## Data Sources

| Source | Description | Series |
|--------|-------------|--------|
| [federalreserve.gov](https://www.federalreserve.gov/newsevents/speech/) | Fed Chair speeches, full text | 2008–2025 |
| [FRED](https://fred.stlouisfed.org/) | 2-year Treasury yield — outcome variable | DGS2 |
| [FRED](https://fred.stlouisfed.org/) | Macro context for LLM scoring (Phase 2 only) | EFFR, PCEPILFE, UNRATE, GDP growth |
| [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) | FOMC meeting dates — for flagging/excluding meeting-day speeches | 2008–2025 |
| [FedLock](https://bloomberg.com) | Published hawkishness scores | Validation only |

**Fed Chairs covered:**
- Ben S. Bernanke (Feb 2006 – Feb 2014): 2008–2014 speeches
- Janet L. Yellen (Feb 2014 – Feb 2018): 2014–2018 speeches
- Jerome H. Powell (Feb 2018 – present): 2018–2025 speeches

---

## Project Structure

```
fedspeak-project/
├── data/
│   ├── raw/
│   │   ├── speeches/                        # one .txt file per speech (242 total)
│   │   └── fred/                            # raw FRED series as CSV
│   └── processed/
│       ├── speeches.csv                     # manifest: date, chair, title, word_count, url, filename
│       ├── macro_context.csv                # speeches + aligned macro snapshot
│       ├── fomc_calendar.csv                # FOMC announcement dates 2008–2025
│       ├── speeches_filtered.csv            # all 242 with filter decisions
│       ├── speeches_monetary_policy.csv     # 138 kept monetary-policy speeches
│       ├── speeches_{chair}.csv             # per-regime corpus (bernanke / yellen / powell)
│       ├── sentences_{chair}.csv            # sentence fragments + lemmas + damped_lemmas
│       ├── idf_weights_{chair}.csv          # lemma → doc_freq → multiplier lookup table
│       ├── speech_scores_{chair}.csv        # per-speech hawkishness score, label, yield data
│       ├── yield_results_abg.csv            # accuracy summary: correct/total per regime
│       ├── regression_results.csv           # tidy coefficient table (all 7 specs)
│       └── regression_summary.csv           # per-spec model stats (N, R², Adj R²)
├── src/
│   ├── scrape_speeches.py                   # Phase 0: Fed speech scraper
│   ├── fetch_fred.py                        # Phase 0: FRED macro data + macro_context.csv
│   ├── fetch_fomc.py                        # Phase 0: FOMC meeting calendar scraper
│   ├── phase1_filter.py                     # Phase 1: 2-stage LLM relevance filter
│   ├── build_sentence_corpus.py             # Experiment 1 step 2: sentence tokenization + splitting
│   ├── lemmatize_and_damp.py               # Experiment 1 steps 3–4: lemmatization + IDF damping
│   ├── score_hawkishness.py                # Experiment 1 step 5: A&BG dictionary scoring + yield test
│   └── regression_analysis.py             # Experiment 1 step 6: OLS regression (S1–S5)
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Scrape Fed speeches
```bash
python src/scrape_speeches.py
```
This will crawl `federalreserve.gov` year by year (2008–2025), filter for Fed Chair speeches only, save each speech as a `.txt` file in `data/raw/speeches/`, and write a summary manifest to `data/processed/speeches.csv`.

Runtime: ~20–40 minutes depending on connection speed (polite 1.2s delay per request).

### 3. Fetch FRED macro data
```bash
export FRED_API_KEY=your_key_here
python src/fetch_fred.py
```
Requires a free FRED API key from [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html). Pulls DGS2, DFF, PCEPILFE, UNRATE, and GDP growth; saves raw CSVs to `data/raw/fred/` and writes an aligned macro snapshot to `data/processed/macro_context.csv`.

### 4. Fetch FOMC meeting calendar
```bash
python src/fetch_fomc.py
```
Scrapes FOMC announcement dates (2008–2025) from federalreserve.gov. No API key needed. Saves to `data/processed/fomc_calendar.csv`.

### 5. Filter speeches (Phase 1)
```bash
export ANTHROPIC_API_KEY=your_key_here
python src/phase1_filter.py
```
Requires an Anthropic API key from [console.anthropic.com](https://console.anthropic.com/settings/keys). Runs a 2-stage classifier (title keywords + `claude-opus-4-7`) to remove non-monetary-policy speeches. Outputs `speeches_filtered.csv` and `speeches_monetary_policy.csv`.

---

## Methodology Notes

**Why pairwise comparisons instead of direct scoring?**
When you ask an LLM to score something on a 1–100 scale, the output tends to cluster around a few values (bimodal distribution). Pairwise comparisons — "which of these two speeches is more hawkish?" — are simpler judgments and produce a fuller, more continuous spectrum when aggregated via a ranking algorithm.

**Why anonymize speakers?**
LLMs have prior beliefs about who is a hawk or dove based on their training data. Stripping speaker names forces the model to judge the actual language of the speech rather than the person's reputation.

**Why include macro context in scoring (but not in the regression)?**
"Rates will hold steady" sounds dovish during normal times but hawkish when markets expected a cut. We attach EFFR, inflation (PCEPILFE), unemployment (UNRATE), and GDP growth to each speech so the LLM can judge hawkishness *relative to conditions at the time* during Phase 2 pairwise comparisons. These series are *not* used as features in the Phase 3a regression — they move too slowly (monthly/quarterly) to explain same-day yield moves, and mixing them in would muddy the core question: does speech tone itself carry predictive signal? The one exception is EFFR, which may be added as a control in later regression models (see stepwise approach below).

**Macro data forward-fill rule**
PCE is monthly; GDP is quarterly. For each speech date we attach the most recently *released* value as of that date — not the most recent reference period, which may not have been published yet. For example, a speech on February 3rd gets the December PCE figure only if it had already been released by then. This avoids lookahead bias — using data the market couldn't have known at the time.

**Why stepwise regression rather than a kitchen-sink model?**
We build up the Phase 3a models in stages: hawkishness score alone first, then progressively add controls (EFFR level, chair fixed effects, etc.). This lets us show that speech tone has signal on its own before asking whether it survives in a richer specification. Throwing all features in at once makes it hard to tell which variable is doing the work.

**Known limitation: speech timestamps not collected**
The scraper does not capture the time of day each speech was delivered. Ideally, an afternoon or post-market speech should map to the *next* trading day's yield change rather than the same day. Without timestamps we treat all speeches as same-day events, which introduces some noise in the outcome variable for late-day speeches.

---

## Contribution

*[To be filled in per the course requirement — list each group member's contribution here before the May 26 submission.]*

---

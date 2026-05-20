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

## Experiment 0 — Simple Hawk/Dove Yield Test (First Deliverable)

Before running the full pipeline, we run a bare-bones sanity check: **do hawkish speeches push the 2-year yield up on the day they're delivered, and dovish speeches push it down?** No ML, no embeddings — just a lexicon score, a rank, and a direction call.

### Regimes

Each Fed Chair is treated as a separate regime and analyzed independently:

| Regime | Chair | Period |
|--------|-------|--------|
| A | Ben S. Bernanke | 2008–2014 |
| B | Janet L. Yellen | 2014–2018 |
| C | Jerome H. Powell | 2018–2025 |

### Scoring: Loughran-McDonald Lexicon

Each speech is scored using the [Loughran-McDonald Master Dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) — a financial-domain sentiment lexicon with explicit hawkish and dovish word lists (built for finance, not general text).

```
score = (hawkish_word_count − dovish_word_count) / total_words
```

Higher score = more hawkish language. Scores are computed per speech and are comparable within each regime.

### Ranking & Bucketing

Within each regime, speeches are ranked by score and divided into three equal buckets:

- **Bottom third → Dovish**
- **Middle third → Neutral**
- **Top third → Hawkish**

### Yield Test

For each speech, compute the same-day 2-year Treasury yield change:

```
yield_change = DGS2 (close, speech day) − DGS2 (close, prior trading day)
```

Prediction: Hawkish → yield rises. Dovish → yield falls. Neutral speeches are excluded from accuracy scoring (no directional prediction made).

Accuracy is reported per regime and overall.

### Data Notes

- Built on top of `speeches_filtered.csv` from Phase 1 (242 speeches with filter decisions)
- **6 evening/dinner speeches excluded** from the yield test — these were delivered after market close, so the same-day yield change cannot reflect the speech. Breakdown: 5 Bernanke, 1 Powell
- Yield data: `DGS2` and `DGS2_prev` from `macro_context.csv` (already collected in Phase 0, no lookahead bias)

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
│   │   ├── speeches/                    # one .txt file per speech (242 total)
│   │   └── fred/                        # raw FRED series as CSV
│   └── processed/
│       ├── speeches.csv                 # manifest: date, chair, title, word_count, url, filename
│       ├── macro_context.csv            # speeches + aligned macro snapshot (output of fetch_fred.py)
│       ├── fomc_calendar.csv            # FOMC announcement dates 2008–2025 (output of fetch_fomc.py)
│       ├── speeches_filtered.csv        # all 242 with filter decisions (output of phase1_filter.py)
│       └── speeches_monetary_policy.csv # 138 kept monetary-policy speeches (output of phase1_filter.py)
├── src/
│   ├── scrape_speeches.py               # Phase 0: Fed speech scraper
│   ├── fetch_fred.py                    # Phase 0: FRED macro data + macro_context.csv
│   ├── fetch_fomc.py                    # Phase 0: FOMC meeting calendar scraper
│   └── phase1_filter.py                 # Phase 1: 2-stage LLM relevance filter
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

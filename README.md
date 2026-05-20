# FedSpeak: Replicating FedLock with LLM Hawkishness Scoring

**BUSN 20800 Big Data Final Project — Spring 2026**

This project replicates and extends [FedLock](https://www.bloomberg.com/news/newsletters/2026-03-05/presenting-fedlock-a-new-way-to-measure-fedspeak-with-llms), a method for measuring the hawk-dove stance of Federal Reserve communications using large language models. We construct a corpus of Fed Chair speeches, score each speech on a hawkishness scale using LLM pairwise comparisons, and then use those scores alongside macro data to predict market reactions.

---

## Research Question

Can the tone of Federal Reserve Chair speeches — specifically how hawkish or dovish they sound — predict same-day market movements (e.g., Treasury yield changes, VIX)?

---

## Pipeline Overview

```
Phase 0  →  Phase 1  →  Phase 2  →  Phase 3  →  Phase 4
Collect     Clean &      Score        Predict      Validate &
data        filter       hawkishness  markets      write up
```

### Phase 0 — Data Ingestion (this phase)
- Scrape all speeches by Fed Chairs (Bernanke, Yellen, Powell) from `federalreserve.gov`, 2008–2025
- Pull macro time series from FRED: 2-year Treasury yield (DGS2), core PCE inflation (PCEPILFE), VIX (VIXCLS), and GDP growth
- Align each macro snapshot to the date of each speech

### Phase 1 — Preprocessing
- Filter out speeches unrelated to monetary policy using a 2-stage LLM classifier
- Anonymize speaker names so downstream LLM scoring isn't biased by known hawk/dove reputations
- Attach the macro context (inflation, unemployment, etc.) that existed on each speech date

### Phase 2 — Hawkishness Scoring (unsupervised)
- Run a pairwise tournament: present an LLM with two speeches side-by-side and ask which is more hawkish given the economic conditions at the time
- Aggregate pairwise outcomes using the TrueSkill ranking algorithm to produce a 0–100 hawkishness score per speech
- This unsupervised approach avoids the "bimodal scoring" problem that arises when asking an LLM to score directly on a 1–100 scale

### Phase 3 — Supervised Learning
- **Market reaction (3a):** Predict same-day yield or VIX changes using hawkishness score + macro features. Baseline: OLS. Also try LASSO, Random Forest, and TabPFN.
- **Score distillation (3b):** Predict hawkishness score from speech text features alone (TF-IDF, SBERT embeddings, fine-tuned FinBERT). This tests whether the score can be reproduced cheaply from text.

### Phase 4 — Validation
- Compare our reproduced scores to FedLock's published scores
- Check speaker-level rankings (e.g., do known hawks score high?)
- Out-of-sample metrics on held-out years

---

## Data Sources

| Source | Description | Series |
|--------|-------------|--------|
| [federalreserve.gov](https://www.federalreserve.gov/newsevents/speech/) | Fed Chair speeches, full text | 2008–2025 |
| [FRED](https://fred.stlouisfed.org/) | Macro time series | DGS2, PCEPILFE, VIXCLS, GDP growth |
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
│   │   ├── speeches/          # one .txt file per speech
│   │   └── fred/              # raw FRED series as CSV
│   └── processed/
│       └── speeches.csv       # manifest: date, chair, title, word_count, url, filename
├── src/
│   ├── scrape_speeches.py     # Phase 0: Fed speech scraper
│   ├── fetch_fred.py          # Phase 0: FRED macro data fetcher (coming next)
│   └── phase0_analysis.ipynb  # Summary stats + baseline regression
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

### 3. Fetch FRED macro data *(coming in Phase 0 Part 2)*
```bash
python src/fetch_fred.py
```
Requires a free FRED API key from [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html). Set it as an environment variable: `export FRED_API_KEY=your_key_here`.

---

## Methodology Notes

**Why pairwise comparisons instead of direct scoring?**
When you ask an LLM to score something on a 1–100 scale, the output tends to cluster around a few values (bimodal distribution). Pairwise comparisons — "which of these two speeches is more hawkish?" — are simpler judgments and produce a fuller, more continuous spectrum when aggregated via a ranking algorithm.

**Why anonymize speakers?**
LLMs have prior beliefs about who is a hawk or dove based on their training data. Stripping speaker names forces the model to judge the actual language of the speech rather than the person's reputation.

**Why include macro context?**
"Rates will hold steady" sounds dovish during normal times but hawkish when markets expected a cut. Providing the macro backdrop (inflation, unemployment, GDP) lets the LLM judge whether a speech is hawkish *relative to conditions at the time*.

---

## Contribution

*[To be filled in per the course requirement — list each group member's contribution here before the May 26 submission.]*

---

## Honor Code

AI tools were used to assist with data collection, feature engineering, and code writing, in accordance with course guidelines. All group members can explain each line of code. The written report was not AI-generated.
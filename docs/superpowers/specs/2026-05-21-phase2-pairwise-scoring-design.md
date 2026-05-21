# Phase 2 — LLM Pairwise Hawkishness Scoring

**Date:** 2026-05-21  
**Project:** FedSpeak (BUSN 20800 Final Project)  
**Status:** Approved, ready for implementation

---

## Goal

Replace the dictionary-based hawkishness scores from Experiment 1 with LLM-derived scores. The dictionary approach failed (37.6% directional accuracy, negative OOS R²) because it could not distinguish monetary policy language from financial regulation language. Phase 2 fixes this by having an LLM compare speeches directly, judging hawkishness from meaning and economic context rather than word counts.

---

## Architecture

One script: `src/pairwise_scoring.py`

```
sentences_{chair}.csv
        │
   [Stage 1: LLM sentence filter — Haiku, batches of 50]
        │
filtered_sentences_{chair}.csv
        │
   [Stage 2: Build speech representations — concat damped_lemmas per speech]
        │
   [Stage 3: Full round-robin pairwise comparisons — Sonnet 4.6, async parallel]
        │
pairwise_results_{chair}.csv
        │
   [Stage 4: TrueSkill aggregation → normalize to 0–100]
        │
speech_scores_phase2_{chair}.csv + speech_scores_phase2.csv (all 138)
```

Runs once per regime (Bernanke → Yellen → Powell). Each stage saves output files incrementally so the run is resumable if interrupted.

---

## Tournament Structure

- **Within-regime only.** Three separate tournaments: Bernanke (71 speeches), Yellen (22), Powell (45).
- **Full round-robin.** Every speech compared to every other in its regime exactly once.
  - Bernanke: 71×70/2 = 2,485 pairs
  - Yellen: 22×21/2 = 231 pairs
  - Powell: 45×44/2 = 990 pairs
  - Total: **3,706 comparisons**
- **Pair ordering randomized** for each pair to eliminate position bias (which speech is A vs B).
- **Estimated cost:** ~$12 total (Haiku filter ~$0.70 + Sonnet comparisons ~$11).
- **Estimated runtime:** ~45 minutes with 40 concurrent async requests.

---

## Stage 1 — Sentence Filter

**Purpose:** Remove sentences about financial regulation, bank supervision, consumer protection, and other non-monetary topics from the corpus before scoring. This directly addresses the root cause of Experiment 1's failures.

**Input:** `data/processed/sentences_{chair}.csv`

**Model:** `claude-haiku-4-5-20251001`

**Method:** Send batches of 50 sentences (using the original `sentence` text column — readable prose — for accurate classification) to Haiku with this prompt:

> *"Below are numbered sentence fragments from Federal Reserve speeches. Return a JSON array of the index numbers of any fragment that discusses: monetary policy stance, interest rates, inflation, employment/unemployment, economic growth, GDP, or the Fed's dual mandate. Exclude anything about bank supervision, financial regulation, capital requirements, consumer protection, community banking, or non-monetary topics. Return only the array, nothing else."*

**Output:** `data/processed/filtered_sentences_{chair}.csv` — same schema as input, only rows that passed. Retains `damped_lemmas` column for Stage 2.

---

## Stage 2 — Speech Representations

**Purpose:** Aggregate each speech's filtered sentences into a single text blob for the pairwise comparison.

**Method:** Group filtered sentences by `filename` (the speech identifier present in both `sentences_{chair}.csv` and `macro_context.csv`). For each speech, concatenate all `damped_lemmas` values into one space-separated string. No length cap — IDF damping already removes common boilerplate, leaving only differentiating terms (typically a few hundred words per speech). Macro context (DFF, PCE_YOY, UNRATE, GDP_GROWTH) joined from `macro_context.csv` on `filename`.

**No separate output file** — representations held in memory and passed directly to Stage 3.

---

## Stage 3 — Pairwise Comparisons

**Model:** `claude-sonnet-4-6`

**Parallelism:** `asyncio` with 40 concurrent requests via `anthropic.AsyncAnthropic`.

**Prompt:**

System:
> *"You are evaluating anonymized Federal Reserve speeches to determine which signals a more hawkish monetary policy stance. Hawkish = more concerned about inflation, more willing to raise or hold rates elevated. Dovish = more concerned about supporting employment and growth, more willing to cut or hold rates low. You will see key terms extracted from two speeches plus the economic conditions at the time each was delivered. Judge which speech signals a more hawkish stance given its economic context. Reply with only 'A' or 'B'."*

User:
```
SPEECH A
Economic conditions: [date] | Fed funds rate: X% | Core PCE: X% YoY | Unemployment: X% | GDP growth: X% annualized
Key terms: [damped_lemmas]

SPEECH B
Economic conditions: [date] | Fed funds rate: X% | Core PCE: X% YoY | Unemployment: X% | GDP growth: X% annualized
Key terms: [damped_lemmas]

Which speech signals a more hawkish stance given its economic context? Reply A or B.
```

**Error handling:** If response is not "A" or "B", retry once, then skip (treat as no result — excluded from TrueSkill update).

**Output:** `data/processed/pairwise_results_{chair}.csv`  
Columns: `filename_a`, `filename_b`, `winner`  
Flushed incrementally (append mode) — crash-safe.

---

## Stage 4 — TrueSkill Aggregation

**Library:** `trueskill` (Python package)

**Method:**
1. Initialize each speech with default TrueSkill rating (μ=25, σ=8.333).
2. For each row in `pairwise_results_{chair}.csv`, call `trueskill.rate_1vs1(winner_rating, loser_rating)` and update both ratings.
3. Final hawkishness score = μ for each speech.
4. Normalize μ values to 0–100 within each regime using min-max scaling.

**Output files:**

`data/processed/speech_scores_phase2_{chair}.csv`  
Columns: `filename`, `chair_key`, `date`, `title`, `trueskill_mu`, `trueskill_sigma`, `hawkishness_phase2`

`data/processed/speech_scores_phase2.csv`  
All 138 speeches stacked — combined file ready for Phase 3 regression.

---

## Dependencies

New packages to add to `requirements.txt`:
- `trueskill`
- `anthropic` (already present — ensure async client available)

---

## Output Summary

| File | Description |
|------|-------------|
| `filtered_sentences_{chair}.csv` | Sentences passing monetary policy filter (3 files) |
| `pairwise_results_{chair}.csv` | Raw pairwise outcomes (3 files) |
| `speech_scores_phase2_{chair}.csv` | Per-speech TrueSkill scores (3 files) |
| `speech_scores_phase2.csv` | Combined 138-speech file for Phase 3 |

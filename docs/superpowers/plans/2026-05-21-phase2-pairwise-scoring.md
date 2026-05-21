# Phase 2 LLM Pairwise Hawkishness Scoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/pairwise_scoring.py` — a four-stage pipeline that filters the sentence corpus to monetary-policy content, builds per-speech representations from `damped_lemmas`, runs a full round-robin pairwise tournament with Sonnet 4.6, and aggregates outcomes via TrueSkill into 0–100 hawkishness scores saved per regime and combined.

**Architecture:** Single script with pure helper functions (pair generation, speech representation building, TrueSkill aggregation) and async functions (Haiku sentence filter, Sonnet pairwise comparison). Each stage saves its output so the run is resumable on crash. Runs Bernanke → Yellen → Powell sequentially, with up to 40 concurrent API calls within each stage.

**Tech Stack:** `anthropic.AsyncAnthropic`, `trueskill`, `pandas`, `asyncio`, `pytest`, `pytest-asyncio`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/pairwise_scoring.py` | Create | Full Phase 2 pipeline (all four stages) |
| `tests/__init__.py` | Create | Makes tests/ a package |
| `tests/test_pairwise_scoring.py` | Create | Unit tests for all functions |
| `requirements.txt` | Modify | Add `trueskill`, `pytest`, `pytest-asyncio` |
| `pyproject.toml` | Create | pytest asyncio_mode = auto |

---

## Task 1: Dependencies, test scaffold, and pyproject.toml

**Files:**
- Modify: `requirements.txt`
- Create: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/test_pairwise_scoring.py`

- [ ] **Step 1: Add new dependencies to requirements.txt**

Append these three lines to `requirements.txt`:
```
trueskill>=0.4.5
pytest>=7.4.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 2: Install them**

```bash
pip install trueskill pytest pytest-asyncio
```
Expected: no errors; `Successfully installed trueskill-...` in output.

- [ ] **Step 3: Create pyproject.toml for pytest asyncio config**

Create `/Applications/fedspeak-project/pyproject.toml`:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 4: Create tests directory and empty init**

```bash
mkdir -p /Applications/fedspeak-project/tests && touch /Applications/fedspeak-project/tests/__init__.py
```

- [ ] **Step 5: Create test file with smoke test**

Create `tests/test_pairwise_scoring.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest


def test_smoke():
    assert True
```

- [ ] **Step 6: Run the smoke test**

```bash
cd /Applications/fedspeak-project && pytest tests/test_pairwise_scoring.py -v
```
Expected: `1 passed`

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pyproject.toml tests/__init__.py tests/test_pairwise_scoring.py
git commit -m "feat: test scaffold and dependencies for Phase 2 pairwise scoring"
```

---

## Task 2: Script skeleton and pair generation

**Files:**
- Create: `src/pairwise_scoring.py`
- Modify: `tests/test_pairwise_scoring.py`

- [ ] **Step 1: Write failing tests for generate_pairs**

Replace the contents of `tests/test_pairwise_scoring.py` with:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from pairwise_scoring import generate_pairs


def test_generate_pairs_count():
    """Round-robin of N speeches produces N*(N-1)/2 pairs."""
    pairs = generate_pairs(["a.txt", "b.txt", "c.txt", "d.txt"])
    assert len(pairs) == 6  # 4*3/2


def test_generate_pairs_each_pair_once():
    """Each unordered pair appears exactly once."""
    pairs = generate_pairs(["a.txt", "b.txt", "c.txt"])
    normalized = {frozenset(p) for p in pairs}
    assert len(normalized) == 3


def test_generate_pairs_no_self():
    """No speech is paired with itself."""
    pairs = generate_pairs(["a.txt", "b.txt", "c.txt"])
    assert all(a != b for a, b in pairs)


def test_generate_pairs_randomized():
    """A/B assignment is randomized — not all pairs keep the same order."""
    filenames = [f"{i}.txt" for i in range(20)]
    pairs = generate_pairs(filenames)
    # With 190 pairs, the probability all stay in original order is (0.5)^190 ≈ 0
    in_order = [(a, b) for a, b in pairs if a < b]
    assert len(in_order) < len(pairs)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Applications/fedspeak-project && pytest tests/test_pairwise_scoring.py -v
```
Expected: `ImportError: cannot import name 'generate_pairs' from 'pairwise_scoring'`

- [ ] **Step 3: Create pairwise_scoring.py with script header and generate_pairs**

Create `src/pairwise_scoring.py`:
```python
#!/usr/bin/env python3
"""
Phase 2 — LLM pairwise hawkishness scoring.

Four stages:
  1. Sentence filter     — Haiku classifies sentences as monetary-policy relevant
  2. Speech repr.        — aggregate filtered damped_lemmas per speech + attach macro
  3. Round-robin tournament — Sonnet compares every pair within each regime
  4. TrueSkill           — aggregate pair outcomes into 0-100 hawkishness scores

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python src/pairwise_scoring.py

Outputs (per regime + combined):
    data/processed/filtered_sentences_{chair}.csv
    data/processed/pairwise_results_{chair}.csv
    data/processed/speech_scores_phase2_{chair}.csv
    data/processed/speech_scores_phase2.csv
"""

import asyncio
import json
import logging
import os
import random
import sys
from itertools import combinations
from pathlib import Path

import anthropic
import pandas as pd
import trueskill

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

FILTER_MODEL   = "claude-haiku-4-5-20251001"
COMPARE_MODEL  = "claude-sonnet-4-6"
BATCH_SIZE     = 50
MAX_CONCURRENT = 40

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

FILTER_SYSTEM = """\
You are a classifier for Federal Reserve speech fragments.

Return a JSON array of the 0-based index numbers of any fragment that discusses:
monetary policy stance, interest rates, inflation, employment, unemployment,
economic growth, GDP, or the Fed's dual mandate (price stability and maximum employment).

Exclude fragments about: bank supervision, financial regulation, capital requirements,
consumer protection, community banking, stress tests, or other non-monetary topics.

Return ONLY a valid JSON array of integers. Example: [0, 2, 5]
If no fragments qualify, return: []"""

COMPARE_SYSTEM = """\
You are evaluating anonymized Federal Reserve speeches to determine which signals \
a more hawkish monetary policy stance.

Hawkish = more concerned about inflation, more willing to raise or hold rates elevated.
Dovish = more concerned about supporting employment and growth, more willing to cut \
or hold rates low.

You will see key terms extracted from two speeches plus the economic conditions at \
the time each was delivered. Judge which speech signals a more hawkish stance given \
its economic context.

Reply with only 'A' or 'B'. Nothing else."""


# ── Stage helpers ──────────────────────────────────────────────────────────────

def generate_pairs(filenames: list[str]) -> list[tuple[str, str]]:
    """
    Return all N*(N-1)/2 unordered pairs for a round-robin tournament.
    Each pair is randomly assigned A/B position to eliminate position bias.
    """
    pairs = list(combinations(filenames, 2))
    result = []
    for a, b in pairs:
        if random.random() < 0.5:
            a, b = b, a
        result.append((a, b))
    random.shuffle(result)
    return result
```

- [ ] **Step 4: Run tests**

```bash
cd /Applications/fedspeak-project && pytest tests/test_pairwise_scoring.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pairwise_scoring.py tests/test_pairwise_scoring.py
git commit -m "feat: Phase 2 script skeleton with pair generation"
```

---

## Task 3: Speech representation builder

**Files:**
- Modify: `src/pairwise_scoring.py`
- Modify: `tests/test_pairwise_scoring.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_pairwise_scoring.py`:
```python
from pairwise_scoring import build_speech_representations


def test_build_representations_keys():
    """Each filename in filtered_df appears as a key in the output."""
    filtered = pd.DataFrame({
        "filename":     ["a.txt", "a.txt", "b.txt"],
        "date":         ["2010-01-01", "2010-01-01", "2012-03-15"],
        "damped_lemmas":["tighten rate", "inflation high", "ease accommodate"],
        "title":        ["Speech A", "Speech A", "Speech B"],
    })
    macro = pd.DataFrame({
        "filename":   ["a.txt", "b.txt"],
        "DFF":        [0.25, 0.25],
        "PCE_YOY":    [1.5, 1.8],
        "UNRATE":     [9.5, 8.1],
        "GDP_GROWTH": [-2.1, 3.0],
    })
    reps = build_speech_representations(filtered, macro)
    assert set(reps.keys()) == {"a.txt", "b.txt"}


def test_build_representations_lemmas_concatenated():
    """damped_lemmas from multiple fragments are joined with a space."""
    filtered = pd.DataFrame({
        "filename":     ["a.txt", "a.txt"],
        "date":         ["2010-01-01", "2010-01-01"],
        "damped_lemmas":["tighten rate", "inflation high"],
        "title":        ["Speech A", "Speech A"],
    })
    macro = pd.DataFrame({
        "filename": ["a.txt"],
        "DFF": [0.25], "PCE_YOY": [1.5], "UNRATE": [9.5], "GDP_GROWTH": [-2.1],
    })
    reps = build_speech_representations(filtered, macro)
    assert reps["a.txt"]["damped_lemmas"] == "tighten rate inflation high"


def test_build_representations_macro_attached():
    """Macro fields (DFF, PCE_YOY, UNRATE, GDP_GROWTH) are joined from macro_context."""
    filtered = pd.DataFrame({
        "filename":     ["a.txt"],
        "date":         ["2022-06-15"],
        "damped_lemmas":["tighten"],
        "title":        ["Speech A"],
    })
    macro = pd.DataFrame({
        "filename": ["a.txt"],
        "DFF": [1.58], "PCE_YOY": [6.3], "UNRATE": [3.6], "GDP_GROWTH": [-1.6],
    })
    reps = build_speech_representations(filtered, macro)
    assert reps["a.txt"]["DFF"] == pytest.approx(1.58)
    assert reps["a.txt"]["PCE_YOY"] == pytest.approx(6.3)
    assert reps["a.txt"]["UNRATE"] == pytest.approx(3.6)
    assert reps["a.txt"]["GDP_GROWTH"] == pytest.approx(-1.6)


def test_build_representations_drops_missing_macro():
    """Speeches with no matching macro row are excluded from output."""
    filtered = pd.DataFrame({
        "filename":     ["a.txt", "b.txt"],
        "date":         ["2010-01-01", "2010-01-01"],
        "damped_lemmas":["tighten", "ease"],
        "title":        ["A", "B"],
    })
    macro = pd.DataFrame({
        "filename": ["a.txt"],
        "DFF": [0.25], "PCE_YOY": [1.5], "UNRATE": [9.5], "GDP_GROWTH": [-2.1],
    })
    reps = build_speech_representations(filtered, macro)
    assert "b.txt" not in reps
    assert "a.txt" in reps
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Applications/fedspeak-project && pytest tests/test_pairwise_scoring.py::test_build_representations_keys -v
```
Expected: `ImportError: cannot import name 'build_speech_representations'`

- [ ] **Step 3: Implement build_speech_representations**

Append to `src/pairwise_scoring.py` (after `generate_pairs`):
```python
def build_speech_representations(
    filtered_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Aggregate filtered sentences into one representation dict per speech.

    Groups by filename, joins damped_lemmas with spaces, then inner-joins
    macro context (DFF, PCE_YOY, UNRATE, GDP_GROWTH) from macro_df.
    Speeches with no macro row are excluded.

    Returns a dict keyed by filename. Each value has keys:
        damped_lemmas, date, title, DFF, PCE_YOY, UNRATE, GDP_GROWTH
    """
    agg = (
        filtered_df
        .groupby("filename", as_index=False)
        .agg(
            date         =("date",         "first"),
            title        =("title",        "first"),
            damped_lemmas=("damped_lemmas", lambda x: " ".join(x.dropna())),
        )
    )
    merged = agg.merge(
        macro_df[["filename", "DFF", "PCE_YOY", "UNRATE", "GDP_GROWTH"]],
        on="filename",
        how="inner",
    )
    return {
        row["filename"]: {
            "damped_lemmas": row["damped_lemmas"],
            "date":          str(row["date"]),
            "title":         row["title"],
            "DFF":           float(row["DFF"]),
            "PCE_YOY":       float(row["PCE_YOY"]),
            "UNRATE":        float(row["UNRATE"]),
            "GDP_GROWTH":    float(row["GDP_GROWTH"]),
        }
        for _, row in merged.iterrows()
    }
```

- [ ] **Step 4: Run tests**

```bash
cd /Applications/fedspeak-project && pytest tests/test_pairwise_scoring.py -v
```
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pairwise_scoring.py tests/test_pairwise_scoring.py
git commit -m "feat: Phase 2 speech representation builder"
```

---

## Task 4: TrueSkill aggregation

**Files:**
- Modify: `src/pairwise_scoring.py`
- Modify: `tests/test_pairwise_scoring.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_pairwise_scoring.py`:
```python
from pairwise_scoring import compute_trueskill


def test_trueskill_output_shape():
    """Returns one row per filename with required columns."""
    results = pd.DataFrame({
        "filename_a": ["a.txt", "b.txt", "a.txt"],
        "filename_b": ["b.txt", "c.txt", "c.txt"],
        "winner":     ["a.txt", "b.txt", "a.txt"],
    })
    out = compute_trueskill(results, ["a.txt", "b.txt", "c.txt"])
    assert set(out.columns) >= {"filename", "trueskill_mu", "trueskill_sigma", "hawkishness_phase2"}
    assert len(out) == 3


def test_trueskill_scores_in_range():
    """hawkishness_phase2 values are in [0, 100]."""
    results = pd.DataFrame({
        "filename_a": ["a.txt"] * 5 + ["b.txt"] * 4 + ["c.txt"] * 3,
        "filename_b": ["b.txt"] * 5 + ["c.txt"] * 4 + ["a.txt"] * 3,
        "winner":     ["a.txt"] * 5 + ["b.txt"] * 4 + ["a.txt"] * 3,
    })
    out = compute_trueskill(results, ["a.txt", "b.txt", "c.txt"])
    assert out["hawkishness_phase2"].min() >= 0.0
    assert out["hawkishness_phase2"].max() <= 100.0


def test_trueskill_consistent_winner_ranks_higher():
    """Speech that wins every comparison gets a higher score than the one that loses every time."""
    results = pd.DataFrame({
        "filename_a": ["a.txt"] * 10,
        "filename_b": ["b.txt"] * 10,
        "winner":     ["a.txt"] * 10,
    })
    out = compute_trueskill(results, ["a.txt", "b.txt"])
    score_a = out.loc[out["filename"] == "a.txt", "hawkishness_phase2"].iloc[0]
    score_b = out.loc[out["filename"] == "b.txt", "hawkishness_phase2"].iloc[0]
    assert score_a > score_b


def test_trueskill_skips_null_winners():
    """Rows where winner is None or NaN are skipped without crashing."""
    results = pd.DataFrame({
        "filename_a": ["a.txt", "b.txt"],
        "filename_b": ["b.txt", "c.txt"],
        "winner":     ["a.txt", None],
    })
    out = compute_trueskill(results, ["a.txt", "b.txt", "c.txt"])
    assert len(out) == 3
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Applications/fedspeak-project && pytest tests/test_pairwise_scoring.py::test_trueskill_output_shape -v
```
Expected: `ImportError: cannot import name 'compute_trueskill'`

- [ ] **Step 3: Implement compute_trueskill**

Append to `src/pairwise_scoring.py`:
```python
def compute_trueskill(
    results_df: pd.DataFrame,
    filenames: list[str],
) -> pd.DataFrame:
    """
    Feed pairwise outcomes into TrueSkill and return per-speech scores.

    draw_probability=0 because the LLM always picks A or B.
    Rows where winner is null/NaN are skipped.
    hawkishness_phase2 is min-max normalized to [0, 100] within this regime.

    Returns a DataFrame with columns:
        filename, trueskill_mu, trueskill_sigma, hawkishness_phase2
    """
    env = trueskill.TrueSkill(draw_probability=0.0)
    ratings = {f: env.create_rating() for f in filenames}

    for _, row in results_df.iterrows():
        winner = row["winner"]
        if winner is None or (isinstance(winner, float) and pd.isna(winner)):
            continue
        a, b = row["filename_a"], row["filename_b"]
        if winner == a:
            ratings[a], ratings[b] = env.rate_1vs1(ratings[a], ratings[b])
        else:
            ratings[b], ratings[a] = env.rate_1vs1(ratings[b], ratings[a])

    rows = [
        {"filename": f, "trueskill_mu": r.mu, "trueskill_sigma": r.sigma}
        for f, r in ratings.items()
    ]
    df = pd.DataFrame(rows)

    mu_min = df["trueskill_mu"].min()
    mu_max = df["trueskill_mu"].max()
    if mu_max > mu_min:
        df["hawkishness_phase2"] = (
            (df["trueskill_mu"] - mu_min) / (mu_max - mu_min) * 100
        )
    else:
        df["hawkishness_phase2"] = 50.0

    return df
```

- [ ] **Step 4: Run tests**

```bash
cd /Applications/fedspeak-project && pytest tests/test_pairwise_scoring.py -v
```
Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pairwise_scoring.py tests/test_pairwise_scoring.py
git commit -m "feat: Phase 2 TrueSkill aggregation with 0-100 normalization"
```

---

## Task 5: Async sentence filter (Haiku)

**Files:**
- Modify: `src/pairwise_scoring.py`
- Modify: `tests/test_pairwise_scoring.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_pairwise_scoring.py`:
```python
from pairwise_scoring import filter_sentences_batch, filter_all_sentences


async def test_filter_batch_returns_indices():
    """Returns 0-based indices of monetary-policy relevant sentences."""
    mock_client = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "[0, 2]"
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    sem = asyncio.Semaphore(1)
    result = await filter_sentences_batch(
        sentences=["tighten rates now", "bank supervision rules", "inflation above target"],
        client=mock_client,
        semaphore=sem,
    )
    assert result == [0, 2]


async def test_filter_batch_bad_json_returns_all():
    """On JSON parse failure, conservatively includes all sentences."""
    mock_client = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "not valid json"
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    sem = asyncio.Semaphore(1)
    result = await filter_sentences_batch(["a", "b", "c"], mock_client, sem)
    assert result == [0, 1, 2]


async def test_filter_all_sentences_keeps_correct_rows():
    """filter_all_sentences returns only rows whose index was returned by the classifier."""
    df = pd.DataFrame({
        "filename":     ["a.txt"] * 4,
        "date":         ["2010-01-01"] * 4,
        "text":         ["monetary policy rates", "bank regulation rules",
                         "inflation above target", "stress test capital"],
        "damped_lemmas":["monetary policy rates", "bank regulation rules",
                         "inflation above target", "stress test capital"],
        "title":        ["Speech A"] * 4,
    })
    mock_client = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "[0, 2]"
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    sem = asyncio.Semaphore(1)
    result = await filter_all_sentences(df, mock_client, sem)
    assert len(result) == 2
    assert list(result["text"]) == ["monetary policy rates", "inflation above target"]
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Applications/fedspeak-project && pytest tests/test_pairwise_scoring.py::test_filter_batch_returns_indices -v
```
Expected: `ImportError: cannot import name 'filter_sentences_batch'`

- [ ] **Step 3: Implement filter_sentences_batch and filter_all_sentences**

Append to `src/pairwise_scoring.py`:
```python
async def filter_sentences_batch(
    sentences: list[str],
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
) -> list[int]:
    """
    Classify a batch of sentences. Returns 0-based indices of those that
    discuss monetary policy, inflation, employment, or economic growth.
    Falls back to including all indices if the API fails or returns bad JSON.
    """
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.messages.create(
                    model=FILTER_MODEL,
                    max_tokens=256,
                    system=FILTER_SYSTEM,
                    messages=[{"role": "user", "content": numbered}],
                )
                indices = json.loads(resp.content[0].text.strip())
                return [i for i in indices if 0 <= i < len(sentences)]
            except (json.JSONDecodeError, ValueError):
                if attempt == 2:
                    return list(range(len(sentences)))
                await asyncio.sleep(2 ** attempt)
            except anthropic.APIError:
                if attempt == 2:
                    return list(range(len(sentences)))
                await asyncio.sleep(2 ** attempt)
    return list(range(len(sentences)))


async def filter_all_sentences(
    df: pd.DataFrame,
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
) -> pd.DataFrame:
    """
    Run the sentence filter across the full DataFrame in batches of BATCH_SIZE.
    Dispatches all batches concurrently (semaphore limits to MAX_CONCURRENT calls).
    Returns a filtered DataFrame with only monetary-policy relevant rows.
    """
    sentences = df["text"].tolist()
    batches = [sentences[i:i + BATCH_SIZE] for i in range(0, len(sentences), BATCH_SIZE)]
    batch_starts = list(range(0, len(sentences), BATCH_SIZE))

    tasks = [filter_sentences_batch(batch, client, semaphore) for batch in batches]
    batch_results = await asyncio.gather(*tasks)

    keep_positions: list[int] = []
    for start, indices in zip(batch_starts, batch_results):
        keep_positions.extend(start + i for i in indices)

    return df.iloc[sorted(keep_positions)].reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

```bash
cd /Applications/fedspeak-project && pytest tests/test_pairwise_scoring.py -v
```
Expected: `15 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pairwise_scoring.py tests/test_pairwise_scoring.py
git commit -m "feat: Phase 2 async sentence filter with Haiku"
```

---

## Task 6: Pairwise comparison and tournament runner

**Files:**
- Modify: `src/pairwise_scoring.py`
- Modify: `tests/test_pairwise_scoring.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_pairwise_scoring.py`:
```python
from pairwise_scoring import compare_pair

_REPR_A = {
    "damped_lemmas": "tighten rate inflation elevated",
    "date": "2022-06-15",
    "DFF": 1.58, "PCE_YOY": 6.3, "UNRATE": 3.6, "GDP_GROWTH": -1.6,
}
_REPR_B = {
    "damped_lemmas": "ease accommodate patient gradual",
    "date": "2019-07-31",
    "DFF": 2.40, "PCE_YOY": 1.6, "UNRATE": 3.7, "GDP_GROWTH": 2.1,
}


async def test_compare_pair_returns_a():
    """Returns filename_a when LLM responds 'A'."""
    mock_client = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "A"
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    sem = asyncio.Semaphore(1)
    fa, fb, winner = await compare_pair("a.txt", "b.txt", _REPR_A, _REPR_B, mock_client, sem)
    assert fa == "a.txt"
    assert fb == "b.txt"
    assert winner == "a.txt"


async def test_compare_pair_returns_b():
    """Returns filename_b when LLM responds 'B'."""
    mock_client = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "B"
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    sem = asyncio.Semaphore(1)
    fa, fb, winner = await compare_pair("a.txt", "b.txt", _REPR_A, _REPR_B, mock_client, sem)
    assert winner == "b.txt"


async def test_compare_pair_none_on_ambiguous():
    """Returns None as winner after two ambiguous responses."""
    mock_client = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "neither"
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    sem = asyncio.Semaphore(1)
    fa, fb, winner = await compare_pair("a.txt", "b.txt", _REPR_A, _REPR_B, mock_client, sem)
    assert winner is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Applications/fedspeak-project && pytest tests/test_pairwise_scoring.py::test_compare_pair_returns_a -v
```
Expected: `ImportError: cannot import name 'compare_pair'`

- [ ] **Step 3: Implement compare_pair and run_tournament**

Append to `src/pairwise_scoring.py`:
```python
def _format_speech_block(label: str, rep: dict) -> str:
    return (
        f"SPEECH {label}\n"
        f"Economic conditions: {rep['date']} | "
        f"Fed funds rate: {rep['DFF']:.2f}% | "
        f"Core PCE: {rep['PCE_YOY']:.1f}% YoY | "
        f"Unemployment: {rep['UNRATE']:.1f}% | "
        f"GDP growth: {rep['GDP_GROWTH']:.1f}% annualized\n"
        f"Key terms: {rep['damped_lemmas']}"
    )


async def compare_pair(
    filename_a: str,
    filename_b: str,
    repr_a: dict,
    repr_b: dict,
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str, str | None]:
    """
    Compare two speeches. Returns (filename_a, filename_b, winner_filename).
    winner_filename is None if the LLM response is not 'A' or 'B' after two attempts.
    """
    user_msg = (
        _format_speech_block("A", repr_a)
        + "\n\n"
        + _format_speech_block("B", repr_b)
        + "\n\nWhich speech signals a more hawkish stance given its economic context? Reply A or B."
    )
    async with semaphore:
        for attempt in range(2):
            try:
                resp = await client.messages.create(
                    model=COMPARE_MODEL,
                    max_tokens=8,
                    system=COMPARE_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                choice = resp.content[0].text.strip().upper()
                if choice == "A":
                    return filename_a, filename_b, filename_a
                if choice == "B":
                    return filename_a, filename_b, filename_b
            except anthropic.APIError:
                if attempt == 0:
                    await asyncio.sleep(2)
    return filename_a, filename_b, None


async def run_tournament(
    pairs: list[tuple[str, str]],
    representations: dict[str, dict],
    client: anthropic.AsyncAnthropic,
    results_path: Path,
    semaphore: asyncio.Semaphore,
) -> pd.DataFrame:
    """
    Run all pairwise comparisons concurrently (semaphore limits to MAX_CONCURRENT).
    Saves results every 200 comparisons for crash-safety.
    Resumes from existing results file if present — skips already-done pairs.
    """
    results: list[dict] = []

    if results_path.exists():
        existing = pd.read_csv(results_path)
        done = {
            frozenset([row["filename_a"], row["filename_b"]])
            for _, row in existing.iterrows()
        }
        results = existing.to_dict("records")
        pairs = [(a, b) for a, b in pairs if frozenset([a, b]) not in done]
        log.info(f"  Resuming: {len(results)} done, {len(pairs)} remaining")

    tasks = [
        compare_pair(a, b, representations[a], representations[b], client, semaphore)
        for a, b in pairs
    ]

    completed = 0
    for coro in asyncio.as_completed(tasks):
        fa, fb, winner = await coro
        results.append({"filename_a": fa, "filename_b": fb, "winner": winner})
        completed += 1
        if completed % 200 == 0:
            pd.DataFrame(results).to_csv(results_path, index=False)
            log.info(f"  Progress: {completed}/{len(pairs)} comparisons")

    pd.DataFrame(results).to_csv(results_path, index=False)
    return pd.DataFrame(results)
```

- [ ] **Step 4: Run all tests**

```bash
cd /Applications/fedspeak-project && pytest tests/test_pairwise_scoring.py -v
```
Expected: `18 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pairwise_scoring.py tests/test_pairwise_scoring.py
git commit -m "feat: Phase 2 pairwise comparison and crash-safe tournament runner"
```

---

## Task 7: Main pipeline

**Files:**
- Modify: `src/pairwise_scoring.py`

- [ ] **Step 1: Implement main()**

Append to `src/pairwise_scoring.py`:
```python
async def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "Error: ANTHROPIC_API_KEY not set.\n"
            "Run: export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client    = anthropic.AsyncAnthropic(api_key=api_key)
    macro     = pd.read_csv(PROCESSED / "macro_context.csv")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    all_scores: list[pd.DataFrame] = []

    for chair in ["bernanke", "yellen", "powell"]:
        log.info(f"\n{'='*60}")
        log.info(f"Regime: {chair}")

        # ── Stage 1: Sentence filter ──────────────────────────────────
        filtered_path = PROCESSED / f"filtered_sentences_{chair}.csv"
        if filtered_path.exists():
            log.info(f"  Stage 1: Loading cached → {filtered_path.name}")
            filtered = pd.read_csv(filtered_path)
        else:
            log.info(f"  Stage 1: Filtering sentences for monetary policy content...")
            sentences = pd.read_csv(PROCESSED / f"sentences_{chair}.csv")
            log.info(f"    Input: {len(sentences)} sentences")
            filtered = await filter_all_sentences(sentences, client, semaphore)
            filtered.to_csv(filtered_path, index=False)
            pct = len(filtered) / len(sentences) * 100
            log.info(f"    Kept: {len(filtered)} ({pct:.0f}%) → saved {filtered_path.name}")

        # ── Stage 2: Build speech representations ────────────────────
        log.info(f"  Stage 2: Building speech representations...")
        representations = build_speech_representations(filtered, macro)
        filenames = list(representations.keys())
        log.info(f"    {len(filenames)} speeches")

        # ── Stage 3: Round-robin tournament ──────────────────────────
        results_path = PROCESSED / f"pairwise_results_{chair}.csv"
        pairs = generate_pairs(filenames)
        log.info(f"  Stage 3: Tournament — {len(pairs)} pairs, {MAX_CONCURRENT} concurrent...")
        results_df = await run_tournament(pairs, representations, client, results_path, semaphore)
        null_n = results_df["winner"].isna().sum()
        log.info(f"    Done: {len(results_df)} comparisons, {null_n} inconclusive")

        # ── Stage 4: TrueSkill ────────────────────────────────────────
        log.info(f"  Stage 4: TrueSkill aggregation...")
        scores = compute_trueskill(results_df, filenames)

        meta = macro[macro["filename"].isin(filenames)][
            ["filename", "chair_key", "date", "title"]
        ].drop_duplicates("filename")
        scores = scores.merge(meta, on="filename", how="left")
        scores = scores.sort_values("hawkishness_phase2", ascending=False).reset_index(drop=True)

        score_path = PROCESSED / f"speech_scores_phase2_{chair}.csv"
        scores.to_csv(score_path, index=False)
        log.info(
            f"    Score range: [{scores['hawkishness_phase2'].min():.1f}, "
            f"{scores['hawkishness_phase2'].max():.1f}] → saved {score_path.name}"
        )

        log.info(f"    Top 3 hawkish:")
        for _, r in scores.head(3).iterrows():
            log.info(f"      {str(r['date'])[:10]}  {r['title'][:55]:<55}  "
                     f"score={r['hawkishness_phase2']:.1f}")
        log.info(f"    Top 3 dovish:")
        for _, r in scores.tail(3).iterrows():
            log.info(f"      {str(r['date'])[:10]}  {r['title'][:55]:<55}  "
                     f"score={r['hawkishness_phase2']:.1f}")

        all_scores.append(scores)

    combined = pd.concat(all_scores, ignore_index=True)
    combined_path = PROCESSED / "speech_scores_phase2.csv"
    combined.to_csv(combined_path, index=False)
    log.info(f"\nAll regimes complete. {combined_path.name}: {len(combined)} speeches")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Verify the script imports cleanly**

```bash
cd /Applications/fedspeak-project && python -c "import sys; sys.path.insert(0, 'src'); import pairwise_scoring; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Run full test suite**

```bash
cd /Applications/fedspeak-project && pytest tests/test_pairwise_scoring.py -v
```
Expected: `18 passed`

- [ ] **Step 4: Commit**

```bash
git add src/pairwise_scoring.py
git commit -m "feat: Phase 2 main pipeline — orchestrates all four stages per regime"
```

---

## Running the Pipeline

Once all tasks are complete:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python src/pairwise_scoring.py
```

Expected runtime: ~45 minutes. Expected cost: ~$12.

The pipeline is crash-safe: if interrupted, delete no files and re-run — it will skip completed stages (cached `filtered_sentences_{chair}.csv`) and resume mid-tournament (skipping pairs already in `pairwise_results_{chair}.csv`).

Output files produced:
- `data/processed/filtered_sentences_{chair}.csv` (3 files)
- `data/processed/pairwise_results_{chair}.csv` (3 files)
- `data/processed/speech_scores_phase2_{chair}.csv` (3 files)
- `data/processed/speech_scores_phase2.csv` (combined, 138 speeches, ready for Phase 3)

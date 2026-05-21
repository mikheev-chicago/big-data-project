import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from pairwise_scoring import generate_pairs, build_speech_representations


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

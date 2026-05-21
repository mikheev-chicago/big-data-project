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

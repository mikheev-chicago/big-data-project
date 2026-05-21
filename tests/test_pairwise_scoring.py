import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest


def test_smoke():
    assert True

"""Shared fixtures. Adds ``src`` to the path so the tests run from a clone."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swarmplan import datasets  # noqa: E402
from swarmplan.graph import GridMap  # noqa: E402

#: Skip marker for tests that need the movingai benchmark files.
requires_data = pytest.mark.skipif(
    not datasets.available(),
    reason="benchmark data not fetched; run tools/fetch_benchmarks.py --all",
)


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


def _has_scipy() -> bool:
    try:
        import scipy  # noqa: F401

        return True
    except ImportError:
        return False


#: Skip marker for the figure tests.
requires_matplotlib = pytest.mark.skipif(
    not _has_matplotlib(), reason="matplotlib not installed"
)
#: Skip marker for the SciPy cross-check of the Hungarian implementation.
requires_scipy = pytest.mark.skipif(not _has_scipy(), reason="scipy not installed")


@pytest.fixture
def open_grid() -> GridMap:
    """A 5x5 grid with no obstacles."""
    return GridMap.from_rows(["....."] * 5, name="open5")


@pytest.fixture
def wall_grid() -> GridMap:
    """A 3x5 grid with a wall across the middle, leaving two ways round."""
    return GridMap.from_rows([".....", ".@@@.", "....."], name="wall")


@pytest.fixture
def alcove_grid() -> GridMap:
    """The three-cell corridor with one alcove that defeats prioritised planning."""
    return GridMap.from_rows(["...", "@.@"], name="alcove")


@pytest.fixture
def bay_grid() -> GridMap:
    """A corridor with a passing bay in the middle."""
    return GridMap.from_rows([".....", "..@..", "....."], name="bay")

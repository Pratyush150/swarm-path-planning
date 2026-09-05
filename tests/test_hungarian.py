"""Our Hungarian algorithm against SciPy, and the bottleneck variant."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_scipy
from swarmplan.assignment.hungarian import (
    assignment_cost,
    bottleneck_assignment,
    hungarian,
    solve_assignment,
)


def test_trivial_and_known_assignments():
    cost = np.array([[1.0, 5.0], [4.0, 2.0]])
    r, c = hungarian(cost)
    assert list(r) == [0, 1]
    assert list(c) == [0, 1]
    assert assignment_cost(cost, r, c) == 3.0
    swapped = np.array([[5.0, 1.0], [2.0, 4.0]])
    r, c = hungarian(swapped)
    assert list(c) == [1, 0]


@requires_scipy
def test_matches_scipy_on_random_matrices():
    """Same total cost as scipy.optimize.linear_sum_assignment, 200 times over.

    Assignments may differ where costs tie; both are then optimal, so the total
    is what has to match.
    """
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(20260904)
    for _ in range(200):
        n = int(rng.integers(1, 9))
        m = int(rng.integers(n, n + 4))
        cost = rng.random((n, m)) * 10.0
        ours = assignment_cost(cost, *hungarian(cost))
        theirs = assignment_cost(cost, *linear_sum_assignment(cost))
        assert ours == pytest.approx(theirs)


@requires_scipy
def test_matches_scipy_on_integer_and_tall_matrices():
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(7)
    for _ in range(50):
        n = int(rng.integers(2, 7))
        m = int(rng.integers(2, 7))
        cost = rng.integers(0, 20, (n, m)).astype(float)
        ours = assignment_cost(cost, *hungarian(cost))
        theirs = assignment_cost(cost, *linear_sum_assignment(cost))
        assert ours == pytest.approx(theirs)


def test_every_row_and_column_is_used_at_most_once():
    rng = np.random.default_rng(3)
    cost = rng.random((6, 9))
    r, c = hungarian(cost)
    assert len(set(r.tolist())) == 6
    assert len(set(c.tolist())) == 6


def test_empty_and_invalid_inputs():
    r, c = hungarian(np.zeros((0, 0)))
    assert len(r) == len(c) == 0
    with pytest.raises(ValueError):
        hungarian(np.zeros(4))
    with pytest.raises(ValueError):
        hungarian(np.array([[np.inf, 1.0], [1.0, 1.0]]))


def test_solve_assignment_falls_back_to_our_implementation():
    cost = np.array([[3.0, 1.0], [1.0, 3.0]])
    r1, c1 = solve_assignment(cost, use_scipy=False)
    r2, c2 = solve_assignment(cost, use_scipy=True)
    assert assignment_cost(cost, r1, c1) == assignment_cost(cost, r2, c2) == 2.0


def test_bottleneck_minimises_the_worst_edge_not_the_sum():
    """Minimum total cost and minimum worst-case cost are different objectives."""
    cost = np.array([[1.0, 9.0], [1.0, 8.0]])
    r_sum, c_sum = hungarian(cost)
    r_b, c_b, worst = bottleneck_assignment(cost)
    assert worst == 8.0
    assert cost[r_b, c_b].max() <= cost[r_sum, c_sum].max()


def test_bottleneck_on_a_square_matrix_is_a_permutation():
    rng = np.random.default_rng(11)
    cost = rng.random((5, 5))
    r, c, worst = bottleneck_assignment(cost)
    assert sorted(c.tolist()) == list(range(5))
    assert cost[r, c].max() == pytest.approx(worst)


def test_bottleneck_rejects_more_rows_than_columns():
    with pytest.raises(ValueError):
        bottleneck_assignment(np.zeros((4, 2)))

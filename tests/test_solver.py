"""Regression tests pinning the exact retrograde-analysis result.

The game of Chopsticks (as implemented here) is a draw with optimal play. These
tests lock in every headline number so a refactor cannot silently change it.
"""

from typing import Dict

import pytest

from chopsticks.env.state import ChopsticksState, Turn
from chopsticks.solver import (
    Solution,
    all_canonical_states,
    children,
    find_missing_states,
    graph_stats,
    solve,
)


@pytest.fixture(scope="module")
def solution() -> Solution:
    """Solve from the canonical opening once for the whole module."""

    return solve()


def test_start_position_is_a_draw(solution: Solution) -> None:
    assert solution.values[ChopsticksState(1, 1, 1, 1, Turn.P1)] == 0


def test_reachable_state_count(solution: Solution) -> None:
    assert len(solution.graph) == 406
    assert len(solution.values) == 406


def test_win_loss_draw_split(solution: Solution) -> None:
    counts = list(solution.values.values())
    assert (counts.count(1), counts.count(-1), counts.count(0)) == (156, 156, 94)
    assert 156 + 156 + 94 == len(solution.values)


def test_unreachable_canonical_states(solution: Solution) -> None:
    assert len(all_canonical_states()) == 450
    assert len(find_missing_states(solution.values)) == 450 - 406


def test_opening_self_move_loses(solution: Solution) -> None:
    # From the start, adding a hand to your own hand makes (1, 2) and loses:
    # Player 2 has a forced win.
    assert solution.values[ChopsticksState(1, 2, 1, 1, Turn.P2)] == -1


def test_graph_stats_summary(solution: Solution) -> None:
    stats = graph_stats(solution)
    assert stats["reachable_states"] == 406
    assert stats["p1_wins"] == 156
    assert stats["p2_wins"] == 156
    assert stats["draws"] == 94
    assert stats["missing_states"] == 44
    assert stats["start_value"] == 0
    assert stats["start_outcome"] == "draw"


def _fixpoint_values() -> Dict[ChopsticksState, int]:
    """Solve the game independently by value iteration over every canonical state.

    This is structurally different from :func:`solve` (a Bellman-style fixpoint
    rather than a retrograde BFS from terminals), so agreement is strong evidence
    the retrograde pass is correct. Values are from Player 1's perspective:
    ``+1`` P1 wins, ``-1`` P2 wins, ``0`` draw / unresolved cycle.
    """

    states = all_canonical_states()
    kids: Dict[ChopsticksState, list[ChopsticksState]] = {}
    values: Dict[ChopsticksState, int] = {}
    for state in states:
        if state.is_terminal():
            winner = state.winner()
            values[state] = 1 if winner == Turn.P1 else -1 if winner == Turn.P2 else 0
            kids[state] = []
        else:
            values[state] = 0
            kids[state] = [child for child, _action in children(state)]

    changed = True
    while changed:
        changed = False
        for state in states:
            if state.is_terminal():
                continue
            child_values = [values[child] for child in kids[state]]
            if state.turn == Turn.P1:
                new = (
                    1
                    if 1 in child_values
                    else -1
                    if all(value == -1 for value in child_values)
                    else 0
                )
            else:
                new = (
                    -1
                    if -1 in child_values
                    else 1
                    if all(value == 1 for value in child_values)
                    else 0
                )
            if new != values[state]:
                values[state] = new
                changed = True
    return values


def test_retrograde_matches_independent_fixpoint(solution: Solution) -> None:
    fixpoint = _fixpoint_values()
    mismatches = {
        state: (value, fixpoint[state])
        for state, value in solution.values.items()
        if fixpoint[state] != value
    }
    assert not mismatches

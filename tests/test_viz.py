"""Tests for the solved-graph visualisations.

These need the ``viz`` extra (``pydot``); the module is skipped without it.
"""

from typing import Dict

import pytest

from chopsticks.env.state import ChopsticksState, Turn
from chopsticks.solver import optimal_children, solve
from chopsticks.solver.graph import CANONICAL_START

pytest.importorskip("pydot")

from chopsticks.solver import viz  # noqa: E402


@pytest.fixture(scope="module")
def solution():
    return solve()


@pytest.fixture(scope="module")
def by_label(solution) -> Dict[str, ChopsticksState]:
    """Map every reachable position's ``str`` back to the position."""

    return {str(state): state for state in solution.graph}


def _edges(dot_graph) -> list[tuple[str, str, str]]:
    """Return ``(source, destination, action-label)`` triples, unquoted."""

    triples = []
    for edge in dot_graph.get_edges():
        src = edge.get_source().strip('"')
        dst = edge.get_destination().strip('"')
        label = (edge.get("label") or "").strip('"')
        triples.append((src, dst, label))
    return triples


@pytest.mark.parametrize("player", [Turn.P1, Turn.P2])
def test_strategy_graph_prunes_only_the_acting_player(player, solution, by_label):
    graph, values = solution
    dot_graph = viz.strategy_graph_dot(solution, player)

    drawn: Dict[str, set[str]] = {}
    for src, _dst, label in _edges(dot_graph):
        drawn.setdefault(src, set()).add(label)

    assert str(CANONICAL_START) in {s for s, _d, _l in _edges(dot_graph)}

    for src, actions in drawn.items():
        state = by_label[src]
        all_actions = {str(a) for _c, a, _e in graph[state]}
        optimal = {str(a) for _c, a, _e in optimal_children(graph, values, state)}
        if state.turn == player:
            # only value-preserving moves survive for the player we solved for
            assert actions <= optimal
        else:
            # every legal reply of the opponent is kept
            assert actions == all_actions


def test_p1_and_p2_helpers_match_the_general_builder(solution):
    assert _edges(viz.p1_winning_dot(solution)) == _edges(
        viz.strategy_graph_dot(solution, Turn.P1)
    )
    assert _edges(viz.p2_winning_dot(solution)) == _edges(
        viz.strategy_graph_dot(solution, Turn.P2)
    )

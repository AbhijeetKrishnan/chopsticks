"""Tests for the solved-graph visualisations.

These need the ``viz`` extra (``pydot``); the module is skipped without it.
"""

import shutil
from typing import Dict

import pytest

from chopsticks.env.state import ChopsticksState, Turn
from chopsticks.solver import optimal_children, solve
from chopsticks.solver.graph import CANONICAL_START

pytest.importorskip("pydot")

from chopsticks.solver import viz  # noqa: E402

START = str(CANONICAL_START)
# the sole position one ply after the canonical opening (P2 to move)
AFTER_START = "p1=(1, 1), p2=(1, 2), t=P2, r=False"


@pytest.fixture(scope="module")
def solution():
    return solve()


@pytest.fixture(scope="module")
def by_label(solution) -> Dict[str, ChopsticksState]:
    """Map every reachable position's ``str`` back to the position."""

    return {str(state): state for state in solution.graph}


def _text(value) -> str:
    """Coerce a pydot attribute value / edge endpoint / name to a bare string."""

    return "" if value is None else str(value).strip('"')


def _edges(dot_graph) -> list[tuple[str, str, str]]:
    """Return ``(source, destination, action-code)`` triples, one per code.

    Parallel moves to the same position share one edge whose label joins their
    codes with ``/``; this splits them back out so the set-based assertions
    below still see every individual action.
    """

    triples = []
    for edge in dot_graph.get_edges():
        src = _text(edge.get_source())
        dst = _text(edge.get_destination())
        for code in _text(edge.get("label")).split("/"):
            code = code.strip()
            if code:
                triples.append((src, dst, code))
    return triples


def _node(dot_graph, name: str):
    for node in dot_graph.get_nodes():
        if _text(node.get_name()) == name:
            return node
    raise AssertionError(f"node {name!r} not in graph")


def _clusters(dot_graph) -> list[str]:
    return [_text(sg.get_name()) for sg in dot_graph.get_subgraphs()]


@pytest.mark.parametrize("player", [Turn.P1, Turn.P2])
def test_strategy_graph_prunes_only_the_acting_player(player, solution, by_label):
    graph, values = solution
    dot_graph = viz.strategy_graph_dot(solution, player, legend=False)

    drawn: Dict[str, set[str]] = {}
    for src, _dst, code in _edges(dot_graph):
        drawn.setdefault(src, set()).add(code)

    assert START in {s for s, _d, _c in _edges(dot_graph)}

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
    assert _edges(viz.p1_winning_dot(solution, legend=False)) == _edges(
        viz.strategy_graph_dot(solution, Turn.P1, legend=False)
    )
    assert _edges(viz.p2_winning_dot(solution, legend=False)) == _edges(
        viz.strategy_graph_dot(solution, Turn.P2, legend=False)
    )


def test_graphs_are_directed(solution):
    assert viz.optimal_graph_dot(solution, legend=False).get_type() == "digraph"
    assert (
        viz.strategy_graph_dot(solution, Turn.P1, legend=False).get_type() == "digraph"
    )
    assert viz.full_state_graph_dot().get_type() == "digraph"


def test_parallel_moves_to_one_child_are_merged(solution):
    # from the (1,1)/(1,1) opening every transfer-to-opponent lands on the same
    # position, so the four codes collapse onto a single edge.
    dot_graph = viz.strategy_graph_dot(solution, Turn.P1, legend=False)
    opening = [e for e in dot_graph.get_edges() if _text(e.get_source()) == START]
    assert len(opening) == 1
    codes = _text(opening[0].get("label")).split("/")
    assert sorted(c.strip() for c in codes) == ["M2M", "M2m", "m2M", "m2m"]


def test_start_node_is_emphasised(solution):
    node = _node(viz.optimal_graph_dot(solution, legend=False), START)
    assert _text(node.get("penwidth")) == "3"
    assert _text(node.get("color")).lower() == viz.START_BORDER.lower()
    assert "START" in _text(node.get("label"))


def test_primary_and_reply_edges_are_styled_by_role(solution):
    dot_graph = viz.strategy_graph_dot(solution, Turn.P1, legend=False)
    by_src: dict[str, list] = {}
    for edge in dot_graph.get_edges():
        by_src.setdefault(_text(edge.get_source()), []).append(edge)

    # P1 to move at the start -> its move is a thick blue PRIMARY edge
    primary = by_src[START][0]
    assert _text(primary.get("color")) == viz.PRIMARY_COLOR_BY_MOVER[Turn.P1]
    assert _text(primary.get("penwidth")) == "2.0"

    # P2 to move one ply later -> every reply is a thin grey edge
    for edge in by_src[AFTER_START]:
        assert _text(edge.get("color")) == "#9aa5b1"
        assert _text(edge.get("fontsize")) == "8"


def test_repetition_edges_are_dashed_and_unconstrained(solution):
    dashed = [
        e
        for e in viz.optimal_graph_dot(solution, legend=False).get_edges()
        if _text(e.get("style")) == "dashed"
    ]
    assert dashed
    for edge in dashed:
        assert _text(edge.get("constraint")) == "false"
        assert _text(edge.get("color")) == "#c0392b"


def test_legend_is_present_by_default_and_optional(solution):
    with_legend = viz.optimal_graph_dot(solution)
    without = viz.optimal_graph_dot(solution, legend=False)

    legend = [c for c in _clusters(with_legend) if c.startswith("cluster_legend")]
    assert legend
    assert not [c for c in _clusters(without) if c.startswith("cluster_legend")]

    cluster = next(
        sg
        for sg in with_legend.get_subgraphs()
        if _text(sg.get_name()).startswith("cluster_legend")
    )
    assert cluster.get_nodes()
    assert all(_text(n.get_name()).startswith("legend_") for n in cluster.get_nodes())


@pytest.mark.skipif(shutil.which("dot") is None, reason="needs the Graphviz dot binary")
def test_render_writes_svg(solution, tmp_path):
    out = tmp_path / "optimal.svg"
    written = viz.render(viz.optimal_graph_dot(solution, legend=False), out, "svg")
    assert written == out
    assert "<svg" in out.read_text()

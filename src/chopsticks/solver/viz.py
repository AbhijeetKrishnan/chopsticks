"""pydot rendering of solved Chopsticks graphs.

Requires the ``viz`` extra (``pydot``) and, for PNG output, the Graphviz ``dot``
binary on ``PATH``.
"""

import shutil
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pydot

from chopsticks.env.state import ChopsticksAction, ChopsticksState, Turn
from chopsticks.solver.graph import CANONICAL_START, EdgeType, Solution, canonical
from chopsticks.solver.retrograde import optimal_children

VAL_TO_COLOR = {
    0: "gray39",
    1: "darkgreen",
    -1: "crimson",
}

MAXIMIZING_PLAYER_TO_SHAPE = {
    True: "house",
    False: "invhouse",
}


def add_node(
    dot_graph: pydot.Dot,
    values: Dict[ChopsticksState, int],
    state: ChopsticksState,
) -> None:
    """Add ``state`` to ``dot_graph`` with a shape and colour for its value."""

    if state.is_terminal():
        shape = "box"
    else:
        shape = MAXIMIZING_PLAYER_TO_SHAPE[state.turn == Turn.P1]
    dot_graph.add_node(
        pydot.Node(
            str(state),
            label=str(state),
            shape=shape,
            color=VAL_TO_COLOR[values[state]],
        )
    )


def add_edge(
    dot_graph: pydot.Dot,
    from_state: ChopsticksState,
    to_state: ChopsticksState,
    action: ChopsticksAction,
    edge_type: EdgeType,
) -> None:
    """Add a styled edge for ``action`` between two positions."""

    dir = "forward"
    style = "solid"
    constraint = True
    color = "#00000000"
    penwidth = 1.0
    match edge_type:
        case EdgeType.FORWARD:
            dir = "forward"
            style = "dashed"
            constraint = True
            color = "#00000080"
            penwidth = 1.0
        case EdgeType.BACK:
            dir = "back"
            style = "dashed"
            constraint = False
            color = "#80808080"
            penwidth = 0.5
        case EdgeType.TREE:
            dir = "forward"
            style = "solid"
            constraint = True
            color = "#000000ff"
            penwidth = 1.0
    dot_graph.add_edge(
        pydot.Edge(
            str(from_state),
            str(to_state),
            arrowhead="normal",
            dir=dir,
            label=str(action),
            style=style,
            constraint=constraint,
            color=color,
            penwidth=penwidth,
        )
    )


def optimal_graph_dot(
    solution: Solution,
    start: ChopsticksState = CANONICAL_START,
) -> pydot.Dot:
    """Build the graph of optimal moves for both players from ``start``.

    Only moves that keep the game-theoretic value unchanged are followed, so the
    result is the tree (with cycles) of lines that occur when neither side errs."""

    graph, values = solution
    dot_graph = pydot.Dot(
        "Optimal Decision Tree for Chopsticks",
        graph_type="graph",
        bgcolor="lightgray",
        simplify=True,
    )
    root = canonical(start)
    queue: deque[ChopsticksState] = deque([root])
    seen: Set[ChopsticksState] = {root}
    add_node(dot_graph, values, root)
    while queue:
        state = queue.popleft()
        for child, action, edge_type in optimal_children(graph, values, state):
            if child not in seen:
                seen.add(child)
                queue.append(child)
                add_node(dot_graph, values, child)
            add_edge(dot_graph, state, child, action, edge_type)
    return dot_graph


def strategy_graph_dot(
    solution: Solution,
    player: Turn,
    start: ChopsticksState = CANONICAL_START,
) -> pydot.Dot:
    """Build the graph of ``player``'s optimal strategy against every opponent reply.

    At a position where ``player`` is to move only value-preserving (optimal)
    actions are followed; at the opponent's positions every legal move is
    followed. The result therefore shows what ``player`` does in every position
    that can arise, no matter how the opponent plays."""

    graph, values = solution
    opponent = Turn.P2 if player == Turn.P1 else Turn.P1
    dot_graph = pydot.Dot(
        f"{player} Optimal Actions with All {opponent} Moves",
        graph_type="graph",
        bgcolor="lightgray",
        simplify=True,
    )
    root = canonical(start)
    queue: deque[ChopsticksState] = deque([root])
    seen: Set[ChopsticksState] = {root}
    add_node(dot_graph, values, root)
    while queue:
        state = queue.popleft()
        if state.turn == player:
            edges = optimal_children(graph, values, state)
        else:
            edges = graph[state]
        for child, action, edge_type in edges:
            if child not in seen:
                seen.add(child)
                queue.append(child)
                add_node(dot_graph, values, child)
            add_edge(dot_graph, state, child, action, edge_type)
    return dot_graph


def p1_winning_dot(
    solution: Solution,
    start: ChopsticksState = CANONICAL_START,
) -> pydot.Dot:
    """Player 1's optimal strategy against every Player 2 move."""

    return strategy_graph_dot(solution, Turn.P1, start)


def p2_winning_dot(
    solution: Solution,
    start: ChopsticksState = CANONICAL_START,
) -> pydot.Dot:
    """Player 2's optimal strategy against every Player 1 move."""

    return strategy_graph_dot(solution, Turn.P2, start)


def full_state_graph_dot() -> pydot.Dot:
    """Build the brute-forced graph of every hand/turn combination and its moves.

    Unlike the solver graph this enumerates all ``5**4 * 2`` states (including
    non-canonical hand orderings) and every legal-move edge, without solving."""

    dot_graph = pydot.Dot(
        "Decision Tree for Chopsticks",
        graph_type="graph",
        bgcolor="lightgray",
    )
    combos: List[Tuple[int, int, int, int, Turn]] = [
        (p1_min, p1_max, p2_min, p2_max, turn)
        for p1_min in range(5)
        for p1_max in range(5)
        for p2_min in range(5)
        for p2_max in range(5)
        for turn in (Turn.P1, Turn.P2)
    ]
    for p1_min, p1_max, p2_min, p2_max, turn in combos:
        state = ChopsticksState(p1_min, p1_max, p2_min, p2_max, turn)
        value = 0
        if state.is_terminal():
            winner = state.winner()
            value = 1 if winner == Turn.P1 else -1 if winner == Turn.P2 else 0
        add_node(dot_graph, {state: value}, state)
    for p1_min, p1_max, p2_min, p2_max, turn in combos:
        state = ChopsticksState(p1_min, p1_max, p2_min, p2_max, turn)
        for action in state.legal_moves():
            add_edge(dot_graph, state, state.transition(action), action, EdgeType.TREE)
    return dot_graph


def render(dot_graph: pydot.Dot, path: Path, fmt: str) -> Path | None:
    """Write ``dot_graph`` to ``path`` in ``fmt`` (``"dot"`` or ``"png"``).

    ``"dot"`` always succeeds. ``"png"`` needs the Graphviz ``dot`` binary; if it
    is missing, a warning is emitted and ``None`` is returned instead of raising."""

    if fmt == "png" and shutil.which("dot") is None:
        warnings.warn(
            "Graphviz 'dot' binary not found on PATH; skipping PNG output. "
            "Install graphviz or pass --format dot.",
            stacklevel=2,
        )
        return None
    dot_graph.write(str(path), format="raw" if fmt == "dot" else "png")
    return path

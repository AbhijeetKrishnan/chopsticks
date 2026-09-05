"""Reachable-state graph construction for the Chopsticks solver.

Positions are identified only by their hands and whose turn it is; the repetition
bookkeeping carried by :class:`~chopsticks.env.state.ChopsticksState` (its
``is_repeated`` flag and ``history``) is intentionally dropped here, because the
retrograde valuation in :mod:`chopsticks.solver.retrograde` already accounts for
repetition by treating unresolved cycles as draws.
"""

from collections import defaultdict
from enum import Enum, auto
from typing import DefaultDict, Dict, Iterator, List, NamedTuple, Tuple

from chopsticks.env.state import ChopsticksAction, ChopsticksState, Turn

CANONICAL_START = ChopsticksState(1, 1, 1, 1, Turn.P1)

Graph = Dict[
    ChopsticksState,
    List[Tuple[ChopsticksState, ChopsticksAction, "EdgeType"]],
]


class Color(Enum):
    """Colour states for nodes during graph traversal, following CLRS."""

    WHITE = auto()
    GRAY = auto()
    BLACK = auto()


class EdgeType(Enum):
    """Types of edges during graph traversal, following CLRS.

    Cross edges are lumped in with ``FORWARD`` since the drawing code treats
    them identically."""

    FORWARD = auto()
    BACK = auto()
    TREE = auto()


class Solution(NamedTuple):
    """A solved game: the reachable graph plus each position's value.

    ``values`` maps every position in ``graph`` to ``+1`` (Player 1 wins with
    optimal play), ``-1`` (Player 2 wins), or ``0`` (draw)."""

    graph: Graph
    values: Dict[ChopsticksState, int]


def canonical(state: ChopsticksState) -> ChopsticksState:
    """Return a copy of ``state`` identified only by its hands and turn.

    The repetition flag and history are cleared so that a position corresponds
    to a single node regardless of the path taken to reach it. ``is_terminal``
    then only fires when a player has lost both hands."""

    return ChopsticksState(
        state.p1_min,
        state.p1_max,
        state.p2_min,
        state.p2_max,
        state.turn,
    )


def children(
    state: ChopsticksState,
) -> List[Tuple[ChopsticksState, ChopsticksAction]]:
    """Return the ``(child, action)`` pairs reachable from a non-terminal position."""

    return [
        (canonical(state.transition(action)), action) for action in state.legal_moves()
    ]


def build_reachable_graph(
    start: ChopsticksState,
    *,
    progress: bool = False,
) -> Graph:
    """Depth-first search from ``start``, recording every edge and its CLRS type.

    Tree edges lead to undiscovered positions, forward/cross edges to positions
    already finished, and back edges close a cycle (a repeated position). Set
    ``progress`` to show a ``tqdm`` progress bar on stderr."""

    graph: DefaultDict[
        ChopsticksState,
        List[Tuple[ChopsticksState, ChopsticksAction, EdgeType]],
    ] = defaultdict(list)
    color: DefaultDict[ChopsticksState, Color] = defaultdict(lambda: Color.WHITE)

    def frame(
        state: ChopsticksState,
    ) -> Tuple[
        ChopsticksState,
        Iterator[Tuple[ChopsticksState, ChopsticksAction]],
    ]:
        moves = [] if state.is_terminal() else children(state)
        return state, iter(moves)

    start = canonical(start)
    color[start] = Color.GRAY
    graph[start]  # ensure the start is a key even with no outgoing edges
    stack: List[
        Tuple[ChopsticksState, Iterator[Tuple[ChopsticksState, ChopsticksAction]]]
    ] = [frame(start)]

    pbar = None
    if progress:
        from tqdm import tqdm

        pbar = tqdm(desc="Exploring states", unit="state")

    while stack:
        state, pending = stack[-1]
        descended = False
        for child, action in pending:
            match color[child]:
                case Color.WHITE:
                    graph[state].append((child, action, EdgeType.TREE))
                    color[child] = Color.GRAY
                    graph[child]
                    stack.append(frame(child))
                    descended = True
                    break
                case Color.GRAY:
                    graph[state].append((child, action, EdgeType.BACK))
                case Color.BLACK:
                    graph[state].append((child, action, EdgeType.FORWARD))
        if not descended:
            color[state] = Color.BLACK
            stack.pop()
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    return dict(graph)

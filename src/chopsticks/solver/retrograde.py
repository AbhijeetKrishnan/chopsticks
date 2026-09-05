"""Retrograde (backward-induction) valuation of the reachable graph.

A position is a win for the player to move if any move reaches a position that
is losing for the opponent, a loss if every move reaches a position that is
winning for the opponent, and a draw otherwise. A "draw" means best play cycles
forever, which the repetition rule settles as a draw, so cyclic positions that
cannot reach a decisive result are left drawn.
"""

from collections import defaultdict, deque
from typing import DefaultDict, Dict, List, Set, Tuple

from chopsticks.env.state import ChopsticksAction, ChopsticksState, Turn
from chopsticks.solver.graph import (
    CANONICAL_START,
    EdgeType,
    Graph,
    Solution,
    build_reachable_graph,
)


def retrograde_values(graph: Graph) -> Dict[ChopsticksState, int]:
    """Label every position in ``graph`` by backward induction.

    Returns ``+1`` if Player 1 wins with optimal play, ``-1`` if Player 2 wins,
    and ``0`` for a draw (an unresolved cycle, settled as a draw by the
    repetition rule)."""

    distinct_children: Dict[ChopsticksState, List[ChopsticksState]] = {}
    parents: DefaultDict[ChopsticksState, Set[ChopsticksState]] = defaultdict(set)
    for state, edges in graph.items():
        kids: List[ChopsticksState] = []
        for child, _action, _edge_type in edges:
            if child not in kids:
                kids.append(child)
        distinct_children[state] = kids
        for child in kids:
            parents[child].add(state)

    values: Dict[ChopsticksState, int] = {}
    # number of a position's moves already known to be winning for the opponent
    losing_moves: DefaultDict[ChopsticksState, int] = defaultdict(int)

    queue: deque[ChopsticksState] = deque()
    for state in graph:
        if state.is_terminal():
            winner = state.winner()
            values[state] = 1 if winner == Turn.P1 else -1 if winner == Turn.P2 else 0
            queue.append(state)

    while queue:
        state = queue.popleft()
        value = values[state]
        for parent in parents[state]:
            if parent in values:
                continue
            parent_win = 1 if parent.turn == Turn.P1 else -1
            if value == parent_win:
                # a move into a position the parent's player already wins
                values[parent] = parent_win
                queue.append(parent)
            elif value == -parent_win:
                losing_moves[parent] += 1
                if losing_moves[parent] == len(distinct_children[parent]):
                    # every move loses for the player to move
                    values[parent] = -parent_win
                    queue.append(parent)
            # a drawing child never forces the parent's value on its own

    for state in graph:
        values.setdefault(state, 0)  # still unresolved => on a cycle => draw

    return values


def optimal_children(
    graph: Graph,
    values: Dict[ChopsticksState, int],
    state: ChopsticksState,
) -> List[Tuple[ChopsticksState, ChopsticksAction, EdgeType]]:
    """Return the outgoing edges that preserve the position's value.

    The value of a position is already the best the player to move can achieve
    (a max for Player 1, a min for Player 2), so an optimal move is exactly one
    whose child shares that value."""

    value = values[state]
    return [
        (child, action, edge_type)
        for child, action, edge_type in graph[state]
        if values[child] == value
    ]


def solve(
    start: ChopsticksState = CANONICAL_START,
    *,
    progress: bool = False,
) -> Solution:
    """Enumerate the reachable graph from ``start`` and label every position.

    ``start`` defaults to the canonical opening position (both players holding
    ``1, 1`` with Player 1 to move). Set ``progress`` to show a progress bar."""

    graph = build_reachable_graph(start, progress=progress)
    return Solution(graph, retrograde_values(graph))

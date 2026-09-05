"""Summary statistics and reachability checks for a solved game."""

from typing import Dict, List

from chopsticks.env.state import ChopsticksState, Turn
from chopsticks.solver.graph import CANONICAL_START, Solution

_OUTCOME = {1: "P1 win", -1: "P2 win", 0: "draw"}


def all_canonical_states() -> List[ChopsticksState]:
    """Return the 450 canonical positions.

    A canonical position has each hand in ``0..4`` with ``min <= max`` per
    player, for both turns."""

    states: List[ChopsticksState] = []
    for p1_min in range(5):
        for p1_max in range(p1_min, 5):
            for p2_min in range(5):
                for p2_max in range(p2_min, 5):
                    for turn in (Turn.P1, Turn.P2):
                        states.append(
                            ChopsticksState(p1_min, p1_max, p2_min, p2_max, turn)
                        )
    return states


def find_missing_states(
    values: Dict[ChopsticksState, int],
) -> List[ChopsticksState]:
    """Return the canonical positions absent from ``values`` (unreachable)."""

    return [state for state in all_canonical_states() if state not in values]


def graph_stats(
    solution: Solution,
    start: ChopsticksState = CANONICAL_START,
) -> Dict[str, object]:
    """Return a machine-readable summary of ``solution``.

    Keys: ``reachable_states``, ``edges``, ``terminal_states``, ``p1_wins``,
    ``p2_wins``, ``draws``, ``missing_states``, ``start_value``,
    ``start_outcome``."""

    graph, values = solution
    counts = list(values.values())
    return {
        "reachable_states": len(graph),
        "edges": sum(len(edges) for edges in graph.values()),
        "terminal_states": sum(1 for state in graph if state.is_terminal()),
        "p1_wins": counts.count(1),
        "p2_wins": counts.count(-1),
        "draws": counts.count(0),
        "missing_states": len(find_missing_states(values)),
        "start_value": values[start],
        "start_outcome": _OUTCOME[values[start]],
    }


def format_stats(stats: Dict[str, object]) -> str:
    """Render :func:`graph_stats` output as a human-readable block."""

    return "\n".join(
        [
            "Chopsticks — retrograde analysis",
            f"  start position:        {stats['start_outcome']} "
            f"(value {stats['start_value']})",
            f"  reachable states:      {stats['reachable_states']}",
            f"  terminal states:       {stats['terminal_states']}",
            f"  edges:                 {stats['edges']}",
            f"  P1 wins / P2 wins:     {stats['p1_wins']} / {stats['p2_wins']}",
            f"  drawn states:          {stats['draws']}",
            f"  unreachable canonical: {stats['missing_states']}",
        ]
    )

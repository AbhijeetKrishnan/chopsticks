"""Exact retrograde-analysis solver for the game of Chopsticks.

Importing this package pulls in only the standard library and
:mod:`chopsticks.env.state`; the pydot visualisation lives in
:mod:`chopsticks.solver.viz` and is imported separately.
"""

from chopsticks.solver.graph import (
    CANONICAL_START,
    Color,
    EdgeType,
    Graph,
    Solution,
    build_reachable_graph,
    canonical,
    children,
)
from chopsticks.solver.retrograde import (
    optimal_children,
    retrograde_values,
    solve,
)
from chopsticks.solver.stats import (
    all_canonical_states,
    find_missing_states,
    format_stats,
    graph_stats,
)

__all__ = [
    "CANONICAL_START",
    "Color",
    "EdgeType",
    "Graph",
    "Solution",
    "build_reachable_graph",
    "canonical",
    "children",
    "optimal_children",
    "retrograde_values",
    "solve",
    "all_canonical_states",
    "find_missing_states",
    "format_stats",
    "graph_stats",
]

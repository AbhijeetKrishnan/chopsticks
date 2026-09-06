"""pydot rendering of solved Chopsticks graphs.

Requires the ``viz`` extra (``pydot``) and, for PNG/SVG output, the Graphviz
``dot`` binary on ``PATH``.

Encoding used by every graph
----------------------------
* Node id is ``str(state)``; the visible label is a compact two-line
  ``P1 m M`` / ``P2 m M`` (a dead hand shows as ``x``). Terminal positions are
  drawn as a ``doubleoctagon`` labelled with the winner.
* Node shape marks whose turn it is: ``house`` = Player 1 to move, ``invhouse``
  = Player 2. Node fill is the game-theoretic value: pale blue-grey = draw,
  green = Player 1 win, red = Player 2 win. The start position has a bold gold
  border.
* Edges point in the order of play. A thick blue edge is an optimal Player 1
  move, a thick orange edge an optimal Player 2 move, a thin grey edge is an
  opponent reply drawn only for completeness (strategy graphs), and a dashed red
  edge returns to a position seen earlier -- those cycles are what the
  repetition rule settles as a draw. Parallel moves that reach the same position
  share one edge whose label joins their codes with ``/``.
"""

import shutil
import warnings
from collections import deque
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pydot

from chopsticks.env.state import ChopsticksAction, ChopsticksState, Turn
from chopsticks.solver.graph import CANONICAL_START, EdgeType, Solution, canonical
from chopsticks.solver.retrograde import optimal_children

# value -> (fill, border, text) tuned to read on a white background
VAL_STYLE: Dict[int, Dict[str, str]] = {
    0: {"fillcolor": "#eef1f4", "color": "#566573", "fontcolor": "#1b2631"},
    1: {"fillcolor": "#d7ecd7", "color": "#2e7d32", "fontcolor": "#14421a"},
    -1: {"fillcolor": "#fbdcdc", "color": "#c62828", "fontcolor": "#5b1515"},
}

MAXIMIZING_PLAYER_TO_SHAPE = {
    True: "house",
    False: "invhouse",
}

START_BORDER = "#c8a415"  # gold

# colour of a PRIMARY (optimal) edge, by the player making the move
PRIMARY_COLOR_BY_MOVER: Dict[Turn | None, str] = {
    Turn.P1: "#1f5fbf",  # blue
    Turn.P2: "#d97706",  # orange
    None: "#333333",
}


class EdgeRole(Enum):
    """How an edge should read, independent of the DFS edge type it came from."""

    PRIMARY = auto()  # an optimal / chosen move -- the line of play being asserted
    REPLY = auto()  # an opponent move shown only for completeness
    REPETITION = auto()  # returns to an already-seen position (a draw cycle)
    PLAIN = auto()  # unclassified (the brute-forced full-state graph)


def _mk_node(name: str, attrs: Dict[str, str]) -> pydot.Node:
    """A ``pydot.Node`` with ``attrs`` applied (avoids ``**`` kwarg unpacking)."""

    node = pydot.Node(name)
    for key, value in attrs.items():
        node.set(key, value)
    return node


def _mk_edge(src: str, dst: str, attrs: Dict[str, str]) -> pydot.Edge:
    """A ``pydot.Edge`` with ``attrs`` applied."""

    edge = pydot.Edge(src, dst)
    for key, value in attrs.items():
        edge.set(key, value)
    return edge


def _hand(n: int) -> str:
    """Render one hand's finger count; a dead hand shows as ``x``."""

    return "x" if n == 0 else str(n)


def _node_label(state: ChopsticksState, is_start: bool) -> str:
    """Compact label: winner for a terminal, else two lines of hands."""

    if state.is_terminal():
        winner = state.winner()
        if winner == Turn.P1:
            return f"P1 WINS\\nP2 {_hand(state.p2_min)} {_hand(state.p2_max)}"
        if winner == Turn.P2:
            return f"P2 WINS\\nP1 {_hand(state.p1_min)} {_hand(state.p1_max)}"
        return "DRAW"
    body = (
        f"P1 {_hand(state.p1_min)} {_hand(state.p1_max)}"
        f"\\nP2 {_hand(state.p2_min)} {_hand(state.p2_max)}"
    )
    return f"START\\n{body}" if is_start else body


def add_node(
    dot_graph: pydot.Dot,
    values: Dict[ChopsticksState, int],
    state: ChopsticksState,
    *,
    is_start: bool = False,
) -> None:
    """Add ``state`` with a shape for its turn and a fill for its value."""

    vstyle = VAL_STYLE[values[state]]
    if state.is_terminal():
        shape = "doubleoctagon"
        penwidth = "2"
    else:
        shape = MAXIMIZING_PLAYER_TO_SHAPE[state.turn == Turn.P1]
        penwidth = "1.2"
    attrs: Dict[str, str] = {
        "label": _node_label(state, is_start),
        "shape": shape,
        "style": "filled",
        "fillcolor": vstyle["fillcolor"],
        "color": vstyle["color"],
        "fontcolor": vstyle["fontcolor"],
        "fontname": "Helvetica",
        "fontsize": "10",
        "margin": "0.08,0.04",
        "penwidth": penwidth,
    }
    if is_start:
        attrs["color"] = START_BORDER
        attrs["penwidth"] = "3"
        attrs["peripheries"] = "2"
    dot_graph.add_node(_mk_node(str(state), attrs))


def _edge_attrs(role: EdgeRole, mover: Turn | None) -> Dict[str, str]:
    """Graphviz attributes for one edge role."""

    if role is EdgeRole.PRIMARY:
        color = PRIMARY_COLOR_BY_MOVER[mover]
        return {
            "color": color,
            "fontcolor": color,
            "penwidth": "2.0",
            "style": "solid",
            "arrowhead": "normal",
            "arrowsize": "0.9",
            "fontsize": "10",
            "constraint": "true",
        }
    if role is EdgeRole.REPLY:
        return {
            "color": "#9aa5b1",
            "fontcolor": "#7b8794",
            "penwidth": "1.0",
            "style": "solid",
            "arrowhead": "vee",
            "arrowsize": "0.7",
            "fontsize": "8",
            "constraint": "true",
        }
    if role is EdgeRole.REPETITION:
        return {
            "color": "#c0392b",
            "fontcolor": "#c0392b",
            "penwidth": "1.4",
            "style": "dashed",
            "arrowhead": "onormal",
            "arrowsize": "0.8",
            "fontsize": "8",
            "constraint": "false",
        }
    return {
        "color": "#8892a0",
        "fontcolor": "#8892a0",
        "penwidth": "0.6",
        "style": "solid",
        "arrowhead": "vee",
        "arrowsize": "0.5",
        "fontsize": "6",
        "constraint": "true",
    }


def add_edge(
    dot_graph: pydot.Dot,
    from_state: ChopsticksState,
    to_state: ChopsticksState,
    label: str,
    role: EdgeRole,
    *,
    mover: Turn | None = None,
) -> None:
    """Add one styled edge; ``label`` is the (possibly merged) move code(s)."""

    attrs = {"fontname": "Helvetica", "label": label, **_edge_attrs(role, mover)}
    dot_graph.add_edge(_mk_edge(str(from_state), str(to_state), attrs))


def _merge_edges(
    edges: Iterable[Tuple[ChopsticksState, ChopsticksAction, EdgeType]],
) -> List[Tuple[ChopsticksState, str, bool]]:
    """Collapse parallel moves to the same child into one labelled edge.

    Returns ``(child, label, is_repetition)`` in first-seen order. ``label``
    joins the move codes with ``/``; ``is_repetition`` is true only when *every*
    move to that child is a DFS back edge (i.e. the child sits on the current
    line of play -- a genuine cycle), not merely a transposition."""

    grouped: Dict[ChopsticksState, List[Tuple[ChopsticksAction, EdgeType]]] = {}
    order: List[ChopsticksState] = []
    for child, action, edge_type in edges:
        if child not in grouped:
            grouped[child] = []
            order.append(child)
        grouped[child].append((action, edge_type))
    merged: List[Tuple[ChopsticksState, str, bool]] = []
    for child in order:
        items = grouped[child]
        label = "/".join(sorted({str(action) for action, _ in items}))
        is_repetition = all(edge_type is EdgeType.BACK for _, edge_type in items)
        merged.append((child, label, is_repetition))
    return merged


def _base_graph(
    name: str,
    label: str,
    *,
    rankdir: str,
    nodesep: str,
    ranksep: str,
    splines: str,
    dpi: str,
) -> pydot.Dot:
    """A directed graph with the shared look (white bg, Helvetica, top caption)."""

    return pydot.Dot(
        name,
        graph_type="digraph",
        simplify=False,
        bgcolor="white",
        compound="true",
        rankdir=rankdir,
        nodesep=nodesep,
        ranksep=ranksep,
        splines=splines,
        ordering="out",
        dpi=dpi,
        fontname="Helvetica",
        fontsize="12",
        labelloc="t",
        label=label,
        pad="0.3",
    )


# --------------------------------------------------------------------------- #
# legend
# --------------------------------------------------------------------------- #


def _key_text(*, strategy: bool) -> str:
    """Left-justified (``\\l``) text block: edge meanings and the action codes."""

    reply = (
        r"  thin grey    = opponent reply (shown for completeness)\l"
        if strategy
        else ""
    )
    return (
        r"Edges point in the order of play:\l"
        r"  thick blue   = Player 1 optimal move\l"
        r"  thick orange = Player 2 optimal move\l"
        + reply
        + r"  dashed red   = returns to a position seen earlier -> draw\l"
        r"\l"
        r"Move codes (mover's hand -> target):\l"
        r"  m2m  min -> opp min       M2m  max -> opp min\l"
        r"  m2M  min -> opp max       M2M  max -> opp max\l"
        r"  m2s  min -> own other     M2s  max -> own other\l"
    )


def _add_legend(dot_graph: pydot.Dot, *, strategy: bool) -> None:
    """Attach a ``cluster_legend`` explaining shapes, fills, edges and codes.

    All node ids are prefixed ``legend_`` so they never collide with a real
    position id (which always starts with ``p1=``)."""

    legend = pydot.Subgraph(
        "cluster_legend",
        label="Legend",
        labelloc="t",
        labeljust="l",
        style="filled",
        fillcolor="#f7f7f7",
        color="#999999",
        fontname="Helvetica",
        fontsize="11",
        margin="8",
    )
    draw = VAL_STYLE[0]

    def swatch(nid: str, text: str, **kw: str) -> str:
        name = f"legend_{nid}"
        attrs = {
            "label": text,
            "style": "filled",
            "fontname": "Helvetica",
            "fontsize": "9",
            **kw,
        }
        legend.add_node(_mk_node(name, attrs))
        return name

    swatches = [
        swatch("p1turn", "P1 to move", shape="house", **draw),
        swatch("p2turn", "P2 to move", shape="invhouse", **draw),
        swatch(
            "terminal",
            "terminal\\n(winner in label)",
            shape="doubleoctagon",
            fillcolor="white",
            color="#566573",
            penwidth="2",
        ),
        swatch("draw", "draw (0)", shape="box", **draw),
        swatch("p1win", "P1 wins (+1)", shape="box", **VAL_STYLE[1]),
        swatch("p2win", "P2 wins (-1)", shape="box", **VAL_STYLE[-1]),
        swatch(
            "start",
            "start position",
            shape="box",
            fillcolor=draw["fillcolor"],
            color=START_BORDER,
            penwidth="3",
            peripheries="2",
        ),
    ]

    legend.add_node(
        _mk_node(
            "legend_key",
            {
                "shape": "box",
                "style": "filled",
                "fillcolor": "white",
                "color": "#cccccc",
                "fontname": "Courier",
                "fontsize": "9",
                "label": _key_text(strategy=strategy),
            },
        )
    )

    # invisible chain keeps the swatches stacked with the text block below them
    chain = [*swatches, "legend_key"]
    for upper, lower in zip(chain, chain[1:]):
        legend.add_edge(_mk_edge(upper, lower, {"style": "invis"}))

    dot_graph.add_subgraph(legend)


# --------------------------------------------------------------------------- #
# graph builders
# --------------------------------------------------------------------------- #


def optimal_graph_dot(
    solution: Solution,
    start: ChopsticksState = CANONICAL_START,
    *,
    legend: bool = True,
) -> pydot.Dot:
    """Build the graph of optimal moves for both players from ``start``.

    Only moves that keep the game-theoretic value unchanged are followed, so the
    result is the tree (with cycles) of lines that occur when neither side errs.
    From the canonical start the game is a draw, so the interesting structure is
    the dashed-red repetition cycles."""

    graph, values = solution
    dot_graph = _base_graph(
        "optimal_graph",
        "Optimal play from the canonical start — the game is a DRAW.\\n"
        "Dashed red edges close the cycles that the repetition rule settles as a draw.",
        rankdir="TB",
        nodesep="0.35",
        ranksep="0.55",
        splines="true",
        dpi="110",
    )
    root = canonical(start)
    queue: deque[ChopsticksState] = deque([root])
    seen: Set[ChopsticksState] = {root}
    add_node(dot_graph, values, root, is_start=True)
    while queue:
        state = queue.popleft()
        for child, label, is_repetition in _merge_edges(
            optimal_children(graph, values, state)
        ):
            if child not in seen:
                seen.add(child)
                queue.append(child)
                add_node(dot_graph, values, child)
            role = EdgeRole.REPETITION if is_repetition else EdgeRole.PRIMARY
            add_edge(dot_graph, state, child, label, role, mover=state.turn)
    if legend:
        _add_legend(dot_graph, strategy=False)
    return dot_graph


def strategy_graph_dot(
    solution: Solution,
    player: Turn,
    start: ChopsticksState = CANONICAL_START,
    *,
    legend: bool = True,
) -> pydot.Dot:
    """Build the graph of ``player``'s optimal strategy against every opponent reply.

    At a position where ``player`` is to move only value-preserving (optimal)
    actions are followed (thick coloured edges); at the opponent's positions
    every legal move is followed (thin grey edges). The result shows what
    ``player`` does in every position that can arise, no matter how the opponent
    plays."""

    graph, values = solution
    opponent = Turn.P2 if player == Turn.P1 else Turn.P1
    dot_graph = _base_graph(
        f"{player}_strategy",
        f"Player {'1' if player == Turn.P1 else '2'}'s optimal strategy: "
        f"{player} plays only value-preserving moves; every {opponent} reply is shown.",
        rankdir="TB",
        nodesep="0.3",
        ranksep="0.5",
        splines="polyline",
        dpi="100",
    )
    root = canonical(start)
    queue: deque[ChopsticksState] = deque([root])
    seen: Set[ChopsticksState] = {root}
    add_node(dot_graph, values, root, is_start=True)
    while queue:
        state = queue.popleft()
        is_owner = state.turn == player
        raw = optimal_children(graph, values, state) if is_owner else graph[state]
        for child, label, is_repetition in _merge_edges(raw):
            if child not in seen:
                seen.add(child)
                queue.append(child)
                add_node(dot_graph, values, child)
            if is_repetition:
                role = EdgeRole.REPETITION
            elif is_owner:
                role = EdgeRole.PRIMARY
            else:
                role = EdgeRole.REPLY
            add_edge(
                dot_graph,
                state,
                child,
                label,
                role,
                mover=player if is_owner else None,
            )
    if legend:
        _add_legend(dot_graph, strategy=True)
    return dot_graph


def p1_winning_dot(
    solution: Solution,
    start: ChopsticksState = CANONICAL_START,
    *,
    legend: bool = True,
) -> pydot.Dot:
    """Player 1's optimal strategy against every Player 2 move."""

    return strategy_graph_dot(solution, Turn.P1, start, legend=legend)


def p2_winning_dot(
    solution: Solution,
    start: ChopsticksState = CANONICAL_START,
    *,
    legend: bool = True,
) -> pydot.Dot:
    """Player 2's optimal strategy against every Player 1 move."""

    return strategy_graph_dot(solution, Turn.P2, start, legend=legend)


def full_state_graph_dot() -> pydot.Dot:
    """Build the brute-forced graph of every hand/turn combination and its moves.

    Unlike the solver graph this enumerates all ``5**4 * 2`` states (including
    non-canonical hand orderings) and every legal-move edge, without solving."""

    dot_graph = _base_graph(
        "full_graph",
        "Every hand/turn combination and every legal move (brute-forced, unsolved).",
        rankdir="LR",
        nodesep="0.2",
        ranksep="0.5",
        splines="polyline",
        dpi="72",
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
            add_edge(
                dot_graph,
                state,
                state.transition(action),
                str(action),
                EdgeRole.PLAIN,
            )
    return dot_graph


_RAW_FORMATS = {"dot": "raw", "png": "png", "svg": "svg"}


def render(dot_graph: pydot.Dot, path: Path, fmt: str) -> Path | None:
    """Write ``dot_graph`` to ``path`` in ``fmt`` (``"dot"``, ``"png"`` or ``"svg"``).

    ``"dot"`` always succeeds. ``"png"``/``"svg"`` need the Graphviz ``dot``
    binary; if it is missing, a warning is emitted and ``None`` is returned
    instead of raising."""

    if fmt in ("png", "svg") and shutil.which("dot") is None:
        warnings.warn(
            f"Graphviz 'dot' binary not found on PATH; skipping {fmt.upper()} output. "
            "Install graphviz or pass --format dot.",
            stacklevel=2,
        )
        return None
    dot_graph.write(str(path), format=_RAW_FORMATS[fmt])
    return path

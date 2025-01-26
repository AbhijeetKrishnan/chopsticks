from collections import defaultdict
from enum import Enum, auto
from typing import Dict, List, Set, Tuple

import pydot

from chopsticks import chopsticks_v0
from chopsticks.env.state import ChopsticksAction, ChopsticksState, Turn

MAX, MIN = 1000, -1000
VAL_TO_COLOR = {
    0: "gray39",
    1: "darkgreen",
    -1: "crimson",
}
MAXIMIZING_PLAYER_TO_SHAPE = {
    True: "house",
    False: "invhouse",
}


class Color(Enum):
    WHITE = auto()
    GRAY = auto()
    BLACK = auto()


class EdgeType(Enum):
    FORWARD = auto()
    BACK = auto()
    TREE = auto()


def minimax(
    env: chopsticks_v0.env,
    alpha_beta_pruning: bool = True,
) -> Tuple[Dict[ChopsticksState, List[ChopsticksState]], Dict[ChopsticksState, int]]:
    stack: List[
        Tuple[
            ChopsticksState,
            bool,
            int,
            int,
        ]
    ] = []  # stack of function calls
    cache: Dict[ChopsticksState, int] = {}
    color: Dict[ChopsticksState, Color] = defaultdict(
        lambda: Color.WHITE
    )  # visited state of each node based on CLRS
    graph: Dict[
        ChopsticksState, List[Tuple[ChopsticksState, ChopsticksAction, EdgeType]]
    ] = defaultdict(list)  # graph of decision tree

    # initialize variables
    stack.append(
        (
            env.game_state.canonical,  # curr state
            True,  # isMaximizingPlayer
            MIN,  # alpha
            MAX,  # beta
        )
    )

    while stack:
        state, isMaximizingPlayer, alpha, beta = stack[-1]
        color[state] = Color.GRAY
        # explore state and find its value
        if state.is_terminal():
            if state.winner() == Turn.P1:
                val = 1
            elif state.winner() == Turn.P2:
                val = -1
            graph[state] = []
        else:
            if isMaximizingPlayer:
                best = MIN
                shouldContinue = False
                for action in state.legal_moves():
                    child = state.transition(action)
                    if (
                        color[child] == Color.BLACK
                    ):  # forward/cross edge, we've seen and computed the value for child
                        assert child in cache
                        val = cache[child]
                        graph[state].append((child, action, EdgeType.FORWARD))
                    elif (
                        color[child] == Color.GRAY
                    ):  # back edge, we've seen but not yet computed the value for child
                        # we have a cycle
                        val = 0
                        graph[state].append((child, action, EdgeType.BACK))
                    else:  # tree edge
                        graph[state].append((child, action, EdgeType.TREE))
                        # we need to compute the value of this state, so "call" the function again on it
                        stack.append((child, False, alpha, beta))
                        shouldContinue = True
                        break
                    # all the children have been explored and values computed
                    best = max(best, val)
                    alpha = max(alpha, best)
                    if alpha_beta_pruning and beta <= alpha:
                        # prune rest of children by not exploring them
                        break
                if shouldContinue:
                    continue
                # we now have the value of the current node
                val = best
            else:
                best = MAX
                shouldContinue = False
                for action in state.legal_moves():
                    child = state.transition(action)
                    if color[child] == Color.BLACK:  # forward/cross edge
                        assert child in cache
                        val = cache[child]
                        graph[state].append((child, action, EdgeType.FORWARD))
                    elif color[child] == Color.GRAY:  # back edge
                        val = 0
                        graph[state].append((child, action, EdgeType.BACK))
                    elif color[child] == Color.WHITE:  # tree edge
                        graph[state].append((child, action, EdgeType.TREE))
                        stack.append((child, True, alpha, beta))
                        shouldContinue = True
                        break
                    best = min(best, val)
                    beta = min(beta, best)
                    if alpha_beta_pruning and beta <= alpha:
                        break
                if shouldContinue:
                    continue
                val = best
        color[state] = Color.BLACK
        cache[state] = val
        stack.pop()
    return graph, cache


def draw_graph(
    graph: Dict[
        ChopsticksState, List[Tuple[ChopsticksState, ChopsticksAction, EdgeType]]
    ],
    results: Dict[ChopsticksState, int],
    graph_name: str = "output",
) -> None:
    dot_graph = pydot.Dot(
        "Decision Tree for Chopsticks", graph_type="graph", bgcolor="lightgray"
    )
    for state in graph.keys():
        # Draw the node
        if state.is_terminal():
            shape = "box"
        else:
            shape = MAXIMIZING_PLAYER_TO_SHAPE[state.turn == Turn.P1]
        color = VAL_TO_COLOR[results[state]]
        dot_graph.add_node(
            pydot.Node(
                str(state),
                label=str(state),
                shape=shape,
                color=color,
            )
        )
    for u in graph.keys():
        for v, action, edge_type in graph[u]:
            if edge_type == EdgeType.TREE:
                style = "solid"
                dot_graph.add_edge(
                    pydot.Edge(
                        str(u),
                        str(v),
                        label=str(action.name),
                        style=style,
                        arrowhead="normal",
                        dir="forward",
                    )
                )
            elif edge_type == EdgeType.BACK:
                style = "dashed"
                pass
            else:
                style = "dotted"
                pass
    print("Writing graph to dot...")
    dot_graph.write_raw(f"{graph_name}.dot")
    print(f"Wrote graph as dot file to {graph_name}.dot")

    print("Writing graph to png...")
    dot_graph.write_png(f"{graph_name}.png")
    print(f"Wrote graph as png file to {graph_name}.png")


def draw_optimal_graph(env: chopsticks_v0, results: Dict[ChopsticksState, int]) -> None:
    env.reset()
    queue: List[ChopsticksState] = [env.game_state]
    seen: Set[ChopsticksState] = {env.game_state}
    dot_graph = pydot.Dot(
        "Optimal Decision Tree for Chopsticks", graph_type="graph", bgcolor="lightgray"
    )
    dot_graph.add_node(
        pydot.Node(
            str(env.game_state),
            label=str(env.game_state),
            shape=MAXIMIZING_PLAYER_TO_SHAPE[env.game_state.turn == Turn.P1],
        )
    )
    while queue:
        state = queue.pop(0)
        for action in state.legal_moves():
            child = state.transition(action)
            if results[child] == 1:
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
                    dot_graph.add_node(
                        pydot.Node(
                            str(child),
                            label=str(child),
                            shape=MAXIMIZING_PLAYER_TO_SHAPE[child.turn == Turn.P1],
                        )
                    )
                dot_graph.add_edge(
                    pydot.Edge(
                        str(state),
                        str(child),
                        label=str(action.name),
                        arrowhead="normal",
                        dir="forward",
                    )
                )
    dot_graph.write_png("optimal_graph.png")


# TODO: build an agent to play optimally


if __name__ == "__main__":
    env = chopsticks_v0.env()
    seed = None
    env.reset(seed=seed)
    graph, results = minimax(env, False)
    # result with alpha_beta_pruning False and True is different - why? I'd expect it to be the same
    print("done")
    draw_graph(graph, results)
    # draw_optimal_graph(env, results)

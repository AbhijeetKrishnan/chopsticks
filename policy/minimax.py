from typing import Dict

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

graph = pydot.Dot(
    "Decision Tree for Chopsticks", graph_type="graph", bgcolor="lightgray"
)


def minimax(
    node: ChopsticksState,
    parent: ChopsticksState | None,
    action: ChopsticksAction | None,
    isMaximizingPlayer: bool,
    alpha: int,
    beta: int,
    cache: Dict[ChopsticksState, int] = {},
) -> int:
    node = node.canonical
    if node in cache:
        value = cache[node]
    else:
        # if node is a leaf node, then return its value
        is_terminal = node.is_terminal()
        if is_terminal:
            if node.winner() == Turn.P1:
                value = 1
            elif node.winner() == Turn.P2:
                value = -1
            else:
                value = 0
        else:
            if isMaximizingPlayer:
                best = MIN
                # for each child node
                for action in node.legal_moves():
                    child = node.transition(action)
                    if child in cache:
                        val = cache[child]
                    else:
                        val = minimax(child, node, action, False, alpha, beta)
                        # build child node here and connect to node
                    best = max(best, val)
                    alpha = max(alpha, best)
                    if beta <= alpha:
                        break
                value = best
            else:
                best = MAX
                # for each child node
                for action in node.legal_moves():
                    child = node.transition(action)
                    if child in cache:
                        val = cache[child]
                    else:
                        val = minimax(child, node, action, True, alpha, beta)
                        # build child node here and connect to node
                    best = min(best, val)
                    beta = min(beta, best)
                    if beta <= alpha:
                        break
                value = best
        cache[node] = value
    # print(node, isMaximizingPlayer, alpha, beta, value)
    if is_terminal:
        shape = "box"
    else:
        shape = MAXIMIZING_PLAYER_TO_SHAPE[isMaximizingPlayer]
    graph.add_node(
        pydot.Node(
            str(node),
            label=str(node),
            color=VAL_TO_COLOR[value],
            shape=shape,
        )
    )
    if parent:
        graph.add_edge(
            pydot.Edge(
                str(parent),
                str(node),
                label=str(action.name if action else ""),
                arrowhead="normal",
                dir="forward",
            )
        )
    return value


if __name__ == "__main__":
    env = chopsticks_v0.env()
    seed = None
    env.reset(seed=seed)
    minimax(env.game_state, None, None, True, MIN, MAX)
    graph.write_png("output.png")

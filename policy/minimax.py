from collections import defaultdict
from enum import Enum, auto
from typing import Dict, List, Set, Tuple

import pydot
from tqdm import tqdm

from chopsticks.env.chopsticks import ChopsticksEnv
from chopsticks.env.state import ChopsticksAction, ChopsticksState, Turn

MAX, MIN = 1000, -1000
VAL_TO_COLOR = {
    0: "gray39",
    1: "darkgreen",
    -1: "crimson",
    1000: "darkorange",
    -1000: "darkorange",
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
    env: ChopsticksEnv,
    use_alpha_beta_pruning: bool = True,
) -> Tuple[
    Dict[
        ChopsticksState,
        List[Tuple[ChopsticksState, ChopsticksAction, EdgeType]],
    ],
    Dict[ChopsticksState, int],
]:
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
        ChopsticksState,
        List[Tuple[ChopsticksState, ChopsticksAction, EdgeType]],
    ] = defaultdict(list)  # graph of decision tree

    # initialize variables
    stack.append(
        (
            env.game_state,  # curr state
            True,  # isMaximizingPlayer
            MIN,  # alpha
            MAX,  # beta
        )
    )

    pbar = tqdm(total=1250, desc="Exploring states", unit="state")
    while stack:
        state, isMaximizingPlayer, alpha, beta = stack[-1]
        color[state] = Color.GRAY
        # explore state and find its value
        val: int = 0
        if state.is_terminal():
            match state.winner():
                case Turn.P1:
                    val = 1
                case Turn.P2:
                    val = -1
                case None:
                    val = 0
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
                    ):  # back edge, we've seen, but not yet computed the value for child
                        # we have a cycle
                        # don't explore this node
                        val = 0
                        graph[state].append((child, action, EdgeType.BACK))
                    else:  # tree edge
                        graph[state].append((child, action, EdgeType.TREE))
                        # we need to compute the value of this state, so "call" the function again on it
                        stack.append((child, not isMaximizingPlayer, alpha, beta))
                        shouldContinue = True
                        break
                    # all the children have been explored and values computed
                    best = max(best, val)
                    alpha = max(alpha, best)
                    if use_alpha_beta_pruning and beta <= alpha:
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
                        stack.append((child, not isMaximizingPlayer, alpha, beta))
                        shouldContinue = True
                        break
                    # all the children have been explored and values computed
                    best = min(best, val)
                    beta = min(beta, best)
                    if use_alpha_beta_pruning and beta <= alpha:
                        # prune rest of children by not exploring them
                        break
                if shouldContinue:
                    continue
                # we now have the value of the current node
                val = best
        color[state] = Color.BLACK
        cache[state] = val
        pbar.update(1)
        # print(f"Explored state: {state} with history {state.history}")
        stack.pop()
    pbar.close()
    return graph, cache


def add_node(
    dot_graph: pydot.Dot,
    results: Dict[ChopsticksState, int],
    state: ChopsticksState,
) -> None:
    "Convenience function to draw a node in a consistent format"

    if state.is_terminal():
        shape = "box"
    else:
        shape = MAXIMIZING_PLAYER_TO_SHAPE[state.turn == Turn.P1]
    dot_graph.add_node(
        pydot.Node(
            str(state),
            label=str(state),
            shape=shape,
            color=VAL_TO_COLOR[results[state]],
        )
    )


def add_edge(
    dot_graph: pydot.Dot,
    from_state: ChopsticksState,
    to_state: ChopsticksState,
    action: ChopsticksAction,
    edge_type: EdgeType,
) -> None:
    "Convenience function to draw an edge in a consistent format"
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
    # color = "black"  # reset color to default
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


def draw_graph(
    graph: Dict[
        ChopsticksState,
        List[Tuple[ChopsticksState, ChopsticksAction, EdgeType]],
    ],
    results: Dict[ChopsticksState, int],
    graph_name: str = "output",
) -> None:
    dot_graph = pydot.Dot(
        "Decision Tree for Chopsticks",
        graph_type="graph",
        bgcolor="lightgray",
        simplify=True,
    )

    start_state = ChopsticksState(1, 1, 1, 1, Turn.P1)

    stack = [start_state]
    seen = {start_state}
    while stack:
        state = stack.pop()
        add_node(dot_graph, results, state)
        if state.turn == Turn.P1:
            best = MIN
            best_children = []
            for child, action, edge_type in graph[state]:
                if results[child] > best:
                    best = results[child]
                    best_children = [(child, action, edge_type)]
                elif results[child] == best:
                    best_children.append((child, action, edge_type))
            for child, action, edge_type in best_children:
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
                add_edge(
                    dot_graph,
                    state,
                    child,
                    action,
                    edge_type,
                )
        else:
            best = MAX
            best_children = []
            for child, action, edge_type in graph[state]:
                if results[child] < best:
                    best = results[child]
                    best_children = [(child, action, edge_type)]
                elif results[child] == best:
                    best_children.append((child, action, edge_type))
            for child, action, edge_type in best_children:
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
                add_edge(
                    dot_graph,
                    state,
                    child,
                    action,
                    edge_type,
                )

    print("Writing graph to dot...")
    dot_graph.write(f"{graph_name}.dot", format="raw")
    print(f"Wrote graph as dot file to {graph_name}.dot")

    print("Writing graph to png...")
    dot_graph.write(f"{graph_name}.png", format="png")
    print(f"Wrote graph as png file to {graph_name}.png")


def draw_optimal_graph(
    env: ChopsticksEnv,
    graph: Dict[
        ChopsticksState,
        List[Tuple[ChopsticksState, ChopsticksAction, EdgeType]],
    ],
    results: Dict[ChopsticksState, int],
) -> None:
    env.reset()
    queue: List[ChopsticksState] = [env.game_state]
    seen: Set[ChopsticksState] = {env.game_state}
    dot_graph = pydot.Dot(
        "Optimal Decision Tree for Chopsticks",
        graph_type="graph",
        bgcolor="lightgray",
        simplify=True,
    )
    add_node(
        dot_graph,
        results,
        env.game_state,
    )
    while queue:
        state = queue.pop(0)
        for child, action, edge_type in graph[state]:
            if results[child] > -1 and child not in seen:
                seen.add(child)
                queue.append(child)
                add_node(dot_graph, results, child)
                add_edge(
                    dot_graph,
                    state,
                    child,
                    action,
                    edge_type,
                )

    print("Writing optimal graph to dot...")
    dot_graph.write("optimal_graph.dot", format="raw")
    print("Wrote optimal graph as dot file to optimal_graph.dot")
    print("Writing optimal graph to png...")
    dot_graph.write("optimal_graph.png", format="png")
    print("Wrote optimal graph as png file to optimal_graph.png")


def draw_p1_winning_p2_all_moves(
    env: ChopsticksEnv,
    graph: Dict[
        ChopsticksState,
        List[Tuple[ChopsticksState, ChopsticksAction, EdgeType]],
    ],
    results: Dict[ChopsticksState, int],
    graph_name: str = "p1_winning_p2_all",
) -> None:
    """Graph showing P1's winning actions and all possible actions for P2."""
    env.reset()
    queue: List[ChopsticksState] = [env.game_state]
    seen: Set[ChopsticksState] = {env.game_state}
    dot_graph = pydot.Dot(
        "P1 Winning Actions with All P2 Moves",
        graph_type="graph",
        bgcolor="lightgray",
        simplify=True,
    )
    add_node(dot_graph, results, env.game_state)

    while queue:
        state = queue.pop(0)
        add_node(dot_graph, results, state)

        if state.turn == Turn.P1:
            # For P1 (maximizing player), only show best move(s)
            best_value = MIN
            best_actions = []
            for child, action, edge_type in graph[state]:
                if results[child] > best_value:
                    best_value = results[child]
                    best_actions = [(child, action, edge_type)]
                elif results[child] == best_value:
                    best_actions.append((child, action, edge_type))

            for child, action, edge_type in best_actions:
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
                add_edge(
                    dot_graph,
                    state,
                    child,
                    action,
                    edge_type,
                )
        else:
            # For P2 (minimizing player), show all possible moves
            for child, action, edge_type in graph[state]:
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
                add_edge(
                    dot_graph,
                    state,
                    child,
                    action,
                    edge_type,
                )

    print("Writing P1 winning/P2 all moves graph to dot...")
    dot_graph.write(f"{graph_name}.dot", format="raw")
    print(f"Wrote graph as dot file to {graph_name}.dot")
    print("Writing graph to png...")
    dot_graph.write(f"{graph_name}.png", format="png")
    print(f"Wrote graph as png file to {graph_name}.png")


# TODO: build an agent to play optimally


if __name__ == "__main__":
    env = ChopsticksEnv(render_mode=None)
    seed = None
    env.reset(seed=seed)
    graph, results = minimax(env, False)
    print("done")
    draw_graph(graph, results)
    draw_optimal_graph(env, graph, results)
    draw_p1_winning_p2_all_moves(env, graph, results)

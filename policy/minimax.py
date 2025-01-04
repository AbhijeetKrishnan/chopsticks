from typing import Dict

from chopsticks import chopsticks_v0
from chopsticks.env.state import ChopsticksState, Turn

MAX, MIN = 1000, -1000


def minimax(
    node: ChopsticksState,
    isMaximizingPlayer: bool,
    alpha: int,
    beta: int,
    cache: Dict[ChopsticksState, int] = {},
) -> int:
    if node in cache:
        value = cache[node]
    else:
        # if node is a leaf node, then return its value
        if node.is_terminal():
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
                        val = minimax(child, False, alpha, beta)
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
                        val = minimax(child, True, alpha, beta)
                    best = min(best, val)
                    beta = min(beta, best)
                    if beta <= alpha:
                        break
                value = best
        cache[node] = value
    print(node, isMaximizingPlayer, alpha, beta, value)
    return value


if __name__ == "__main__":
    env = chopsticks_v0.env()
    seed = None
    env.reset(seed=seed)
    minimax(env.game_state, True, MIN, MAX)

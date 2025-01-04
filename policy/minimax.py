from chopsticks import chopsticks_v0
from chopsticks.env.state import ChopsticksState, Turn

MAX, MIN = 1000, -1000


def minimax(
    node: ChopsticksState,
    isMaximizingPlayer: bool,
    alpha: int,
    beta: int,
) -> int:
    print(node, isMaximizingPlayer, alpha, beta)

    # if node is a leaf node, then return its value
    if node.is_terminal():
        if node.winner() == Turn.P1:
            return 1
        else:
            return -1

    if isMaximizingPlayer:
        best = MIN
        # for each child node
        for action in node.legal_moves():
            child = node.transition(action)
            val = minimax(child, False, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = MAX
        # for each child node
        for action in node.legal_moves():
            child = node.transition(action)
            val = minimax(child, True, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


if __name__ == "__main__":
    env = chopsticks_v0.env()
    seed = None
    env.reset(seed=seed)
    minimax(env.game_state, True, MIN, MAX)

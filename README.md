# Chopsticks AEC Environment

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pyrefly](https://img.shields.io/endpoint?url=https://pyrefly.org/badge.json)](https://github.com/facebook/pyrefly)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

This is an implementation of the game of Chopsticks as a [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo)
[AEC](https://arxiv.org/abs/2009.13051) game.

## Rules

Chopsticks is a simple, 2-player, perfect information, zero-sum game that only requires both your hands to play.  Each
player has a number of "chopsticks" in each hand, denoted by the number of fingers held out. Since players will tend to
have only 5 fingers, the number of held chopsticks can only be between $[1, 4]$. Players take turns transferring
chopsticks to another hand. Transferring a chopstick involves touching the other player's (or your own) hand with
another, thereby adding the chopsticks in your hand to the other, modulo 5. If the total reaches 0, that hand is
considered "dead", and is out of play. If a player loses both their hands, then they lose.

It is possible to enter loops while playing caused due to a repetition of positions. We introduce a new rule that
declares a game a draw if any position is repeated more than once. A position is repeated if the players have the same
number of chopsticks in their hands (unordered) and it's the same player's turn as a previous position.

## Requirements

- Python 3.12+
- [Graphviz](https://graphviz.org/) (the `dot` binary) — only needed for PNG output from the
  solver: `sudo apt install graphviz` or `brew install graphviz`. Without it the solver still
  writes `.dot` files.

## Installation

### Local

```bash
git clone git@github.com:AbhijeetKrishnan/chopsticks.git
cd chopsticks
uv sync                 # runtime + dev environment
uv sync --extra viz     # add the visualisation extra (pydot) for graph rendering
```

## Usage

### Setting up a basic environment

In a Python shell, run the following:

```python
from chopsticks import chopsticks_v0
env = chopsticks_v0.env()
```

See [`demo.py`](./demo.py) for a script that implements a simple random policy to interact with
the environment.

## Solving the game / reproducing the analysis

The package ships an exact solver that enumerates every reachable position and labels it by
retrograde analysis (backward induction). Chopsticks, as defined by the rules above, is a
**draw** with optimal play.

```bash
uv sync --extra viz
uv run chopsticks-solve --output-dir build/analysis
```

Expected output (a progress bar is also shown on stderr):

```
Chopsticks — retrograde analysis
  start position:        draw (value 0)
  reachable states:      406
  terminal states:       28
  edges:                 1628
  P1 wins / P2 wins:     156 / 156
  drawn states:          94
  unreachable canonical: 44
wrote build/analysis/optimal_graph.dot
wrote build/analysis/optimal_graph.png
wrote build/analysis/p1_winning_p2_all.dot
wrote build/analysis/p1_winning_p2_all.png
wrote build/analysis/p2_winning_p1_all.dot
wrote build/analysis/p2_winning_p1_all.png
```

`optimal_graph` is the optimal moves for both players. `p1_winning_p2_all` is Player 1's
optimal strategy: at every Player 1 turn only value-preserving moves are drawn, while every
Player 2 reply is followed, so the graph shows what Player 1 does no matter how Player 2
plays. `p2_winning_p1_all` is the mirror image for Player 2. `--graph full` adds the
brute-forced full-state graph (~1250 nodes) — pair it with `--format dot`, as a PNG of that
graph takes many minutes to lay out.

### Per-player strategy graphs

The two strategy graphs are rendered by default. To build only one of them, or to pick a
format, select it with `--graph` (repeatable):

```bash
# Player 1's strategy against every Player 2 reply, .dot and .png
uv run chopsticks-solve --graph p1-winning --output-dir build/analysis

# Player 2's strategy against every Player 1 move, .dot only
uv run chopsticks-solve --graph p2-winning --format dot --output-dir build/analysis

# both strategy graphs from a non-standard opening
uv run chopsticks-solve --graph p1-winning --graph p2-winning \
    --start "1,2,1,3,P1" --output-dir build/analysis
```

Reading the graphs: nodes are coloured by game-theoretic value (grey draw, green Player 1
win, red Player 2 win) and shaped by the player to move (`house` Player 1, `invhouse`
Player 2, `box` terminal). Solid edges are tree edges of the traversal; dashed edges rejoin
a position already discovered elsewhere (a transposition or a repetition-rule cycle). In
`p2_winning_p1_all` the root is Player 1 to move, so it fans out over *every* Player 1
opening — including the losing self-move, after which the red subtree is Player 2's forced
win.

Rendering the strategy graphs from Python:

```python
from chopsticks.env.state import Turn
from chopsticks.solver import solve
from chopsticks.solver.viz import strategy_graph_dot, render
from pathlib import Path

solution = solve()
for player in (Turn.P1, Turn.P2):
    graph = strategy_graph_dot(solution, player)
    render(graph, Path(f"{player.name.lower()}_strategy.png"), "png")
```

Solve only, machine-readable, no Graphviz needed:

```bash
uv run chopsticks-solve --no-render --json
```

From Python:

```python
from chopsticks.env.state import ChopsticksState, Turn
from chopsticks.solver import solve, graph_stats

solution = solve()
print(graph_stats(solution))    # {'reachable_states': 406, ..., 'start_outcome': 'draw'}
print(solution.values[ChopsticksState(1, 1, 1, 1, Turn.P1)])   # 0  -> a draw
```

## Testing

Tests are run using [pytest](http://doc.pytest.org/).

```bash
git clone git@github.com:AbhijeetKrishnan/chopsticks.git
cd chopsticks
uv sync
uv run pytest
```

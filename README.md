# Chopsticks AEC Environment

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

This is an implementation of the game of Chopsticks as a [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo) [AEC](https://arxiv.org/abs/2009.13051) game.

## Rules

Chopsticks is a simple, 2-player, perfect information, zero-sum game that only requires both your hands to play.  Each player has a number of "chopsticks" in each hand, denoted by the number of fingers held out. Since players will tend to have only 5 fingers, the number of held chopsticks can only be between $[1, 4]$. Players take turns transferring chopsticks to another hand. Transferring a chopstick involves touching the other player's (or your own) hand with another, thereby adding the chopsticks in your hand to the other, modulo 5. If the total reaches 0, that hand is considered "dead", and is out of play. If a player loses both their hands, then they lose.

## Installation

### Local

```bash
git clone git@github.com:AbhijeetKrishnan/chopsticks.git
cd chopsticks
python3 -m pip install '.'
```

## Usage

### Setting up a basic environment

In a Python shell, run the following:

```python
from chopsticks import chopsticks_v0
env = chopsticks_v0.env()
```

See [`demo.py](./demo.py) for a script that implements a simple random policy to interact with the environment.

## Testing

Tests are run using [pytest](http://doc.pytest.org/).

```bash
git clone hit@github.com:AbhijeetKrishnan/chopsticks.git
cd chopsticks
python3 -m pip install '.[dev]'
pytest
```

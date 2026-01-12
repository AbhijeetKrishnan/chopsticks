import functools
from typing import Any, Dict, Tuple, TypedDict

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from .state import ChopsticksAction, ChopsticksState, Turn


class Observation(TypedDict):
    observation: np.ndarray[Tuple[int, ...], np.dtype[np.int8]]
    action_mask: np.ndarray[Tuple[int, ...], np.dtype[np.int8]]


def env(render_mode: str | None = None) -> AECEnv:
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = ChopsticksEnv(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class ChopsticksEnv(AECEnv):
    """
    Description:
        Implements the game of Chopsticks. The game starts with each player having one chopstick on their left and right
        hands. On their turn, a player can choose to "add" their chopsticks to another hand, either their own or their
        opponent's. The resultant number of chopsticks in the hand will be the sum of the original number of chopsticks
        plus the number of chopsticks added modulo 5. If the number of chopsticks in the hand is exactly 5, the hand is
        "dead" and cannot be used. The game ends when both hands of a player are dead, with the other player declared
        the winner.

    Observation:
        An array of five integers, with the first four representing the number of chopsticks in each player's hands, and
        the last one representing who's turn it is. The first two integers represent the number of chopsticks in player
        0's hands (left, right), and the next two integers represent the number of chopsticks in player 1's hands. The
        integers are in the range [0, 4]. The last integer is {0, 1} representing player 0 and player 1 respectively.

    Reward:
        - +1 for winning the game
        - -1 for losing the game
        -  0 otherwise

    Starting State:
        - Each player has one chopstick on each hand.
        - Player 0 goes first.

    Episode Termination:
        - At least one of the players has both hands dead.
    """

    metadata = {
        "render.modes": ["human", "ansi"],
        "name": "chopsticks_v0",
        "is_parallelizable": True,
        "render_fps": 1,
    }

    def __init__(self, render_mode: str | None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.possible_agents = ["player_0", "player_1"]

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> None:
        self.agents = self.possible_agents[:]
        self.timestep = 0

        self.agent_selection = "player_0"

        self.terminations: Dict[str, bool] = {agent: False for agent in self.agents}
        self.truncations: Dict[str, bool] = {agent: False for agent in self.agents}
        self.rewards: Dict[str, float] = {agent: 0 for agent in self.agents}
        self._cumulative_rewards: Dict[str, float] = {
            agent: 0.0 for agent in self.agents
        }
        self.infos: Dict[str, Dict[str, Any]] = {agent: {} for agent in self.agents}

        self.game_state = ChopsticksState(1, 1, 1, 1, Turn.P1)

        if self.render_mode == "human":
            self.render()

    def _opponent(self, agent: str) -> str:
        return "player_1" if agent == "player_0" else "player_0"

    def step(self, action: int) -> None:
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return None

        # make the move
        self.game_state = self.game_state.transition(ChopsticksAction(action))

        # check termination conditions
        terminated = self.game_state.is_terminal()
        self.terminations = {agent: terminated for agent in self.agents}
        self.rewards = {
            agent: 1 if self.terminations[self._opponent(agent)] else 0
            for agent in self.agents
        }

        # check truncation conditions
        self.truncations = {agent: self.terminations[agent] for agent in self.agents}

        # get infos
        self.infos = {agent: {} for agent in self.agents}

        self._accumulate_rewards()

        # update agent selection
        self.agent_selection = self._opponent(self.agent_selection)

    def observe(self, agent: str) -> Observation:
        observation = np.array(
            [
                self.game_state.p1_min,
                self.game_state.p1_max,
                self.game_state.p2_min,
                self.game_state.p2_max,
                self.game_state.turn.value,
            ],
            dtype=np.int8,
        )
        legal_moves = list(
            map(lambda action: action.value, self.game_state.legal_moves())
        )
        action_mask = np.zeros(len(ChopsticksAction), np.int8)
        action_mask[list(legal_moves)] = 1
        return {"observation": observation, "action_mask": action_mask}

    def render(self) -> None:
        print(self.game_state)

    def close(self) -> None:
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:  # type: ignore[override]
        return spaces.Dict(
            {
                "observation": spaces.MultiDiscrete([5, 5, 5, 5, 2], dtype=np.int8),
                "action_mask": spaces.MultiBinary(len(ChopsticksAction)),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete[np.integer[Any]]:  # type: ignore[override]
        return spaces.Discrete(len(ChopsticksAction))

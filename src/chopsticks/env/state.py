from enum import IntEnum
from typing import Generator, Set, Tuple


class ChopsticksAction(IntEnum):
    MIN_TO_MIN = 0
    MIN_TO_MAX = 1
    MAX_TO_MIN = 2
    MAX_TO_MAX = 3
    MIN_TO_SELF = 4
    MAX_TO_SELF = 5

    def __str__(self) -> str:
        match self.name:
            case "MIN_TO_MIN":
                return "m2m"
            case "MIN_TO_MAX":
                return "m2M"
            case "MAX_TO_MIN":
                return "M2m"
            case "MAX_TO_MAX":
                return "M2M"
            case "MIN_TO_SELF":
                return "m2s"
            case "MAX_TO_SELF":
                return "M2s"
        return self.name


class Turn(IntEnum):
    P1 = 0
    P2 = 1


class ChopsticksState:
    def __init__(
        self,
        p1_min: int,
        p1_max: int,
        p2_min: int,
        p2_max: int,
        turn: Turn,
    ) -> None:
        self.p1_min = p1_min
        self.p1_max = p1_max
        self.p2_min = p2_min
        self.p2_max = p2_max
        self.turn = turn
        self.is_repeated = False
        self.history: Set[Tuple[int, int, int, int, Turn]] = set()

    def __hash__(self) -> int:
        return hash(
            (
                self.p1_min,
                self.p1_max,
                self.p2_min,
                self.p2_max,
                self.turn,
                self.is_repeated,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChopsticksState):
            return NotImplemented
        return (
            self.p1_min == other.p1_min
            and self.p1_max == other.p1_max
            and self.p2_min == other.p2_min
            and self.p2_max == other.p2_max
            and self.turn == other.turn
            and self.is_repeated == other.is_repeated
        )

    def __str__(self) -> str:
        return f"p1=({self.p1_min}, {self.p1_max}), p2=({self.p2_min}, {self.p2_max}), t={self.turn}, r={self.is_repeated}"

    def __repr__(self) -> str:
        return f"ChopsticksState(p1_min={self.p1_min}, p1_max={self.p1_max}, p2_min={self.p2_min}, p2_max={self.p2_max}, turn={self.turn}, repeated={self.is_repeated})"

    def transition(self, action: ChopsticksAction) -> "ChopsticksState":
        # assign p (player whose turn it is) and o (other player) based on whose turn it is
        if self.turn == Turn.P1:
            p_min = self.p1_min
            p_max = self.p1_max
            o_min = self.p2_min
            o_max = self.p2_max
        else:
            p_min = self.p2_min
            p_max = self.p2_max
            o_min = self.p1_min
            o_max = self.p1_max

        # update the hands based on the action
        match action:
            case ChopsticksAction.MIN_TO_MIN:
                o_min = (o_min + p_min) % 5
            case ChopsticksAction.MIN_TO_MAX:
                o_max = (o_max + p_min) % 5
            case ChopsticksAction.MAX_TO_MIN:
                o_min = (o_min + p_max) % 5
            case ChopsticksAction.MAX_TO_MAX:
                o_max = (o_max + p_max) % 5
            case ChopsticksAction.MIN_TO_SELF:
                p_max = (p_max + p_min) % 5
            case ChopsticksAction.MAX_TO_SELF:
                p_min = (p_min + p_max) % 5

        # update min and max hands
        p_min, p_max = sorted([p_min, p_max])
        o_min, o_max = sorted([o_min, o_max])

        # return the new state
        if self.turn == Turn.P1:
            new_state = ChopsticksState(
                p_min,
                p_max,
                o_min,
                o_max,
                Turn.P2,
            )
        else:
            new_state = ChopsticksState(
                o_min,
                o_max,
                p_min,
                p_max,
                Turn.P1,
            )

        # invariant: a state's history is the set of all states visited *prior* to the current state
        new_state.history = self.history | {
            (self.p1_min, self.p1_max, self.p2_min, self.p2_max, self.turn)
        }
        new_state.is_repeated = (
            new_state.p1_min,
            new_state.p1_max,
            new_state.p2_min,
            new_state.p2_max,
            new_state.turn,
        ) in new_state.history
        return new_state

    def is_terminal(self) -> bool:
        p1_dead = self.p1_min == 0 and self.p1_max == 0
        p2_dead = self.p2_min == 0 and self.p2_max == 0
        return p1_dead or p2_dead or self.is_repeated

    def legal_moves(self) -> Generator[ChopsticksAction, None, None]:
        if self.turn == Turn.P1:
            if self.p1_min != 0 and self.p2_min != 0:
                yield ChopsticksAction.MIN_TO_MIN
            if self.p1_min != 0 and self.p2_max != 0:
                yield ChopsticksAction.MIN_TO_MAX
            if self.p1_max != 0 and self.p2_min != 0:
                yield ChopsticksAction.MAX_TO_MIN
            if self.p1_max != 0 and self.p2_max != 0:
                yield ChopsticksAction.MAX_TO_MAX
            if self.p1_max != 0 and self.p1_min != 0:
                yield ChopsticksAction.MIN_TO_SELF
                yield ChopsticksAction.MAX_TO_SELF
        else:
            if self.p2_min != 0 and self.p1_min != 0:
                yield ChopsticksAction.MIN_TO_MIN
            if self.p2_min != 0 and self.p1_max != 0:
                yield ChopsticksAction.MIN_TO_MAX
            if self.p2_max != 0 and self.p1_min != 0:
                yield ChopsticksAction.MAX_TO_MIN
            if self.p2_max != 0 and self.p1_max != 0:
                yield ChopsticksAction.MAX_TO_MAX
            if self.p2_max != 0 and self.p2_min != 0:
                yield ChopsticksAction.MIN_TO_SELF
                yield ChopsticksAction.MAX_TO_SELF

    def winner(self) -> Turn | None:
        if self.is_terminal():
            if self.p1_min == 0 and self.p1_max == 0:
                return Turn.P2
            elif self.p2_min == 0 and self.p2_max == 0:
                return Turn.P1
            else:
                return None
        else:
            raise ValueError("Game is not over yet")

from enum import IntEnum
from typing import Generator


class ChopsticksAction(IntEnum):
    LEFT_TO_LEFT = 0
    LEFT_TO_RIGHT = 1
    RIGHT_TO_LEFT = 2
    RIGHT_TO_RIGHT = 3
    LEFT_TO_SELF_RIGHT = 4
    RIGHT_TO_SELF_LEFT = 5


class Turn(IntEnum):
    P1 = 0
    P2 = 1


class ChopsticksState:
    def __init__(
        self,
        p1_left: int,
        p1_right: int,
        p2_left: int,
        p2_right: int,
        turn: Turn,
    ) -> None:
        self.p1_left = p1_left
        self.p1_right = p1_right
        self.p2_left = p2_left
        self.p2_right = p2_right
        self.turn = turn

    @property
    def canonical(self) -> "ChopsticksState":
        return ChopsticksState(
            min(self.p1_left, self.p1_right),
            max(self.p1_left, self.p1_right),
            min(self.p2_left, self.p2_right),
            max(self.p2_left, self.p2_right),
            self.turn,
        )

    def __hash__(self) -> int:
        return hash(
            (
                min(self.p1_left, self.p1_right),
                max(self.p1_left, self.p1_right),
                min(self.p2_left, self.p2_right),
                max(self.p2_left, self.p2_right),
                self.turn,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChopsticksState):
            return NotImplemented
        return (
            self.p1_left == other.p1_left
            and self.p1_right == other.p1_right
            and self.p2_left == other.p2_left
            and self.p2_right == other.p2_right
            and self.turn == other.turn
        )

    def __str__(self) -> str:
        return f"(l1={self.p1_left}, r1={self.p1_right}, l2={self.p2_left}, r2={self.p2_right}, t={self.turn})"

    def __repr__(self) -> str:
        return f"ChopsticksState(p1_left={self.p1_left}, p1_right={self.p1_right}, p2_left={self.p2_left}, p2_right={self.p2_right}, turn={self.turn})"

    def transition(self, action: ChopsticksAction) -> "ChopsticksState":
        # assign p1 and p2 based on whose turn it is
        if self.turn == Turn.P1:
            p1_left = self.p1_left
            p1_right = self.p1_right
            p2_left = self.p2_left
            p2_right = self.p2_right
        else:
            p1_left = self.p2_left
            p1_right = self.p2_right
            p2_left = self.p1_left
            p2_right = self.p1_right

        # update the hands based on the action
        match action:
            case ChopsticksAction.LEFT_TO_LEFT:
                p2_left = (p2_left + p1_left) % 5
            case ChopsticksAction.LEFT_TO_RIGHT:
                p2_right = (p2_right + p1_left) % 5
            case ChopsticksAction.RIGHT_TO_LEFT:
                p2_left = (p2_left + p1_right) % 5
            case ChopsticksAction.RIGHT_TO_RIGHT:
                p2_right = (p2_right + p1_right) % 5
            case ChopsticksAction.LEFT_TO_SELF_RIGHT:
                p1_right = (p1_right + p1_left) % 5
            case ChopsticksAction.RIGHT_TO_SELF_LEFT:
                p1_left = (p1_left + p1_right) % 5

        # return the new state
        if self.turn == Turn.P1:
            new_state = ChopsticksState(
                p1_left,
                p1_right,
                p2_left,
                p2_right,
                Turn.P2,
            )
        else:
            new_state = ChopsticksState(
                p2_left,
                p2_right,
                p1_left,
                p1_right,
                Turn.P1,
            )
        return new_state.canonical

    def is_terminal(self) -> bool:
        return (self.p1_left == 0 and self.p1_right == 0) or (
            self.p2_left == 0 and self.p2_right == 0
        )

    def legal_moves(self) -> Generator[ChopsticksAction, None, None]:
        if self.turn == Turn.P1:
            if self.p1_left != 0 and self.p2_left != 0:
                yield ChopsticksAction.LEFT_TO_LEFT
            if self.p1_left != 0 and self.p2_right != 0:
                yield ChopsticksAction.LEFT_TO_RIGHT
            if self.p1_right != 0 and self.p2_left != 0:
                yield ChopsticksAction.RIGHT_TO_LEFT
            if self.p1_right != 0 and self.p2_right != 0:
                yield ChopsticksAction.RIGHT_TO_RIGHT
            if self.p1_right != 0 and self.p1_left != 0:
                yield ChopsticksAction.RIGHT_TO_SELF_LEFT
                yield ChopsticksAction.LEFT_TO_SELF_RIGHT
        else:
            if self.p2_left != 0 and self.p1_left != 0:
                yield ChopsticksAction.LEFT_TO_LEFT
            if self.p2_left != 0 and self.p1_right != 0:
                yield ChopsticksAction.LEFT_TO_RIGHT
            if self.p2_right != 0 and self.p1_left != 0:
                yield ChopsticksAction.RIGHT_TO_LEFT
            if self.p2_right != 0 and self.p1_right != 0:
                yield ChopsticksAction.RIGHT_TO_RIGHT
            if self.p2_right != 0 and self.p2_left != 0:
                yield ChopsticksAction.RIGHT_TO_SELF_LEFT
                yield ChopsticksAction.LEFT_TO_SELF_RIGHT

    def winner(self) -> Turn:
        if self.p1_left == 0 and self.p1_right == 0:
            return Turn.P2
        elif self.p2_left == 0 and self.p2_right == 0:
            return Turn.P1
        else:
            raise ValueError("Game is not over yet")

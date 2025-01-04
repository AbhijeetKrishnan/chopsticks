from chopsticks.env.state import ChopsticksState, Turn


def test_is_terminal() -> None:
    state = ChopsticksState(4, 1, 0, 0, Turn.P2)
    assert state.is_terminal() is True

import pettingzoo
import pettingzoo.test
import pytest
from chopsticks import chopsticks_v0

from typing import Generator


@pytest.fixture(scope="function")
def env() -> Generator[chopsticks_v0.env, None, None]:
    env = chopsticks_v0.env()
    env.reset()
    yield env
    env.close()


def test_api(env: chopsticks_v0.env) -> None:
    pettingzoo.test.api_test(env)


def test_performance_benchmark(env: chopsticks_v0.env) -> None:
    pettingzoo.test.performance_benchmark(env)


def test_seed(env: chopsticks_v0.env) -> None:
    pettingzoo.test.seed_test(chopsticks_v0.env)

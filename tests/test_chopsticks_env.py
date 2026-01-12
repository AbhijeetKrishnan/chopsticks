from typing import Generator

import pettingzoo
import pettingzoo.test
import pytest
from pettingzoo import AECEnv

from chopsticks import chopsticks_v0


@pytest.fixture(scope="function")
def env() -> Generator[AECEnv, None, None]:
    env = chopsticks_v0.env()
    env.reset()
    yield env
    env.close()


def test_api(env: AECEnv) -> None:
    pettingzoo.test.api_test(env)


def test_performance_benchmark(env: AECEnv) -> None:
    pettingzoo.test.performance_benchmark(env)


def test_seed(env: AECEnv) -> None:
    pettingzoo.test.seed_test(chopsticks_v0.env)

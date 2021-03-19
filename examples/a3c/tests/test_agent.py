import os

import gym
import pytest

from examples.a3c.a3c_agent import A3CAgent, CNNPolicy
from examples.a3c.a3c_worker_environment import run_environment_to_done


@pytest.mark.parametrize("env_name", ["FFAI-5-v3"])
def test_ac_agent(env_name):
    env = gym.make(env_name)

    ac_agent = CNNPolicy(env, hidden_nodes=10, kernels=[10, 10])

    agent = A3CAgent("trainee", env_name=env_name, policy=ac_agent)

    run_environment_to_done(env, agent)

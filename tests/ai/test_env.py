from tests.util import *
import gym
from tests.performance.run_env import get_random_action_from_env


def test_observation_ranges():
    def find_first_index(array_, value_):
        indices = (array_ == value_).nonzero()
        return [x[0] for x in indices]

    env = gym.make("botbowl-v3")
    rnd = np.random.RandomState(np.random.randint(0, 2 ** 16))

    for _ in range(10):
        obs = env.reset()

        done = False
        while not done:
            for layer_name, array in obs['board'].items():
                max_val = np.max(array)
                assert max_val <= 1.0, \
                    f"obs['board']['{layer_name}'][{find_first_index(array, max_val)}] is too high ({max_val})"
                min_val = np.min(array)
                assert min_val >= 0.0, \
                    f"obs['board']['{layer_name}'][{find_first_index(array, min_val)}] is too low ({min_val})"

            for obs_key in ['state', 'procedures', 'available-action-types']:
                for key_name, value in obs[obs_key].items():
                    assert 0.0 <= value <= 1.0, \
                        f"obs['{obs_key}']['{key_name}'] is too {'high' if value>1.0 else 'low'}: {value}"

            obs, _, done, _ = env.step(get_random_action_from_env(env, rnd))
    env.close()

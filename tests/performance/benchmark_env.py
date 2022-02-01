from botbowl import RandomBot
from botbowl.ai.env import BotBowlEnv
import numpy as np
from time import perf_counter as timer

def run_env():
    env = BotBowlEnv()
    steps = 0
    for _ in range(2):
        env.reset()

        done = False
        _, _, mask = env.get_state()

        while not done:
            steps += 1
            aa = np.where(mask > 0.0)[0]
            action_idx = np.random.choice(aa, 1)[0]
            obs, reward, done, info = env.step(action_idx)
            mask = info['action_mask']

    return steps

def run_game():
    env = BotBowlEnv()
    steps = 0
    random_bot = RandomBot(name='')

    for _ in range(2):
        env.reset()
        game = env.game
        while not game.state.game_over:
            game.step(random_bot.act(game))
            steps += 1
    return steps


if __name__ == "__main__":
    start_time = timer()
    steps = run_env()
    elapsed_time = timer() - start_time


    print(f"took {elapsed_time:.2f} seconds, steps={steps}, step_rate = {elapsed_time/steps:.5f}")
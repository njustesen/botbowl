import pickle
from pytest import set_trace
import numpy as np
import matplotlib.pyplot as plt
import os


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def summerize_training(name, reports):
    print(name)
    total_time = np.array([r.elapsed_time for r in reports]).sum()
    total_time_steps = np.array([r.time_steps for r in reports]).sum()
    total_episdoes = np.array([r.episodes for r in reports]).sum()
    print(f"{total_time_steps} steps in {total_time}. rate={total_time_steps/total_time:.1f}")
    print(f"episodes = {total_episdoes}")
    print(f"step per episode = {total_time_steps/total_episdoes:.1f}")

# Config
rolling_width = 50

x_values = ["time_steps", "elapsed_time"]
y_values = ["total_reward", "td_rate", "win_rate"]

# Get content of logs.
log_dir = "logs/FFAI-3-v3/"
logs = os.listdir(log_dir)
logs = [os.path.join(log_dir, log) for log in logs]

num_x_values = len(x_values)
num_y_values = len(y_values)

fig, axs = plt.subplots(num_x_values, num_y_values, figsize=(14, 7))

for log_path in logs:
    reports = pickle.load(open(log_path, "rb"))
    name = log_path.split("/")[-1]
    name = name.split(".")[0]
    summerize_training(name, reports)

    for i, x_name in enumerate(x_values):
        x = moving_average(np.cumsum(np.array([getattr(r, x_name) for r in reports])), rolling_width)

        for j, y_name in enumerate( y_values):
            y = moving_average(np.array([getattr(r, y_name) for r in reports]), rolling_width)

            ax = axs[i][j]

            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.plot(x, y, label=name)

            if j == 0:
                ax.set_ylabel(x_name)
            if i == 0:
                ax.set_title(y_name)

        plt.legend()

fig.tight_layout()


plt.show()

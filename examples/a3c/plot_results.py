import pickle
from pytest import set_trace
import numpy as np
import matplotlib.pyplot as plt
import os


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def summerize_training(reports):
    total_time = np.array([r.elapsed_time for r in reports]).sum()
    total_time_steps = np.array([r.time_steps for r in reports]).sum()
    total_episdoes = np.array([r.episodes for r in reports]).sum()

    return {#"steps": total_time_steps,
            "step_rate": total_time_steps/total_time,
            #"time": total_time,
            "steps per episode": total_time_steps/total_episdoes,
            }


# Config
rolling_width = 25

x_values = ["time_steps", "elapsed_time"]
y_values = ["total_reward", "td_rate", "win_rate"]

# Get content of logs.
log_dir = "logs/FFAI-1-v3/"
logs = os.listdir(log_dir)
logs = [os.path.join(log_dir, log) for log in logs]

names = [path.split("/")[-1].split(".")[0] for path in logs]


num_x_values = len(x_values)
num_y_values = len(y_values)

fig, axs = plt.subplots(num_y_values, num_x_values, figsize=(3*num_x_values, 3*num_y_values))

summary = {}

for name, log_path in zip(names, logs):
    reports = pickle.load(open(log_path, "rb"))
    summary[name] = summerize_training(reports)

    for i, x_name in enumerate(x_values):
        x = moving_average(np.cumsum(np.array([getattr(r, x_name) for r in reports])), rolling_width)

        for j, y_name in enumerate( y_values):
            y = moving_average(np.array([getattr(r, y_name) for r in reports]), rolling_width)

            ax = axs[j][i]

            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.plot(x, y, label=name)

            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)

            #if i == 0:
            #    ax.xlim(right=)
        plt.legend()

# Print a nice table
keys = [key for key in summary[names[0]]]
column_headers = ", ".join(keys)

len_first_col = max([len(name) for name in names])
print(" "*len_first_col + column_headers)

for name in summary:
    print(name, end=" "*(1+len_first_col-len(name)))
    for key in summary[name]:
        print(f"{summary[name][key]:.1f}",end=" "*5)
    print(" ")


fig.tight_layout()


plt.show()

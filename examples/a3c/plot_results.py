
import pickle
from pytest import set_trace
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


results = pickle.load( open("A3c_log_2.p","rb") )

rolling_width = 50


reports = [r[0] for r in results]

steps = moving_average(np.cumsum(np.array([r.time_steps for r in reports])), rolling_width)

win_rate = moving_average(np.array([r.win_rate for r in reports]), rolling_width)
td_rate = moving_average(np.array([r.own_td/r.episodes for r in reports]), rolling_width)
reward = moving_average(np.array([r.total_reward/r.episodes for r in reports]), rolling_width)
memory = moving_average(np.array([r.memory_usage for r in reports])/1000000, rolling_width)


fig, axs = plt.subplots(1, 4, figsize=(4*4, 5))
axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[0].plot(steps, win_rate)
axs[0].set_title('Win rate')
#axs[0].set_xlim(left=0)

axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[1].plot(steps, td_rate, label="Learner")
axs[1].set_title('TD/Episode')
#axs[1].set_ylim(bottom=0.0)
#axs[1].set_xlim(left=0)

axs[2].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[2].plot(steps, reward)
axs[2].set_title('Reward')
#axs[2].set_yticks(np.arange(0, 1.001, step=0.1))
#axs[2].set_xlim(left=0)

axs[3].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[3].plot(steps, memory)
axs[3].set_title('Memory')

fig.tight_layout()
plt.show()
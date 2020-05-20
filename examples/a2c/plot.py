import matplotlib.pyplot as plt
import csv
import numpy as np

model_name = 'FFAI-v2-ppcg'

log_updates = []
log_episode = []
log_steps = []
log_win_rate = []
log_td_rate = []
log_mean_reward = []
log_difficulty = []

with open('logs/' + model_name + '.dat') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    all_updates, all_episodes, all_steps, win_rate, td_rate, mean_reward, difficulty
    for row in csv_reader:
        log_updates.append(row[0])
        log_episode.append(row[1])
        log_steps.append(row[2])
        log_win_rate.append(row[3])
        log_td_rate.append(row[4])
        log_mean_reward.append(row[5])
        #log_difficulty.append(row[6])
        line_count += 1
    print(f'Processed {line_count} lines.')

fig, axs = plt.subplots(1, 4, figsize=(16, 5))
axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[0].plot(log_steps, log_mean_reward)
axs[0].set_title('Reward')
#axs[0].set_ylim(bottom=0.0)
axs[0].set_xlim(left=0)
axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[1].plot(log_steps, log_td_rate)
axs[1].set_title('TD/Episode')
axs[1].set_ylim(bottom=0.0)
axs[1].set_xlim(left=0)
axs[2].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[2].plot(log_steps, log_win_rate)
axs[2].set_title('Win rate')
axs[2].set_yticks(np.arange(0, 1.001, step=0.1))
axs[2].set_xlim(left=0)
axs[3].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
#axs[3].plot(log_steps, log_difficulty)
#axs[3].set_title('Difficulty')
#axs[3].set_yticks(np.arange(0, 1.001, step=0.1))
#axs[3].set_xlim(left=0)
fig.tight_layout()
fig.savefig("plots/"+model_name+"_enhanced.png")
plt.close('all')

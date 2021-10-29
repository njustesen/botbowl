import matplotlib.pyplot as plt
import numpy as np

plot_name = "botbowl-1"
exp_id_nopf = '7f556c38-7853-11eb-b93b-faffc23fefdb.dat'
exp_id_pf = '129055f2-7846-11eb-a07b-faffc23fefdb.dat'
window_size = 2

plot_name = "botbowl-3"
exp_id_nopf = '48ed35ea-79b7-11eb-ba58-19faed4b7487.dat'
exp_id_pf = '4cd8c832-794d-11eb-b2ed-19faed4b7487.dat'
window_size = 20

#plot_name = "botbowl-5"
#exp_id_nopf = 'f1e2354e-79d8-11eb-bb13-19faed4b7487.dat'
#exp_id_pf = '70310ce8-7a80-11eb-8061-19faed4b7487.dat'
#window_size = 100

logfile_nopf = f"logs/logs-pf/{plot_name}/nopf/{exp_id_nopf}"
logfile_pf = f"logs/logs-pf/{plot_name}/pf/{exp_id_pf}"
fig, axs = plt.subplots(1, 3, figsize=(4 * 3, 5))

axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
axs[0].set_title('Reward')
axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
axs[1].set_title('TD/Episode')
#axs[1].set_ylim(bottom=0.0)
#axs[1].set_xlim(left=0)
axs[2].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
axs[2].set_title('Win rate')
#axs[2].set_yticks(np.arange(0, 1.001, step=0.1))
#axs[2].set_xlim(left=0)


def plot(log_steps, log_mean_reward, log_td_rate, log_win_rate, label):
    axs[0].plot(log_steps, log_mean_reward, label=label)
    axs[1].plot(log_steps, log_td_rate, label=label)
    axs[2].plot(log_steps, log_win_rate, label=label)


def load_data(logfile, window_size):
    data = []
    with open(logfile, 'r') as file:
        for line in file:
            data.append([float(v) for v in line.strip().split(", ")])
    cols = []
    for i in range(len(data[0])):
        col = []
        vals = []
        for v in np.array(data)[:, i]:
            vals.append(v)
            if len(vals) == window_size:
                col.append(np.average(vals))
                vals.clear()
        cols.append(col)
    return np.array(cols).transpose()


data = load_data(logfile_nopf, window_size=window_size)
plot(log_steps=data[:, 2], log_mean_reward=data[:, 6], log_td_rate=data[:, 4], log_win_rate=data[:, 3], label="No Pathfinding")
data = load_data(logfile_pf, window_size=window_size)
plot(log_steps=data[:, 2], log_mean_reward=data[:, 6], log_td_rate=data[:, 4], log_win_rate=data[:, 3], label="Pathfinding")

# axs[0].legend(['Baseline', 'Pathfinding'])
# axs[1].legend(['Baseline', 'Pathfinding'])
axs[2].legend(['Baseline', 'Pathfinding'])

fig.tight_layout()
plotfile = f"plots/{plot_name}.png"
fig.savefig(plotfile)
plt.close('all')

import psutil
import os


class EndOfGameReport:
    def __init__(self, time_steps=0, reward=0, game=None, empty_report=False):
        self.time_steps = time_steps
        self.td_rate = game.state.home_team.state.score if game is not None else 0
        self.opp_td_rate = game.state.away_team.state.score if game is not None else 0
        self.total_reward = reward
        self.episodes = 0 if empty_report else 1
        self.win_rate = 1.0 if self.td_rate > self.opp_td_rate else 0.5 if self.td_rate == self.opp_td_rate else 0
        self.memory_usage = get_process_memory_usage()

    def merge(self, other):
        self.time_steps += other.time_steps
        self.td_rate = calculate_new_average_rate(self.td_rate, other.td_rate, self.episodes, other.episodes)
        self.opp_td_rate = calculate_new_average_rate(self.opp_td_rate, other.opp_td_rate, self.episodes,
                                                      other.episodes)
        self.total_reward = calculate_new_average_rate(self.total_reward, other.total_reward, self.episodes,
                                                       other.episodes)
        self.win_rate = calculate_new_average_rate(self.win_rate, other.win_rate, self.episodes, other.episodes)

        self.memory_usage = calculate_new_average_rate(self.memory_usage, other.memory_usage, self.episodes,
                                                       other.episodes)

        self.episodes += other.episodes

    def get_dict_repr(self):
        return {"Win rate": f"{self.win_rate:.2f}",
                "TD rate": f"{self.td_rate:.2f}",
                "TD rate opp": f"{self.opp_td_rate:.2f}",
                "Mean reward": f"{self.total_reward:.3f}"}

    def __repr__(self):
        s = ""
        for key, val in self.get_dict_repr().items():
            s += f"{key}: {val}, "

        return s[:-2]  # Remove trailing colon and blank


def calculate_new_average_rate(r1, r2, n1, n2):
    return (r1 * n1 + r2 * n2) / (n1 + n2) if n1+n2 > 0 else 0

def get_process_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss



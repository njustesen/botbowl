"""
==========================
Author: Niels Justesen
Year: 2019
==========================
This module contains a competition class to handle a competition between two bots.
"""
from copy import deepcopy
import numpy as np
from ffai.ai.registry import make_bot
from ffai.core.game import Game
from ffai.core.table import CasualtyType
from ffai.core.model import Agent
from multiprocessing import Process, Pipe
from ffai.core.load import get_team, get_rule_set, get_config
import time 
import signal
from contextlib import contextmanager
import pickle
from ffai.core.util import get_data_path


class TeamResult:

    def __init__(self, game, name, team):
        self.name = name
        self.tds = team.state.score
        self.cas = len(game.get_casualties(team))
        self.cas_inflicted = len(game.get_casualties(game.get_opp_team(team)))
        self.kills = len([player for player in game.get_casualties(team) if player.state.casualty_type == CasualtyType.DEAD])
        self.kills_inflicted = len([player for player in game.get_casualties(game.get_opp_team(team)) if player.state.casualty_type == CasualtyType.DEAD])

    def print(self):
        print("-- {}".format(self.name))
        print("TDs: {}".format(self.tds))
        print("Cas: {}".format(self.cas))
        print("Cas inflicted: {}".format(self.cas_inflicted))
        print("Kills: {}".format(self.kills_inflicted))
        print("Kills inflicted: {}".format(self.kills_inflicted))


class GameResult:

    def __init__(self, game, crashed=False):
        self.home_agent_name = game.home_agent.name
        self.away_agent_name = game.away_agent.name
        self.home_result = TeamResult(game, game.home_agent.name, game.state.home_team)
        self.away_result = TeamResult(game, game.away_agent.name, game.state.away_team)
        self.draw = self.home_result.tds == self.away_result.tds
        self.winner = game.winner()
        self.tds = self.home_result.tds + self.away_result.tds
        self.cas_inflicted = self.home_result.cas_inflicted == self.away_result.cas_inflicted
        self.kills = self.home_result.kills_inflicted == self.away_result.kills_inflicted
        self.home_time_violation = game.state.home_team.state.time_violation
        self.away_time_violation = game.state.away_team.state.time_violation
        self.timed_out = game.timed_out() and not self.winner
        self.crashed = crashed
        if crashed:
            if game.actor is not None:
                self.winner = game.other_agent(game.actor)

    def print(self):
        print("############ GAME RESULTS ###########")
        print("Final score:")
        print("{} {} - {} {}".format(self.away_agent_name, self.away_result.tds, self.home_result.tds, self.home_agent_name))
        print("Casualties inflicted:")
        print("{} {} - {} {}".format(self.away_agent_name, self.away_result.cas_inflicted, self.home_result.cas_inflicted, self.home_agent_name))
        print("Kills inflicted:")
        print("{} {} - {} {}".format(self.away_agent_name, self.away_result.kills_inflicted, self.home_result.kills_inflicted, self.home_agent_name))
        print("Time violation:")
        print("{} {} - {} {}".format(self.away_agent_name, self.away_time_violation, self.home_time_violation, self.home_agent_name))
        print("Result:")
        if self.winner is not None:
            print(f"Winner: {self.winner.name}")
        elif self.timed_out:
            print("Timed out with no winner - no winner")
        elif self.crashed:
            print("Game crashed - no winner")
        else:
            print("Draw")
        print("#####################################")


class CompetitionResult:

    def __init__(self, competitor_a_id, competitor_b_id, game_results):
        self.game_results = game_results
        self.competitor_a_name = competitor_a_id
        self.competitor_b_name = competitor_b_id
        self.wins = {
            competitor_a_id: np.sum([1 if result.winner is not None and result.winner.name == competitor_a_id else 0 for result in game_results]),
            competitor_b_id: np.sum([1 if result.winner is not None and result.winner.name == competitor_b_id else 0 for result in game_results])
        }
        self.decided = self.wins[competitor_a_id] + self.wins[competitor_b_id]
        self.undecided = len(game_results) - self.decided
        self.crashes = len([result for result in game_results if result.crashed])
        self.timed_out = len([result for result in game_results if result.timed_out])
        self.tds = {
            competitor_a_id: [result.home_result.tds if result.home_agent_name == competitor_a_id else result.away_result.tds for result in game_results],
            competitor_b_id: [result.home_result.tds if result.home_agent_name == competitor_b_id else result.away_result.tds for result in game_results]
        }
        self.cas_inflicted = {
            competitor_a_id: [result.home_result.cas_inflicted if result.home_agent_name == competitor_a_id else result.away_result.cas_inflicted for result in game_results],
            competitor_b_id: [result.home_result.cas_inflicted if result.home_agent_name == competitor_b_id else result.away_result.cas_inflicted for result in game_results]
        }
        self.kills_inflicted = {
            competitor_a_id: [result.home_result.kills_inflicted if result.home_agent_name == competitor_a_id else result.away_result.kills_inflicted for result in game_results],
            competitor_b_id: [result.home_result.kills_inflicted if result.home_agent_name == competitor_b_id else result.away_result.kills_inflicted for result in game_results]
        }

    def print(self):
        print("%%%%%%%%% COMPETITION RESULTS %%%%%%%%%")
        print("Wins:")
        print("{}: {}".format(self.competitor_a_name, self.wins[self.competitor_a_name]))
        print("{}: {}".format(self.competitor_b_name, self.wins[self.competitor_b_name]))
        print(f"Draws: {self.undecided}")
        print(f"Crashes: {self.crashes}")
        print(f"Timed out: {self.timed_out}")
        print("TDs:")
        print("{}: {} (avg. {})".format(self.competitor_a_name, np.sum(self.tds[self.competitor_a_name]), np.mean(self.tds[self.competitor_a_name])))
        print("{}: {} (avg. {})".format(self.competitor_b_name, np.sum(self.tds[self.competitor_b_name]), np.mean(self.tds[self.competitor_b_name])))
        print("Casualties inflicted:")
        print("{}: {} (avg. {})".format(self.competitor_a_name, np.sum(self.cas_inflicted[self.competitor_a_name]), np.mean(self.cas_inflicted[self.competitor_a_name])))
        print("{}: {} (avg. {})".format(self.competitor_b_name, np.sum(self.cas_inflicted[self.competitor_b_name]), np.mean(self.cas_inflicted[self.competitor_b_name])))
        print("Kills inflicted:")
        print("{}: {} (avg. {})".format(self.competitor_a_name, np.sum(self.kills_inflicted[self.competitor_a_name]), np.mean(self.kills_inflicted[self.competitor_a_name])))
        print("{}: {} (avg. {})".format(self.competitor_b_name, np.sum(self.kills_inflicted[self.competitor_b_name]), np.mean(self.kills_inflicted[self.competitor_b_name])))
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

class TimeoutException(Exception): pass


class Competition:

    def __init__(self, name, competitor_a_team_id, competitor_b_team_id, competitor_a_id, competitor_b_id, config):
        self.name = name
        self.competitor_a_id = competitor_a_id
        self.competitor_b_id = competitor_b_id
        self.config = config
        self.config.competition_mode = True
        self.ruleset = get_rule_set(self.config.ruleset)
        self.competitor_a_team = get_team(competitor_a_team_id, self.ruleset)
        self.competitor_b_team = get_team(competitor_b_team_id, self.ruleset)

    def run(self, num_games):
        results = []
        for i in range(num_games):
            print(f"Setting up bots for game {i+1}")
            competitor_a = make_bot(self.competitor_a_id)
            competitor_b = make_bot(self.competitor_b_id)
            # TODO: Time limit
            print(f"Starting game {i+1}")
            match_id = self.name + "_" + str(i+1)
            if i%2==0:
                result = self._run_match(match_id, home_team=self.competitor_a_team, away_team=self.competitor_b_team, home_agent=competitor_a, away_agent=competitor_b)
            else:
                result = self._run_match(match_id, home_team=self.competitor_b_team, away_team=self.competitor_a_team, home_agent=competitor_b, away_agent=competitor_a)
            results.append(result)
            result.print()
        return CompetitionResult(self.competitor_a_id, self.competitor_b_id, results)

    @contextmanager
    def time_limit(self, seconds):
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

    def _run_match(self, match_id, home_team, away_team, home_agent, away_agent):
        game = Game(match_id, home_team=deepcopy(home_team), away_team=deepcopy(away_team), home_agent=home_agent, away_agent=away_agent, config=self.config, auto_save=True)
        game.config.fast_mode = True
        game.config.competition_mode = True
        try:
            with self.time_limit(self.config.time_limits.game):
                game.init()
        except TimeoutException:
            print("Game timed out!")
            # Load from autosave
            data_path = get_data_path(rel_path=f"auto/{game.game_id}")
            game = pickle.load(open(data_path, "rb"))
            game.end_time = time.time()
            return GameResult(game)
        except Exception as e:
            print("Game crashed!", e)
            return GameResult(game, crashed=True)
        assert game.state.game_over
        return GameResult(game)
        
    
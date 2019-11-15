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
from ffai.core.table import CasualtyType, OutcomeType
from ffai.core.load import load_team_by_name, load_rule_set, load_config
import time
import pickle
from ffai.core.util import get_data_path
import stopit
import uuid

class TeamResult:

    def __init__(self, game, name, team, winner, crashed, timed_out):
        self.name = name
        self.win = winner is not None and winner.name == name
        self.draw = winner is None
        self.loss = not (self.win or self.draw)
        self.tds = team.state.score
        self.cas = len(game.get_casualties(team))
        self.cas_inflicted = len(game.get_casualties(game.get_opp_team(team)))
        self.kills = len([player for player in game.get_casualties(team) if player.state.casualty_type == CasualtyType.DEAD])
        self.kills_inflicted = len([player for player in game.get_casualties(game.get_opp_team(team)) if player.state.casualty_type == CasualtyType.DEAD])
        # Count inflicted casualties and kills from reports
        self.timed_out_win = timed_out and self.win
        self.timed_out_loss = timed_out and not self.win and not self.draw
        self.crashed_win = crashed and self.win
        self.crashed_loss = crashed and not self.win and not self.draw

    def print(self):
        print("-- {}".format(self.name))
        print("Result: {}".format("Win" if self.win else ("Draw" if self.draw else "Loss")))
        if self.timed_out_win or self.timed_out_loss:
            print("Game timed out")
        if self.crashed_win or self.crashed_loss:
            print("Game crashed")
        print("TDs: {}".format(self.tds))
        print("Cas: {}".format(self.cas))
        print("Cas inflicted: {}".format(self.cas_inflicted))
        print("Kills: {}".format(self.kills_inflicted))
        print("Kills inflicted: {}".format(self.kills_inflicted))
        

class GameResult:

    def __init__(self, game, crashed=False):
        self.home_agent_name = game.home_agent.name
        self.away_agent_name = game.away_agent.name
        self.timed_out = game.is_timed_out()
        self.crashed = crashed
        self.winner = game.get_winner()
        self.disqualified_agent = game.disqualified_agent

        # If game crashed award the non-acting player
        if crashed:
            if game.actor is not None:
                self.winner = game.get_other_agent(game.actor)

        self.home_result = TeamResult(game, game.home_agent.name, game.state.home_team, self.winner, self.crashed, self.timed_out)
        self.away_result = TeamResult(game, game.away_agent.name, game.state.away_team, self.winner, self.crashed, self.timed_out)

        self.draw = self.winner is None
        self.tds = self.home_result.tds + self.away_result.tds
        self.cas_inflicted = self.home_result.cas_inflicted + self.away_result.cas_inflicted
        self.kills = self.home_result.kills_inflicted + self.away_result.kills_inflicted

    def print(self):
        print("############ GAME RESULTS ###########")
        print("Final score:")
        print("- {} {} - {} {}".format(self.away_agent_name, self.away_result.tds, self.home_result.tds, self.home_agent_name))
        print("Casualties inflicted:")
        print("- {} {} - {} {}".format(self.away_agent_name, self.away_result.cas_inflicted, self.home_result.cas_inflicted, self.home_agent_name))
        print("Kills inflicted:")
        print("- {} {} - {} {}".format(self.away_agent_name, self.away_result.kills_inflicted, self.home_result.kills_inflicted, self.home_agent_name))
        print("Result:")
        if self.winner is not None:
            if self.crashed and self.winner.name == self.home_agent_name.lower():
                print(f"- {self.away_agent_name} chrashed!")
            if self.crashed and self.winner.name == self.away_agent_name.lower():
                print(f"- {self.home_agent_name} chrashed!")
            if self.disqualified_agent is not None:
                print(f"- {self.disqualified_agent.name} was disqualified!")
            if self.timed_out and self.winner.name.lower() == self.home_agent_name.lower():
                print(f"- {self.away_agent_name} timed out!")
            if self.timed_out and self.winner.name.lower() == self.away_agent_name.lower():
                print(f"- {self.home_agent_name} timed out!")
            print(f"- Winner: {self.winner.name}")
        elif self.timed_out:
            print("- Timed out with no winner - no winner")
        elif self.crashed:
            print("- Game crashed - no winner")
        else:
            print("- Draw")
        print("#####################################")


class CompetitionResult:

    def __init__(self, competitor_a_name, competitor_b_name, game_results, disqualified=None):
        self.game_results = game_results
        self.disqualified = disqualified
        self.competitor_a_name = competitor_a_name
        self.competitor_b_name = competitor_b_name
        self.wins = {
            competitor_a_name: np.sum([1 if result.winner is not None and result.winner.name.lower() == competitor_a_name.lower() else 0 for result in game_results]),
            competitor_b_name: np.sum([1 if result.winner is not None and result.winner.name.lower() == competitor_b_name.lower() else 0 for result in game_results])
        }
        self.decided = self.wins[competitor_a_name] + self.wins[competitor_b_name]
        self.undecided = len(game_results) - self.decided
        self.crashes = len([result for result in game_results if result.crashed])
        self.a_crashes = len([result for result in game_results if result.crashed and result.winner is not None and result.winner.name.lower() != self.competitor_a_name.lower()])
        self.b_crashes = len([result for result in game_results if result.crashed and result.winner is not None and result.winner.name.lower() != self.competitor_b_name.lower()])
        self.a_disqualifications = len([result for result in game_results if result.disqualified_agent is not None and result.disqualified_agent.name.lower() == competitor_a_name.lower()])
        self.b_disqualifications = len([result for result in game_results if result.disqualified_agent is not None and  result.disqualified_agent.name.lower() == competitor_b_name.lower()])
        self.disquiaifications = self.a_disqualifications + self.b_disqualifications
        self.tds = {
            competitor_a_name: [result.home_result.tds if result.home_agent_name.lower() == competitor_a_name.lower() else result.away_result.tds for result in game_results],
            competitor_b_name: [result.home_result.tds if result.home_agent_name.lower() == competitor_b_name.lower() else result.away_result.tds for result in game_results]
        }
        self.cas_inflicted = {
            competitor_a_name: [result.home_result.cas_inflicted if result.home_agent_name.lower() == competitor_a_name.lower() else result.away_result.cas_inflicted for result in game_results],
            competitor_b_name: [result.home_result.cas_inflicted if result.home_agent_name.lower() == competitor_b_name.lower() else result.away_result.cas_inflicted for result in game_results]
        }
        self.kills_inflicted = {
            competitor_a_name: [result.home_result.kills_inflicted if result.home_agent_name.lower() == competitor_a_name.lower() else result.away_result.kills_inflicted for result in game_results],
            competitor_b_name: [result.home_result.kills_inflicted if result.home_agent_name.lower() == competitor_b_name.lower() else result.away_result.kills_inflicted for result in game_results]
        }

    def print(self):
        print("%%%%%%%%% COMPETITION RESULTS %%%%%%%%%")
        if self.disqualified is not None:
            print(f"{self.disqualified} WAS DISQULIFIED!")
        if len(self.game_results) == 0:
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            return
        print("Wins:")
        print("- {}: {}".format(self.competitor_a_name, self.wins[self.competitor_a_name]))
        print("- {}: {}".format(self.competitor_b_name, self.wins[self.competitor_b_name]))
        print(f"Draws: {self.undecided}")
        print(f"Crashes: {self.crashes}")
        if self.crashes > 0:
            print(f"- {self.competitor_a_name}: {self.a_crashes}")
            print(f"- {self.competitor_b_name}: {self.b_crashes}")
        print(f"Disqualifications:")
        print(f"- {self.competitor_a_name}: {self.a_disqualifications}")
        print(f"- {self.competitor_b_name}: {self.b_disqualifications}")
        print("TDs:")
        print("- {}: {} (avg. {})".format(self.competitor_a_name, np.sum(self.tds[self.competitor_a_name]), np.mean(self.tds[self.competitor_a_name])))
        print("- {}: {} (avg. {})".format(self.competitor_b_name, np.sum(self.tds[self.competitor_b_name]), np.mean(self.tds[self.competitor_b_name])))
        print("Casualties inflicted:")
        print("- {}: {} (avg. {})".format(self.competitor_a_name, np.sum(self.cas_inflicted[self.competitor_a_name]), np.mean(self.cas_inflicted[self.competitor_a_name])))
        print("- {}: {} (avg. {})".format(self.competitor_b_name, np.sum(self.cas_inflicted[self.competitor_b_name]), np.mean(self.cas_inflicted[self.competitor_b_name])))
        print("Kills inflicted:")
        print("- {}: {} (avg. {})".format(self.competitor_a_name, np.sum(self.kills_inflicted[self.competitor_a_name]), np.mean(self.kills_inflicted[self.competitor_a_name])))
        print("- {}: {} (avg. {})".format(self.competitor_b_name, np.sum(self.kills_inflicted[self.competitor_b_name]), np.mean(self.kills_inflicted[self.competitor_b_name])))
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

class TimeoutException(Exception): pass


class Competition:

    def __init__(self, name, competitor_a_team, competitor_b_team, competitor_a_name, competitor_b_name, config, record=False):
        assert competitor_a_name != competitor_b_name
        self.name = name
        self.competitor_a_name = competitor_a_name
        self.competitor_b_name = competitor_b_name
        self.config = config
        self.config.competition_mode = True
        self.ruleset = load_rule_set(self.config.ruleset)
        self.competitor_a_team = deepcopy(competitor_a_team)
        self.competitor_a_team.team_id = str(uuid.uuid1())
        self.competitor_b_team = deepcopy(competitor_b_team)
        self.competitor_b_team.team_id = str(uuid.uuid1())
        self.disqualified = None
        self.record = record

    def run(self, num_games):
        results = []
        for i in range(num_games):
            print(f"Setting up bots for game {i+1}")
            init_time = int(self.config.time_limits.init + self.config.time_limits.disqualification)
            competitor_a = self._get_competitor(self.competitor_a_name, init_time)
            if competitor_a is None:
                self.disqualified = self.competitor_a_name
                return CompetitionResult(self.competitor_a_name, self.competitor_b_name, results, disqualified=self.competitor_a_name)
            competitor_b = self._get_competitor(self.competitor_b_name, init_time)
            if competitor_b is None:
                return CompetitionResult(self.competitor_a_name, self.competitor_b_name, results, disqualified=self.competitor_b_name)
            print(f"Starting game {i+1}")
            match_id = self.name + "_" + str(i+1)
            if i%2==0:
                result = self._run_match(match_id, home_team=self.competitor_a_team, away_team=self.competitor_b_team, home_agent=competitor_a, away_agent=competitor_b)
            else:
                result = self._run_match(match_id, home_team=self.competitor_b_team, away_team=self.competitor_a_team, home_agent=competitor_b, away_agent=competitor_a)
            results.append(result)
            result.print()
        return CompetitionResult(self.competitor_a_name, self.competitor_b_name, results)

    def _get_competitor(self, name, init_time):
        bot = None
        try:
            with stopit.ThreadingTimeout(init_time) as to_ctx_mgr:
                assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
                bot = make_bot(name)
            if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
                return bot
            else:
                print(f"{name} timed out while initializing.")
        except Exception as e:
            print(f"{name} crashed while initializing:", e)
        return bot

    def _run_match(self, match_id, home_team, away_team, home_agent, away_agent):
        game = Game(match_id, home_team=deepcopy(home_team), away_team=deepcopy(away_team), home_agent=home_agent, away_agent=away_agent, config=self.config, record=self.record)
        game.config.fast_mode = True
        game.config.competition_mode = True
        print("Starting new match")
        try:
            with stopit.ThreadingTimeout(int(self.config.time_limits.game)) as to_ctx_mgr:
                game.init()
            if to_ctx_mgr.state != to_ctx_mgr.EXECUTED:
                print("Game timed out!")
                game.end_time = time.time()
                game.game_over = True
                game.disqualified_agent = game.actor
                return GameResult(game)
        except Exception as e:
            print(f"Game crashed by {game.actor.name if game.actor is not None else 'the framework'}: ", e)
            game.disqualified_agent = game.actor
            return GameResult(game, crashed=True)
        return GameResult(game)

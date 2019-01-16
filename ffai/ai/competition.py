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

    def __init__(self, game):
        self.home_agent_name = game.home_agent.name
        self.away_agent_name = game.away_agent.name
        self.home_result = TeamResult(game, game.home_agent.name, game.state.home_team)
        self.away_result = TeamResult(game, game.away_agent.name, game.state.away_team)
        self.draw = self.home_result.tds == self.away_result.tds
        self.winner = game.winner()
        self.tds = self.home_result.tds + self.away_result.tds
        self.cas_inflicted = self.home_result.cas_inflicted == self.away_result.cas_inflicted
        self.kills = self.home_result.kills_inflicted == self.away_result.kills_inflicted

    def print(self):
        print("############ GAME RESULTS ###########")
        print("Final score:")
        print("{} {} - {} {}".format(self.away_agent_name, self.away_result.tds, self.home_result.tds, self.home_agent_name))
        print("Casualties inflicted:")
        print("{} {} - {} {}".format(self.away_agent_name, self.away_result.cas_inflicted, self.home_result.cas_inflicted, self.home_agent_name))
        print("Kills inflicted:")
        print("{} {} - {} {}".format(self.away_agent_name, self.away_result.kills_inflicted, self.home_result.kills_inflicted, self.home_agent_name))
        print("#####################################")


class CompetitionResult:

    def __init__(self, competitor_a, competitor_b, game_results):
        self.game_results = game_results
        self.competitor_a_name = competitor_a.bot_id
        self.competitor_b_name = competitor_b.bot_id
        self.wins = {
            competitor_a.bot_id: np.sum([1 if result.winner is not None and result.winner.name == competitor_a.bot_id else 0 for result in game_results]),
            competitor_b.bot_id: np.sum([1 if result.winner is not None and result.winner.name == competitor_b.bot_id else 0 for result in game_results])
        }
        self.decided = self.wins[competitor_a.bot_id] + self.wins[competitor_b.bot_id]
        self.undecided = len(game_results) - self.decided
        self.tds = {
            competitor_a.bot_id: [result.home_result.tds if result.home_agent_name == competitor_a.bot_id else result.away_result.tds for result in game_results],
            competitor_b.bot_id: [result.home_result.tds if result.home_agent_name == competitor_b.bot_id else result.away_result.tds for result in game_results]
        }
        self.cas_inflicted = {
            competitor_a.bot_id: [result.home_result.cas_inflicted if result.home_agent_name == competitor_a.bot_id else result.away_result.cas_inflicted for result in game_results],
            competitor_b.bot_id: [result.home_result.cas_inflicted if result.home_agent_name == competitor_b.bot_id else result.away_result.cas_inflicted for result in game_results]
        }
        self.kills_inflicted = {
            competitor_a.bot_id: [result.home_result.kills_inflicted if result.home_agent_name == competitor_a.bot_id else result.away_result.kills_inflicted for result in game_results],
            competitor_b.bot_id: [result.home_result.kills_inflicted if result.home_agent_name == competitor_b.bot_id else result.away_result.kills_inflicted for result in game_results]
        }

    def print(self):
        print("%%%%%%%%% COMPETITION RESULTS %%%%%%%%%")
        print("Decided:", self.decided)
        print("Undecided:", self.undecided)
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


class Competitor:

    def __init__(self, bot_id, team):
        self.bot_id = bot_id
        self.team = team
        self.bot = make_bot(bot_id)


class Competition:

    def __init__(self, name, competitor_a, competitor_b, config):
        self.name = name
        self.competitor_a = competitor_a
        self.competitor_b = competitor_b
        self.config = config

    def run(self, num_games):
        results = []
        for i in range(num_games):
            print("Starting game", i+1)
            if i%2==0:
                result = self._compete(self.name + "_" + str(i+1), home=self.competitor_a, away=self.competitor_b)
            else:
                result = self._compete(self.name + "_" + str(i+1), home=self.competitor_b, away=self.competitor_a)
            results.append(result)
            result.print()
        return CompetitionResult(self.competitor_a, self.competitor_b, results)

    def _compete(self, match_id, home, away):
        game = Game(match_id, home_team=deepcopy(home.team), away_team=deepcopy(away.team), home_agent=home.bot, away_agent=away.bot, config=self.config)
        game.config.fast_mode = True
        game.init()
        game.step()
        assert game.state.game_over
        return GameResult(game)

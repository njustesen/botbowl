"""
==========================
Author: Niels Justesen
Year: 2019
==========================
This module contains a competition class to handle a competition between two bots.
"""
import numpy as np
from botbowl.core.table import CasualtyType
from botbowl.core import Game, InvalidActionError
from botbowl.core import load_arena, load_rule_set


class TeamResult:

    def __init__(self, game, name, team, winner, crashed):
        self.name = name
        self.win = winner is not None and winner.name == name
        self.draw = winner is None
        self.loss = not (self.win or self.draw)
        self.tds = team.state.score
        self.cas = len(game.get_casualties(team))
        self.cas_inflicted = len(game.get_casualties(game.get_opp_team(team)))
        self.killed = len([player for player in game.get_casualties(team) if CasualtyType.DEAD in player.state.injuries_gained])
        self.kills_inflicted = len([player for player in game.get_casualties(game.get_opp_team(team)) if CasualtyType.DEAD in player.state.injuries_gained])
        # Count inflicted casualties and kills from reports
        self.crashed_win = crashed and self.win
        self.crashed_loss = crashed and not self.win and not self.draw

    def print(self):
        print("-- {}".format(self.name))
        print("Result: {}".format("Win" if self.win else ("Draw" if self.draw else "Loss")))
        if self.crashed_win or self.crashed_loss:
            print("Game crashed")
        print("TDs: {}".format(self.tds))
        print("Cas: {}".format(self.cas))
        print("Cas inflicted: {}".format(self.cas_inflicted))
        print("Killed: {}".format(self.killed))
        print("Kills: {}".format(self.kills_inflicted))
        

class GameResult:

    def __init__(self, game, crashed=False):
        self.home_agent_name = game.home_agent.name
        self.away_agent_name = game.away_agent.name
        self.crashed = crashed
        if crashed:
            self.winner = None
        else:
            self.winner = game.get_winner()

        self.home_result = TeamResult(game, game.home_agent.name, game.state.home_team, self.winner, self.crashed)
        self.away_result = TeamResult(game, game.away_agent.name, game.state.away_team, self.winner, self.crashed)

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
            print(f"- Winner: {self.winner.name}")
        elif self.crashed:
            print("- Game crashed - no winner")
        else:
            print("- Draw")
        print("#####################################")


class CompetitionResults:

    def __init__(self, competitor_a_name, competitor_b_name, game_results):
        self.game_results = game_results
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

    def __init__(self, agent_a, agent_b, team_a, team_b, config, ruleset, arena, n=2, record=False):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.team_a = team_a
        self.team_b = team_b
        self.config = config
        self.config.competition_mode = True
        self.arena = load_arena(config.arena)
        self.ruleset = load_rule_set(config.ruleset)
        self.n = n
        self.results = None
        self.ruleset = ruleset
        self.arena = arena
        self.record = record

    def run(self):
        results = []
        for i in range(self.n):
            crashed = False
            home_agent = self.agent_a if i % 2 == 0 else self.agent_b
            away_agent = self.agent_b if i % 2 == 0 else self.agent_a
            home_team = self.team_a if i % 2 == 0 else self.team_b
            away_team = self.team_b if i % 2 == 0 else self.team_a
            game = Game(i, home_team, away_team, home_agent, away_agent, self.config, arena=self.arena, ruleset=self.ruleset, record=self.record)
            self._run_game(game)

            print(f"{home_agent.name} {game.state.home_team.state.score} - {game.state.away_team.state.score} {away_agent.name}")

            result = GameResult(game, crashed=crashed)
            results.append(result)
        self.results = CompetitionResults(self.agent_a.name, self.agent_b.name, results)

    def _run_game(self, game):
        while not game.state.game_over:
            try:
                if not game.is_started():
                    game.init()
                elif game.get_seconds_left() > 0:
                    action = game.actor.act(game)  # Allow actor to try again
                    game.step(action)
                else:
                    game.step(game._forced_action())
            except InvalidActionError as e:
                print(e)

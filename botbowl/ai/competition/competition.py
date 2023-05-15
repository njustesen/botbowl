"""
==========================
Author: Niels Justesen
Year: 2019
==========================
This module contains a competition class to handle a competition between two bots.
"""
import dataclasses
import tabulate
from itertools import combinations
from typing import Callable, Optional, Union
import numpy as np
from botbowl.core.model import Team
from botbowl.core.table import CasualtyType
from botbowl.core import (
    Game,
    InvalidActionError,
    Agent,
    Configuration,
    RuleSet,
    TwoPlayerArena,
)
from botbowl.core import load_arena, load_rule_set

from typing import Optional, Any, List


class TeamResult:
    name: str
    win: bool
    draw: bool
    loss: bool
    tds: int
    cas: int
    cas_inflicted: int
    killed: int
    kills_inflicted: int
    crashed_win: bool
    crashed_loss: bool

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
        print(f"-- {self.name}")
        print(f"Result: {'Win' if self.win else 'Draw' if self.draw else 'Loss'}")
        if self.crashed_win or self.crashed_loss:
            print("Game crashed")
        print(f"TDs: {self.tds}")
        print(f"Cas: {self.cas}")
        print(f"Cas inflicted: {self.cas_inflicted}")
        print(f"Killed: {self.killed}")
        print(f"Kills: {self.kills_inflicted}")


class GameResult:
    home_agent_name: str
    away_agent_name: str
    winner: Optional[Agent]
    home_result: TeamResult
    away_result: TeamResult
    draw: bool
    crashed: bool
    tds: int
    cas_inflicted: int
    kills: int

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
        away_agent_name = self.away_agent_name
        home_agent_name = self.home_agent_name
        away_result = self.away_result
        home_result = self.home_result

        print(f"Final score:\n- {away_agent_name} {away_result.tds} - {home_result.tds} {home_agent_name}")
        print(f"Casualties inflicted:\n- {away_agent_name} {away_result.cas_inflicted} - {home_result.cas_inflicted} {home_agent_name}")
        print(f"Kills inflicted:\n- {away_agent_name} {away_result.kills_inflicted} - {home_result.kills_inflicted} {home_agent_name}")
        print("Result:")
        if self.winner is not None:
            print(f"- Winner: {self.winner.name}")
        elif self.crashed:
            print("- Game crashed - no winner")
        else:
            print("- Draw")
        print("#####################################")

    def __str__(self):
        away_agent_name = self.away_agent_name
        home_agent_name = self.home_agent_name
        away_result = self.away_result
        home_result = self.home_result
        # if home_result.win:
        #     home_agent_name = f"**{home_agent_name}**"
        # elif away_result.win:
        #     away_agent_name = f"**{away_agent_name}**"

        return f"{home_agent_name} vs. {away_agent_name}, {home_result.tds} - {away_result.tds}"+\
                ", cas_inflicted: {home_result.cas_inflicted} - {away_result.cas_inflicted}"


class CompetitionResults:
    game_results: list[GameResult]
    competitor_a_name: str
    competitor_b_name: str
    wins: dict[str, int]
    decided: int
    undecided: int
    crashes: int
    a_crashes: int
    b_crashes: int
    tds: dict[str, list[int]]
    cas_inflicted: dict[str, list[int]]
    kills: dict

    def __init__(self, competitor_a_name: str, competitor_b_name: str, game_results: list[GameResult]):
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


@dataclasses.dataclass
class AgentSummaryResult:
    name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    tds_scored: int = 0
    tds_conceded: int = 0
    final_score: float = 0.0

    def add_comp_result(self, result: CompetitionResults):
        assert self.name in {result.competitor_a_name, result.competitor_b_name}
        assert result.crashes == 0
        self.wins += result.wins[self.name]
        self.losses += sum(result.wins.values()) - result.wins[self.name]
        self.draws += result.undecided # ugh.. is this correct?

        tds_scored = sum(result.tds[self.name])
        tds_total = sum(result.tds[result.competitor_a_name]) + sum(result.tds[result.competitor_b_name])
        tds_conceded = tds_total - tds_scored
        
        self.tds_scored += tds_scored  
        self.tds_conceded += tds_conceded

    @staticmethod
    def get_titles() -> list[str]:
        return ["Name", "Final Score", "Wins", "Losses", "Draws", "TDs Scored", "TDs Conceded"]

    def get_values(self) -> list[Union[float, str]]:
        return [self.name, self.final_score, self.wins, self.losses, self.draws, self.tds_scored, self.tds_conceded]

    @staticmethod
    def csv_header() -> str:
        return ",".join(AgentSummaryResult.get_titles())

    def csv_row(self) -> str:
        return ",".join([str(value) for value in self.get_values()])

class TimeoutException(Exception):
    pass


class Competition:
    agent_a: Agent
    agent_b: Agent
    team_a: Team
    team_b: Team
    config: Configuration
    arena: TwoPlayerArena
    ruleset: Any
    n: int
    results: Optional[CompetitionResults]
    record: bool

    def __init__(
        self,
        agent_a: Agent,
        agent_b: Agent,
        team_a: Team,
        team_b: Team,
        config: Configuration,
        ruleset,
        arena: Optional[TwoPlayerArena],
        n: int = 2,
        record=False,
    ):
        assert n % 2 == 0, "Number of games must be even"
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.team_a = team_a
        self.team_b = team_b
        self.config = config
        self.config.competition_mode = True
        self.arena = load_arena(config.arena) if arena is None else arena
        self.ruleset = load_rule_set(config.ruleset)
        self.n = n
        self.results = None
        self.ruleset = ruleset
        self.record = record

    def run(self):
        results: list[GameResult] = []

        for i in range(self.n):
            crashed = False
            home_agent = self.agent_a if i % 2 == 0 else self.agent_b
            away_agent = self.agent_b if i % 2 == 0 else self.agent_a
            home_team = self.team_a if i % 2 == 0 else self.team_b
            away_team = self.team_b if i % 2 == 0 else self.team_a
            game = Game(
                str(i),
                home_team,
                away_team,
                home_agent,
                away_agent,
                self.config,
                arena=self.arena,
                ruleset=self.ruleset,
                record=self.record,
            )
            self._run_game(game)

            print(
                f"{home_agent.name} {game.state.home_team.state.score} - {game.state.away_team.state.score} {away_agent.name}"
            )

            result = GameResult(game, crashed=crashed)
            results.append(result)
        self.results = CompetitionResults(self.agent_a.name, self.agent_b.name, results)
        return self.results

    def _run_game(self, game: Game):
        assert not game.is_started()
        try: 
            # game will finish or throw exception 
            game.init()
        except InvalidActionError as e:
            print(e)

        while not game.state.game_over:
            time_left = game.get_seconds_left()
            if time_left is None or time_left > 0:
                try: 
                    assert game.actor is not None
                    action = game.actor.act(game)  # Allow actor to try again
                    game.step(action)
                except InvalidActionError as e:
                    print(e)
            else:
                print("Using forced action")
                game.step(game._forced_action())


AgentCreator = Callable[[], Agent]

default_score_calculator = lambda x: 3*x.wins + x.draws

class MultiAgentCompetition:
    agents: list[AgentCreator]
    results: list[CompetitionResults]
    home_team: Team
    away_team: Team
    config: Configuration
    ruleset: Optional[RuleSet]
    arena: Optional[TwoPlayerArena]
    record: bool
    number_of_games: int
    score_calculator: Callable[[AgentSummaryResult], float]

    def __init__(
        self,
        agents: list[AgentCreator],
        home_team: Team,
        away_team: Team,
        config: Configuration,
        ruleset: Optional[RuleSet] = None,
        arena: Optional[TwoPlayerArena] = None,
        record: bool = False,
        number_of_games: int = 2,
        score_calc_func: Optional[Callable[[AgentSummaryResult], float]] = None,
    ):
        self.home_team = home_team
        self.away_team = away_team
        self.config = config
        self.agents = agents
        self.ruleset = ruleset
        self.arena = arena
        self.record = record
        self.number_of_games = number_of_games

        self.results = []
        self.matchups = list(combinations(self.agents, 2))
        self.score_calculator = score_calc_func if score_calc_func is not None else default_score_calculator

        names = set()
        for create_agent in self.agents:
            agent = create_agent()
            if agent.name in names:
                raise ValueError(
                    f"Agent names must be unique, '{agent.name}' is used more than once"
                )
            names.add(agent.name)

    def run(self):
        for create_agent_a, create_agent_b in self.matchups:
            agent_a = create_agent_a()
            agent_b = create_agent_b()

            print(f"Running {agent_a.name} vs {agent_b.name}")
            competition = Competition(
                agent_a,
                agent_b,
                self.home_team,
                self.away_team,
                self.config,
                self.ruleset,
                self.arena,
                self.number_of_games,
                self.record,
            )
            result = competition.run()
            self.results.append(result)

    def get_game_results(self) -> list[GameResult]:
        results = [] 
        for comp_result in self.results:
            results.extend(sorted(comp_result.game_results, key=str))
        return results

    def result_summarized_csv(self) -> str:
        agent_summaries: dict[str, AgentSummaryResult] = {}
        for result in self.results:
            for agent_name in [result.competitor_a_name, result.competitor_b_name]:
                agent_summary = agent_summaries.get(agent_name, AgentSummaryResult(agent_name))
                agent_summary.add_comp_result(result)
                agent_summaries[agent_name] = agent_summary

        for agent_summary in agent_summaries.values():
            agent_summary.final_score = self.score_calculator(agent_summary)

        csv_str = AgentSummaryResult.csv_header() + "\n"

        ordered_summaries = sorted(agent_summaries.values(), key=lambda x: x.final_score, reverse=True)
        csv_str += "\n".join(summary.csv_row() for summary in ordered_summaries)

        return csv_str

    def print_summarized_result(self):
        table = [line.split(',') for line in self.result_summarized_csv().split('\n')]
        print(tabulate.tabulate(table, headers="firstrow", tablefmt="github"))
        

    def result_versus_csv(self) -> str:
        num_agents = len(self.agents)
        matchup_data: List[List[str]] = [["" for _ in range(num_agents)] for _ in range(num_agents)]
        agent_names = []
        for result in self.results:
            name_a = result.competitor_a_name
            name_b = result.competitor_b_name
            assert name_a != name_b  
            if name_a not in agent_names:
                agent_names.append(name_a)
            if name_b not in agent_names:
                agent_names.append(name_b)
            idx_a = agent_names.index(name_a)
            idx_b = agent_names.index(name_b)

            result_list = [result.wins[name_a], result.undecided, result.wins[name_b]]
            result_str = "/".join(str(x) for x in result_list)
            result_str_reversed = "/".join(str(x) for x in result_list[::-1])
            matchup_data[idx_a][idx_b] = result_str
            matchup_data[idx_b][idx_a] = result_str_reversed

        table = [["Bot Names"]+agent_names[:]]
        table += [[agent_name] + data_row for agent_name, data_row in zip(agent_names, matchup_data)]

        return "\n".join(",".join(row) for row in table)

    def print_versus_result(self):
        table = [line.split(',') for line in self.result_versus_csv().split('\n')]
        print(tabulate.tabulate(table, tablefmt="grid"))


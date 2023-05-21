"""
==========================
Author: Niels Justesen
Year: 2019
==========================
This module contains a competition class to handle a competition between two bots.
"""
import tabulate
from itertools import combinations
from typing import Callable, Optional, Any, List
from botbowl.ai.competition.result_structures import (
    CompetitionResults,
    GameResult,
    AgentSummaryResult,
)
from botbowl.core.model import Team
from botbowl.core import (
    Game,
    InvalidActionError,
    Agent,
    Configuration,
    RuleSet,
    TwoPlayerArena,
    load_arena,
    load_rule_set,
)


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
        results: List[GameResult] = []

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


def default_score_calculator(agent_summary: AgentSummaryResult):
    return 3 * agent_summary.wins + agent_summary.draws


class MultiAgentCompetition:
    agents: List[AgentCreator]
    results: List[CompetitionResults]
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
        agents: List[AgentCreator],
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
        self.score_calculator = (
            score_calc_func if score_calc_func is not None else default_score_calculator
        )

        names = set()
        for create_agent in self.agents:
            agent = create_agent()
            if "," in agent.name:
                raise ValueError(
                    f"Agent names must not contain commas, '{agent.name}' contains a comma"
                )
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

    def get_game_results(self) -> List[GameResult]:
        results = []
        for comp_result in self.results:
            results.extend(sorted(comp_result.game_results, key=str))
        return results

    def result_summarized_csv(self) -> str:
        agent_summaries: dict[str, AgentSummaryResult] = {}
        for result in self.results:
            for agent_name in [result.competitor_a_name, result.competitor_b_name]:
                agent_summary = agent_summaries.get(
                    agent_name, AgentSummaryResult(agent_name)
                )
                agent_summary.add_comp_result(result)
                agent_summaries[agent_name] = agent_summary

        for agent_summary in agent_summaries.values():
            agent_summary.final_score = self.score_calculator(agent_summary)

        ordered_summaries = sorted(
            agent_summaries.values(), key=lambda x: x.final_score, reverse=True
        )
        csv_str = ordered_summaries[0].csv_header() + "\n"
        csv_str += "\n".join(summary.csv_row() for summary in ordered_summaries)

        return csv_str

    def print_summarized_result(self):
        table = [line.split(",") for line in self.result_summarized_csv().split("\n")]
        print(tabulate.tabulate(table, headers="firstrow", tablefmt="github"))

    def result_versus_csv(self) -> str:
        num_agents = len(self.agents)
        matchup_data: List[List[str]] = [
            ["" for _ in range(num_agents)] for _ in range(num_agents)
        ]
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

        table = [["Bot Names"] + agent_names[:]]
        table += [
            [agent_name] + data_row
            for agent_name, data_row in zip(agent_names, matchup_data)
        ]

        return "\n".join(",".join(row) for row in table)

    def print_versus_result(self):
        table = [line.split(",") for line in self.result_versus_csv().split("\n")]
        print(tabulate.tabulate(table, tablefmt="grid"))

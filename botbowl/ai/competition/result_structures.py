import dataclasses

from botbowl.core import Agent
from botbowl.core.table import CasualtyType
import numpy as np
from typing import Optional, Union, List, Dict, Tuple


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
        self.killed = len(
            [
                player
                for player in game.get_casualties(team)
                if CasualtyType.DEAD in player.state.injuries_gained
            ]
        )
        self.kills_inflicted = len(
            [
                player
                for player in game.get_casualties(game.get_opp_team(team))
                if CasualtyType.DEAD in player.state.injuries_gained
            ]
        )
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

        self.home_result = TeamResult(
            game, game.home_agent.name, game.state.home_team, self.winner, self.crashed
        )
        self.away_result = TeamResult(
            game, game.away_agent.name, game.state.away_team, self.winner, self.crashed
        )

        self.draw = self.winner is None
        self.tds = self.home_result.tds + self.away_result.tds
        self.cas_inflicted = (
            self.home_result.cas_inflicted + self.away_result.cas_inflicted
        )
        self.kills = self.home_result.kills_inflicted + self.away_result.kills_inflicted

    def print(self):
        print("############ GAME RESULTS ###########")
        away_agent_name = self.away_agent_name
        home_agent_name = self.home_agent_name
        away_result = self.away_result
        home_result = self.home_result

        print(
            f"Final score:\n- {away_agent_name} {away_result.tds} - {home_result.tds} {home_agent_name}"
        )
        print(
            f"Casualties inflicted:\n- {away_agent_name} {away_result.cas_inflicted} - {home_result.cas_inflicted} {home_agent_name}"
        )
        print(
            f"Kills inflicted:\n- {away_agent_name} {away_result.kills_inflicted} - {home_result.kills_inflicted} {home_agent_name}"
        )
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

        return (
            f"{home_agent_name} vs. {away_agent_name}, {home_result.tds} - {away_result.tds}"
            + f", cas_inflicted: {home_result.cas_inflicted} - {away_result.cas_inflicted}"
        )


class CompetitionResults:
    game_results: List[GameResult]
    competitor_a_name: str
    competitor_b_name: str
    wins: Dict[str, int]
    decided: int
    undecided: int
    crashes: int
    a_crashes: int
    b_crashes: int
    tds: Dict[str, List[int]]
    cas_inflicted: Dict[str, List[int]]
    kills: Dict

    def __init__(
        self,
        competitor_a_name: str,
        competitor_b_name: str,
        game_results: List[GameResult],
    ):
        self.game_results = game_results
        self.competitor_a_name = competitor_a_name
        self.competitor_b_name = competitor_b_name
        self.wins = {
            competitor_a_name: sum(
                result.winner is not None
                and result.winner.name.lower() == competitor_a_name.lower()
                for result in game_results
            ),
            competitor_b_name: sum(
                result.winner is not None
                and result.winner.name.lower() == competitor_b_name.lower()
                for result in game_results
            ),
        }
        self.decided = self.wins[competitor_a_name] + self.wins[competitor_b_name]
        self.undecided = len(game_results) - self.decided
        self.crashes = len([result for result in game_results if result.crashed])
        self.a_crashes = len(
            [
                result
                for result in game_results
                if result.crashed
                and result.winner is not None
                and result.winner.name.lower() != self.competitor_a_name.lower()
            ]
        )
        self.b_crashes = len(
            [
                result
                for result in game_results
                if result.crashed
                and result.winner is not None
                and result.winner.name.lower() != self.competitor_b_name.lower()
            ]
        )
        self.tds = {
            competitor_a_name: [
                result.home_result.tds
                if result.home_agent_name.lower() == competitor_a_name.lower()
                else result.away_result.tds
                for result in game_results
            ],
            competitor_b_name: [
                result.home_result.tds
                if result.home_agent_name.lower() == competitor_b_name.lower()
                else result.away_result.tds
                for result in game_results
            ],
        }
        self.cas_inflicted = {
            competitor_a_name: [
                result.home_result.cas_inflicted
                if result.home_agent_name.lower() == competitor_a_name.lower()
                else result.away_result.cas_inflicted
                for result in game_results
            ],
            competitor_b_name: [
                result.home_result.cas_inflicted
                if result.home_agent_name.lower() == competitor_b_name.lower()
                else result.away_result.cas_inflicted
                for result in game_results
            ],
        }
        self.kills_inflicted = {
            competitor_a_name: [
                result.home_result.kills_inflicted
                if result.home_agent_name.lower() == competitor_a_name.lower()
                else result.away_result.kills_inflicted
                for result in game_results
            ],
            competitor_b_name: [
                result.home_result.kills_inflicted
                if result.home_agent_name.lower() == competitor_b_name.lower()
                else result.away_result.kills_inflicted
                for result in game_results
            ],
        }

    def print(self):
        print("%%%%%%%%% COMPETITION RESULTS %%%%%%%%%")
        if len(self.game_results) == 0:
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            return
        print("Wins:")
        print(
            "- {}: {}".format(self.competitor_a_name, self.wins[self.competitor_a_name])
        )
        print(
            "- {}: {}".format(self.competitor_b_name, self.wins[self.competitor_b_name])
        )
        print(f"Draws: {self.undecided}")
        print(f"Crashes: {self.crashes}")
        if self.crashes > 0:
            print(f"- {self.competitor_a_name}: {self.a_crashes}")
            print(f"- {self.competitor_b_name}: {self.b_crashes}")
        print("TDs:")
        print(
            "- {}: {} (avg. {})".format(
                self.competitor_a_name,
                np.sum(self.tds[self.competitor_a_name]),
                np.mean(self.tds[self.competitor_a_name]),
            )
        )
        print(
            "- {}: {} (avg. {})".format(
                self.competitor_b_name,
                np.sum(self.tds[self.competitor_b_name]),
                np.mean(self.tds[self.competitor_b_name]),
            )
        )
        print("Casualties inflicted:")
        print(
            "- {}: {} (avg. {})".format(
                self.competitor_a_name,
                np.sum(self.cas_inflicted[self.competitor_a_name]),
                np.mean(self.cas_inflicted[self.competitor_a_name]),
            )
        )
        print(
            "- {}: {} (avg. {})".format(
                self.competitor_b_name,
                np.sum(self.cas_inflicted[self.competitor_b_name]),
                np.mean(self.cas_inflicted[self.competitor_b_name]),
            )
        )
        print("Kills inflicted:")
        print(
            "- {}: {} (avg. {})".format(
                self.competitor_a_name,
                np.sum(self.kills_inflicted[self.competitor_a_name]),
                np.mean(self.kills_inflicted[self.competitor_a_name]),
            )
        )
        print(
            "- {}: {} (avg. {})".format(
                self.competitor_b_name,
                np.sum(self.kills_inflicted[self.competitor_b_name]),
                np.mean(self.kills_inflicted[self.competitor_b_name]),
            )
        )
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
        self.draws += result.undecided

        tds_scored = sum(result.tds[self.name])
        tds_total = sum(sum(tds) for tds in result.tds.values())
        tds_conceded = tds_total - tds_scored

        self.tds_scored += tds_scored
        self.tds_conceded += tds_conceded

    def _csv_header_and_row(self) -> Tuple[List[str], List[str]]:
        csv_header_and_value = [
            ("Name", self.name),
            ("Final Score", self.final_score),
            ("Wins", self.wins),
            ("Losses", self.losses),
            ("Draws", self.draws),
            ("TDs Scored", self.tds_scored),
            ("TDs Conceded", self.tds_conceded),
        ]
        return tuple(zip(*csv_header_and_value))  # transpose

    def get_titles(self) -> List[str]:
        return self._csv_header_and_row()[0]

    def get_values(self) -> List[str]:
        return self._csv_header_and_row()[1]

    def csv_header(self) -> str:
        return ",".join(self.get_titles())

    def csv_row(self) -> str:
        return ",".join([str(value) for value in self.get_values()])

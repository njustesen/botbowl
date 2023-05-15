#!/usr/bin/env python3
import botbowl
import os 
import pickle
from botbowl.ai.competition import DockerAgent, MultiAgentCompetition
import botbowl
from botbowl.ai.registry import make_bot
from examples import scripted_bot_example
from typing import Optional


class AgentCreator:
    def __init__(self, name: Optional[str], make_bot_arg: str):
        self.name = name
        self.make_bot_arg = make_bot_arg

    def __call__(self):
        bot = make_bot(self.make_bot_arg)
        if self.name is not None:
            bot.name = self.name
        return bot


class DockerAgentCreator:
    def __init__(
        self,
        name: str,
        image: str = "botbowl:latest",
        command: str = "python botbowl/examples/containerized_bot.py",
    ):
        self.name = name
        self.image = image
        self.command = command

    def __call__(self):
        bot = DockerAgent(self.name, image=self.image, command=self.command)
        return bot


if __name__ == "__main__":
    # Load configurations, rules, arena and teams
    config = botbowl.load_config("bot-bowl")
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    team_a = botbowl.load_team_by_filename("human", ruleset)
    team_b = botbowl.load_team_by_filename("human", ruleset)

    agent_creators = [
        AgentCreator("Random One", "random"),
        AgentCreator("Random Two", "random"),
        AgentCreator("Scripted One", "scripted"),
        AgentCreator("Scripted Two", "scripted"),
    ]

    competition = MultiAgentCompetition(
        agent_creators,
        team_a,
        team_b,
        config=config,
        ruleset=ruleset,
        arena=arena,
        number_of_games=2,
        record=False,
    )
    competition.run()
    print("-- Summarized result --")
    competition.print_summarized_result()
    print("\n-- Verses table --")
    competition.print_versus_result()
    print("\n-- Game details --")
    game_results = map(str, competition.get_game_results())
    for game_result in game_results:
        print(game_result)

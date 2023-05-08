#!/usr/bin/env python3

import botbowl
from botbowl.ai.competition import DockerAgent
import botbowl

if __name__ == "__main__":
    agent_a = DockerAgent("botbowl:latest", "python botbowl/examples/containerized_bot.py")
    agent_b = DockerAgent("botbowl:latest", "python botbowl/examples/containerized_bot.py")

    # Load configurations, rules, arena and teams
    config = botbowl.load_config("bot-bowl")

    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    team_a = botbowl.load_team_by_filename("human", ruleset)
    team_b = botbowl.load_team_by_filename("human", ruleset)

    # Run competition
    competition = botbowl.Competition(
        agent_a,
        agent_b,
        team_a,
        team_b,
        config=config,
        ruleset=ruleset,
        arena=arena,
        n=2,
        record=True,
    )
    competition.run()
    competition.results.print()

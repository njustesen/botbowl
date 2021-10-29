#!/usr/bin/env python3
from scripted_bot_example import *

import botbowl as botbowl
import time as time

config = botbowl.load_config("bot-bowl-iii")
config.competition_mode = False
config.pathfinding_enabled = True
config.debug_mode = False
ruleset = botbowl.load_rule_set(config.ruleset, all_rules=False)  # We don't need all the rules
arena = botbowl.load_arena(config.arena)
home = botbowl.load_team_by_filename("human", ruleset)
away = botbowl.load_team_by_filename("human", ruleset)

# Play 10 games
for i in range(10):
    home_agent = botbowl.make_bot('scripted')
    home_agent.name = "Bot 1"
    away_agent = botbowl.make_bot('scripted')
    away_agent.name = "Bot 2"
    game = botbowl.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
    game.config.fast_mode = True

    print("Starting game", (i+1))
    start = time.time()
    game.init()
    end = time.time()
    print(end - start)
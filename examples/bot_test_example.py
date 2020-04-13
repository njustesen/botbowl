#!/usr/bin/env python3
from grodbot import *
from scripted_bot_example import *

import ffai as ffai
import time as time

config = ffai.load_config("bot-bowl-ii")
config.competition_mode = False
ruleset = ffai.load_rule_set(config.ruleset, all_rules=False)  # We don't need all the rules
arena = ffai.load_arena(config.arena)
home = ffai.load_team_by_filename("human", ruleset)
away = ffai.load_team_by_filename("human", ruleset)

# Play 10 games
for i in range(10):
    home_agent = ffai.make_bot('grodbot')
    home_agent.set_verbose(True)
    home_agent.set_debug(False)
    home_agent.name = "Grod"
    away_agent = ffai.make_bot('scripted')
    away_agent.name = "RandomBot"
    config.debug_mode = False
    game = ffai.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
    game.config.fast_mode = True

    print("Starting game", (i+1))
    start = time.time()
    game.init()
    end = time.time()
    print(end - start)
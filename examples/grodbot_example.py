#!/usr/bin/env python3

import ffai
import time
import examples.scripted_bot_example

# Load configurations, rules, arena and teams
config = ffai.load_config("bot-bowl-i.json")
config.competition_mode = False
ruleset = ffai.load_rule_set(config.ruleset, all_rules=False)  # We don't need all the rules
arena = ffai.load_arena(config.arena)
home = ffai.load_team_by_filename("human", ruleset)
away = ffai.load_team_by_filename("human", ruleset)
config.competition_mode = False

# Play 5 games as away
for i in range(5):
    away_agent = ffai.make_bot('grodbot')
    away_agent.name = 'grodbot'
    home_agent = ffai.make_bot('scripted')
    home_agent.name = 'scripted'
    config.debug_mode = False
    game = ffai.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
    game.config.fast_mode = True

    print("Starting game", (i + 1))
    start = time.time()
    game.init()
    end = time.time()
    print(end - start)

# Play 5 games as home
for i in range(5):
    away_agent = ffai.make_bot('scripted')
    away_agent.name = 'scripted'
    home_agent = ffai.make_bot('grodbot')
    home_agent.name = 'grodbot'
    config.debug_mode = False
    game = ffai.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
    game.config.fast_mode = True

    print("Starting game", (i + 1))
    start = time.time()
    game.init()
    end = time.time()
    print(end - start)


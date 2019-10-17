#!/usr/bin/env python3

from ffai.ai.competition import Competition
from ffai.core.load import get_config

from ffai.ai.bots import crash_bot
from ffai.ai.bots import grodbot
from ffai.ai.bots import idle_bot
from ffai.ai.bots import init_crash_bot
from ffai.ai.bots import just_in_time_bot
from ffai.ai.bots import manipulator_bot
from ffai.ai.bots import random_bot
from ffai.ai.bots import violator_bot


# Load competition configuration for the bot bowl
config = get_config('ff-11-bot-bowl-i.json')


# Random vs. Random
competition = Competition('MyCompetition', competitor_a_team_id='human-1', competitor_b_team_id='human-2', competitor_a_name='random', competitor_b_name='grodbot', config=config)
results = competition.run(num_games=2)
results.print()

# Random vs. idle
config.time_limits.game = 10  # 10 second time limit per game
config.time_limits.turn = 1  # 1 second time limit per turn
competition = Competition('MyCompetition', competitor_a_team_id='human-1', competitor_b_team_id='human-2', competitor_a_name='random', competitor_b_name='idle', config=config)
results = competition.run(num_games=2)
results.print()

# Random vs. violator
config.time_limits.game = 60  # 60 second time limit per game
config.time_limits.turn_ = 1  # 1 second time limit per turn
config.time_limits.secondary = 1  # 1 second time limit for secondary choices
config.time_limits.disqualification = 1  # 1 second disqualification limit 
competition = Competition('MyCompetition', competitor_a_team_id='human-1', competitor_b_team_id='human-2', competitor_a_name='random', competitor_b_name='violator', config=config)
results = competition.run(num_games=2)
results.print()

# Random vs. just-in-time
config.time_limits.game = 600  # 60 second time limit per game
config.time_limits.turn = 1  # 1 second time limit per turn
config.time_limits.secondary = 1  # 1 second time limit for secondary choices
config.time_limits.disqualification = 1  # 1 second disqualification limit 
#config.debug_mode = True
competition = Competition('MyCompetition', competitor_a_team_id='human-1', competitor_b_team_id='human-2', competitor_a_name='random', competitor_b_name='just-in-time', config=config)
results = competition.run(num_games=2)
results.print()

# Random vs. init crash
config.time_limits.game = 60  # 60 second time limit per game
config.time_limits.turn = 1  # 1 second time limit per turn
config.time_limits.secondary = 1  # 1 second time limit for secondary choices
config.time_limits.disqualification = 1  # 1 second disqualification threshold 
config.time_limits.init = 20  # 3 init limit
competition = Competition('MyCompetition', competitor_a_team_id='human-1', competitor_b_team_id='human-2', competitor_a_name='random', competitor_b_name='init-crash', config=config)
results = competition.run(num_games=2)
results.print()

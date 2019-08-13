#!/usr/bin/env python3

from ffai.ai.competition import Competition
import examples.scripted_bot_example
import examples.grodbot
from copy import deepcopy
from ffai.core.load import get_team, get_rule_set, get_config
import noone.a2c_agent
import kim.grodbot_gen613

# Load competition configuration for the bot bowl
config = get_config('ff-11-bot-bowl-i.json')

agent = 'grodbot_gen613'
# agent = 'NO_ONE'
competition = Competition('MyCompetition', competitor_a_team_id='human-1', competitor_b_team_id='human-2', competitor_a_name='grodbot_gen613', competitor_b_name='random', config=config)
results = competition.run(num_games=2)
results.print()

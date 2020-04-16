#!/usr/bin/env python3

import ffai
import ffai.ai.bots.testbots


# Load competition configuration for the bot bowl
config = ffai.load_config('bot-bowl-ii')

# Get ruleset
ruleset = ffai.load_rule_set(config.ruleset, all_rules=False)

# Load team to be used
human_team_a = ffai.load_team_by_filename('human', ruleset)
human_team_b = ffai.load_team_by_filename('human', ruleset)

# Random vs. Random
competition = ffai.Competition('MyCompetition', competitor_a_team=human_team_a, competitor_b_team=human_team_b, competitor_a_name='random', competitor_b_name='grodbot', config=config)
results = competition.run(num_games=2)
results.print()

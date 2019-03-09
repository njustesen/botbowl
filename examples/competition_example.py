'''
Still WiP. Do not expect it work.
'''

from ffai.ai.competition import Competition, Competitor
import examples.scripted_bot_example
from copy import deepcopy
from ffai.core.load import get_team, get_rule_set, get_config

config = get_config('ff-11.json')
ruleset = get_rule_set(config.ruleset)
team1 = get_team('human-1', ruleset)
team2 = get_team('human-2', ruleset)
competitor_random = Competitor('random', deepcopy(team1))
competitor_scripted = Competitor('scripted', deepcopy(team2))
competition = Competition('MyCompetition', competitor_random, competitor_scripted, config=config)
results = competition.run(num_games=20)
results.print()

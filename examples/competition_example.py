from ffai.ai.competition import Competition
import examples.scripted_bot_example
import examples.grodbot
from copy import deepcopy
from ffai.core.load import get_team, get_rule_set, get_config

config = get_config('ff-11.json')
competition = Competition('MyCompetition', competitor_a_team_id='human-1', competitor_b_team_id='human-2', competitor_a_id='random', competitor_b_id='random', config=config)
results = competition.run(num_games=10)
results.print()

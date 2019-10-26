from gym.envs.registration import register
from ffai.core.load import *

ruleset = get_rule_set('LRB5-Experimental')

register(
    id='FFAI-v1',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': get_config("ff-11.json"),
            'home_team': get_team_by_filename('human', ruleset, board_size=11),
            'away_team': get_team_by_filename('human', ruleset, board_size=11)
            }
)

register(
    id='FFAI-7-v1',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': get_config("ff-7.json"),
            'home_team': get_team_by_filename('human-7', ruleset, board_size=7),
            'away_team': get_team_by_filename('human-7', ruleset, board_size=7)
            }
)

register(
    id='FFAI-5-v1',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': get_config("ff-5.json"),
            'home_team': get_team_by_filename('human-5', ruleset, board_size=5),
            'away_team': get_team_by_filename('human-5', ruleset, board_size=5)
            }
)

register(
    id='FFAI-3-v1',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': get_config("ff-3.json"),
            'home_team': get_team_by_filename('human-3', ruleset, board_size=3),
            'away_team': get_team_by_filename('human-3', ruleset, board_size=3)
            }
)

register(
    id='FFAI-1-v1',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': get_config("ff-1.json"),
            'home_team': get_team_by_filename('human-1', ruleset, board_size=1),
            'away_team': get_team_by_filename('human-1', ruleset, board_size=1)
            }
)

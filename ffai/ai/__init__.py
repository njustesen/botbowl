from gym.envs.registration import register
from ffai.core.load import *

ruleset = load_rule_set('LRB5-Experimental')

register(
    id='FFAI-v1',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': load_config("ff-11"),
            'home_team': load_team_by_filename('human', ruleset, board_size=11),
            'away_team': load_team_by_filename('human', ruleset, board_size=11)
            }
)

register(
    id='FFAI-11-v1',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': load_config("ff-11"),
            'home_team': load_team_by_filename('human', ruleset, board_size=11),
            'away_team': load_team_by_filename('human', ruleset, board_size=11)
            }
)

register(
    id='FFAI-7-v1',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': load_config("ff-7"),
            'home_team': load_team_by_filename('human-7', ruleset, board_size=7),
            'away_team': load_team_by_filename('human-7', ruleset, board_size=7)
            }
)

register(
    id='FFAI-5-v1',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': load_config("ff-5"),
            'home_team': load_team_by_filename('human-5', ruleset, board_size=5),
            'away_team': load_team_by_filename('human-5', ruleset, board_size=5)
            }
)

register(
    id='FFAI-3-v1',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': load_config("ff-3"),
            'home_team': load_team_by_filename('human-3', ruleset, board_size=3),
            'away_team': load_team_by_filename('human-3', ruleset, board_size=3)
            }
)

register(
    id='FFAI-1-v1',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': load_config("ff-1"),
            'home_team': load_team_by_filename('human-1', ruleset, board_size=1),
            'away_team': load_team_by_filename('human-1', ruleset, board_size=1)
            }
)

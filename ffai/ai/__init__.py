from gym.envs.registration import register

from ffai.core.load import *
from .env import FFAIEnv
from .layers import *
from .registry import *
from .pathfinding import *
from .competition import *
from .renderer import *
from .proc_bot import *
from .bots import *

ruleset = load_rule_set('BB2016')


register(
    id='FFAI-v2',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': load_config("ff-11"),
            'home_team': load_team_by_filename('human', ruleset, board_size=11),
            'away_team': load_team_by_filename('human', ruleset, board_size=11)
            }
)

# Same as above but with explicit name
register(
    id='FFAI-11-v2',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': load_config("ff-11"),
            'home_team': load_team_by_filename('human', ruleset, board_size=11),
            'away_team': load_team_by_filename('human', ruleset, board_size=11)
            }
)

register(
    id='FFAI-7-v2',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': load_config("ff-7"),
            'home_team': load_team_by_filename('human-7', ruleset, board_size=7),
            'away_team': load_team_by_filename('human-7', ruleset, board_size=7)
            }
)

register(
    id='FFAI-5-v2',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': load_config("ff-5"),
            'home_team': load_team_by_filename('human-5', ruleset, board_size=5),
            'away_team': load_team_by_filename('human-5', ruleset, board_size=5)
            }
)

register(
    id='FFAI-3-v2',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': load_config("ff-3"),
            'home_team': load_team_by_filename('human-3', ruleset, board_size=3),
            'away_team': load_team_by_filename('human-3', ruleset, board_size=3)
            }
)

register(
    id='FFAI-1-v2',
    entry_point='ffai.ai.env:FFAIEnv',
    kwargs={'config': load_config("ff-1"),
            'home_team': load_team_by_filename('human-1', ruleset, board_size=1),
            'away_team': load_team_by_filename('human-1', ruleset, board_size=1)
            }
)

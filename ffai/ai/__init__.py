from gym.envs.registration import registry, register, make, spec
from bb.core.load import *

ruleset = get_rule_set('LRB5-Experimental.xml')

register(
    id='FFAI-v1',
    entry_point='bb.ai.env:FFAIEnv',
    kwargs={'config': get_config("ff-11.json"),
            'home_team': get_team('human-1', ruleset),
            'away_team': get_team('human-2', ruleset)
            }
)

register(
    id='FFAI-7-v1',
    entry_point='bb.ai.env:FFAIEnv',
    kwargs={'config': get_config("ff-7.json"),
            'home_team': get_team('human-1-7', ruleset),
            'away_team': get_team('human-2-7', ruleset)
            }
)

register(
    id='FFAI-5-v1',
    entry_point='bb.ai.env:FFAIEnv',
    kwargs={'config': get_config("ff-5.json"),
            'home_team': get_team('human-1-5', ruleset),
            'away_team': get_team('human-2-5', ruleset)
            }
)

register(
    id='FFAI-3-v1',
    entry_point='bb.ai.env:FFAIEnv',
    kwargs={'config': get_config("ff-3.json"),
            'home_team': get_team('human-1-3', ruleset),
            'away_team': get_team('human-2-3', ruleset)
            }
)

from gym.envs.registration import register

from .env import BotBowlEnv, EnvConf, BotBowlWrapper, RewardWrapper, ScriptedActionWrapper, PPCGWrapper
from .layers import *
from .registry import *
from .competition import *
from .renderer import *
from .proc_bot import *
from .bots import *

ruleset = load_rule_set('BB2016')

register(
    id='botbowl-v4',
    entry_point='botbowl.ai.env:BotBowlEnv',
    kwargs={}
)

register(
    id='botbowl-11-v4',
    entry_point='botbowl.ai.env:BotBowlEnv',
    kwargs={}
)

register(
    id='botbowl-7-v4',
    entry_point='botbowl.ai.env:BotBowlEnv',
    kwargs={'env_conf': EnvConf(size=7)}
)

register(
    id='botbowl-5-v4',
    entry_point='botbowl.ai.env:BotBowlEnv',
    kwargs={'env_conf': EnvConf(size=5)}
)

register(
    id='botbowl-3-v4',
    entry_point='botbowl.ai.env:BotBowlEnv',
    kwargs={'env_conf': EnvConf(size=3)}
)

register(
    id='botbowl-1-v4',
    entry_point='botbowl.ai.env:BotBowlEnv',
    kwargs={'env_conf': EnvConf(size=1)}
)

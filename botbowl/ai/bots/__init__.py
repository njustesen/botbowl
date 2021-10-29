from .random_bot import *
from .crash_bot import *
from .idle_bot import *
from .init_crash_bot import *
from .just_in_time_bot import *
from .manipulator_bot import *
from .violator_bot import *
from .illegal_action_bot import *


# Register bot(s)
register_bot('random', RandomBot)
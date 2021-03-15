import ffai
from ffai.core import Action, Agent

# Setup a game
config = ffai.load_config("bot-bowl-iii")
ruleset = ffai.load_rule_set(config.ruleset)
arena = ffai.load_arena(config.arena)
home = ffai.load_team_by_filename("human", ruleset)
away = ffai.load_team_by_filename("human", ruleset)
agent_home = Agent("home agent", human=True)
agent_away = Agent("home agent", human=True)
game = ffai.Game(1, home, away, agent_home, agent_away, config, arena=arena, ruleset=ruleset)

# Enable forward model
game.enable_forward_model()
step_id = game.get_forward_model_current_step()

# Force determinism?
# game.set_seed(1)  # Force determinism

# Take some actions
game.step(Action(ActionType.START_GAME))
game.step(Action(ActionType.HEADS))
game.step(Action(ActionType.TAILS))
game.step(Action(ActionType.RECEIVE))

# Kicking team is random if you didn't set the seed
print("Home is kicking: ", game.get_kicking_team() == game.state.home_team)

# Revert state
game.revert_state(step_id)

# Print available actions: Should only contain START_GAME
for action_choice in game.get_available_actions():
    print(action_choice.action_type.name)

# Output: START_GAME

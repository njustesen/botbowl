# Forward Model
Previously, it was difficult to get a fast forward model up and running in botbowl due to the reliance on the slow copy.deepcopy() function. Thanks to amazing work by Mattias Bermell, botbowl now has a built-in forward model that is reasonably fast. At least much faster that what we had before!

It works by tracking changes to non-immutable properties in the game state. Such changes can then be reverted to go back in time, e.g. to reset the state, where we had to completely reinstantiate the entire game object before.

Here's a small example showing how to first enable the forward model, then take some steps in the game, and finally revert back to the original state: 

```python
import botbowl
from botbowl.core import Action, Agent

# Setup a game
config = botbowl.load_config("bot-bowl-iii")
ruleset = botbowl.load_rule_set(config.ruleset)
arena = botbowl.load_arena(config.arena)
home = botbowl.load_team_by_filename("human", ruleset)
away = botbowl.load_team_by_filename("human", ruleset)
agent_home = Agent("home agent", human=True)
agent_away = Agent("home agent", human=True)
game = botbowl.Game(1, home, away, agent_home, agent_away, config, arena=arena, ruleset=ruleset)
game.init()

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
```
It is important that you initialize the game before you enabled the forward model.

This example script is also available [here](../examples/forward_model_example.py).

Notice, that with this forward model, we can go forward and the back. After this, we can go forward again, and then back to a previous state on the trajectory but we cannot revert to a step that lies forward in time or that we have already reverted away from. I.e. the tracked history of changes are cleared when we revert, and the information about the future is lost. 

If you want to re-visit an already visited state, you first have save the actions that got you there. Since the forward model is not reverting the state of the random number generator used by the game, you have to control its seed manually. Try to run the script a few times and you will notice that the kicking team is random. 

To force determinisim in the forward model, simply set the seed before stepping forward. Try to uncomment the line ```game.set_seed(1)``` and run the script a few times. You will see that the kicking team is always the same. If you want to re-visit a state after reverting, remember to set the seed again before stepping forward to reset the random number generator.

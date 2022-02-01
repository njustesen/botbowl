# Forward Model
Previously, it was difficult to get a fast forward model up and running in botbowl due to the reliance on the slow copy.deepcopy() function. Thanks to amazing work by Mattias Bermell, botbowl now has a built-in forward model that is reasonably fast. At least much faster that what we had before!

It works by tracking changes to non-immutable properties in the game state. Such changes can then be reverted to go back in time, e.g. to reset the state, where we had to completely reinstantiate the entire game object before.

Here's a small example showing how to first enable the forward model, then take some steps in the game, then revert back to the original state and finally revert it forward again: 

```python
import botbowl
from botbowl.core import Action, Agent, ActionType


def print_available_action_types(game):
    for action_choice in game.get_available_actions():
        print(action_choice.action_type.name, end=', ')
    print("\n", "-"*5, sep="")


def main():

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
    step_id = game.get_step()

    # Force determinism?
    # game.set_seed(1)  # Force determinism

    # Take some actions
    game.step(Action(ActionType.START_GAME))
    game.step(Action(ActionType.HEADS))
    game.step(Action(ActionType.RECEIVE))

    # Kicking team is random if you didn't set the seed
    print("Home is kicking: ", game.get_kicking_team() == game.state.home_team)
    print_available_action_types(game)

    # Revert state and save the steps
    steps = game.revert(step_id)

    # Print available actions: Should only contain START_GAME
    print_available_action_types(game)

    # step the game forward again
    game.forward(steps)
    print("Home is kicking: ", game.get_kicking_team() == game.state.home_team)
    print_available_action_types(game)


if __name__ == "__main__":
    main()
```
It is important that you initialize the game before you enabled the forward model.

This example script is also available [here](../examples/forward_model_example.py).

With this forward model, we can go forward and the back. After this, we can go forward again, and then back to a previous state on the trajectory. 

We can also forward revert to a state that we previously revert from but the forward model itself does not store _"the history of the future"_. So we have to manage the that ourselves. The forward model makes that easy, `game.revert()` returns the steps that was reverted. And we simply provide them as argument to ´game.forward()` to get back our future state.  

Notice that the random generator's state is not reverted. To force determinisim in the forward model you have to manually store the seed before taking actions and setting it after the revert, simply set the seed before stepping forward. 

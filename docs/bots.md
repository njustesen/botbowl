# Develop a Scripted Bot
This tutorial will introduce you to the Fantasy Football AI framework (FFAI) that allows you to make your own Blood Bowl bot in Python. First, I will explain how to download and set up the framework, then how to make a simple bot that uses FFAI’s API to retrieve information about the game state in order to make actions. Finally, I will introduce a fully fledged bot called GrodBot (developed by Peter Moore) that you can use as solid starting point.

If you end up developing your own bot, please submit it to the Bot Bowl II competition.

Make sure that FFAI is installed. If you haven't installed it yet, go to the [installation guide](installation.md).

## A Random Bot
Let’s start by making a bot that takes random actions. The code below, which can also be found in [examples/random_bot_example.py](../examples/random_bot_example.py), implements a bot that takes random actions.

```
#!/usr/bin/env python3

import ffai
import numpy as np


class MyRandomBot(ffai.Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.rnd = np.random.RandomState(seed)

    def new_game(self, game, team):
        self.my_team = team

    def act(self, game):
        # Select a random action type
        while True:
            action_choice = self.rnd.choice(game.state.available_actions)
            # Ignore PLACE_PLAYER actions
            if action_choice.action_type != ffai.ActionType.PLACE_PLAYER:
                break

        # Select a random position and/or player
        position = self.rnd.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
        player = self.rnd.choice(action_choice.players) if len(action_choice.players) > 0 else None

        # Make action object
        action = ffai.Action(action_choice.action_type, position=position, player=player)

        # Return action to the framework
        return action

    def end_game(self, game):
        pass


# Register the bot to the framework
ffai.register_bot('my-random-bot', MyRandomBot)


if __name__ == "__main__":

    # Load configurations, rules, arena and teams
    config = ffai.load_config("bot-bowl-ii")
    ruleset = ffai.load_rule_set(config.ruleset)
    arena = ffai.load_arena(config.arena)
    home = ffai.load_team_by_filename("human", ruleset)
    away = ffai.load_team_by_filename("human", ruleset)
    config.competition_mode = False
    config.debug_mode = False

    # Play 10 games
    game_times = []
    for i in range(10):
        away_agent = ffai.make_bot("my-random-bot")
        home_agent = ffai.make_bot("my-random-bot")

        game = ffai.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i+1))
        game.init()
        print("Game is over")

```

Let’s go through the code step by step. First, we import the ffai package as well as numpy. Then, we create a new class called MyRandomBot that inherits from the Agent class. Doing so, requires us to implement three functions:

* **new_game(self, game, team):** is called whenever a new game is started with this bot, telling it which team it is controlling as well as a few initial information in the game object (such as the name of the opponent).
* **act(self, game):** is called at every step in the game where the bot is supposed to perform an action. Here, the game object is given as well which contains information about the entire game state. This function must return an instance of the class Action which contains both an action type and optionally a position or a player.
* **end_game(self, game):** is called when the game is over. Information about the game, such as the score and who the winner is can be accessed from the game object.

Because we just want to take a random action for now, let’s forget about the game object. Instead, let’s look at the Action class that we need to instantiate whenever act is called.

```
class Action:

    def __init__(self, action_type, pos=None, player=None):
        ...
```

The only required parameter in the constructor is ```action_type```, which should be an instance of the enum ```ActionType```. You can see all the different action types in [ffai/core/table.py](../ffai/core/table.py). Here are some examples of actions that could be instantiated in a sequence of ```act()```-calls:

```
Action(ActionType.START_BLITZ, player=game.get_players_on_pitch(self.my_team)[0])
Action(ActionType.MOVE, position=Square(3,5))
Action(ActionType.MOVE, position=Square(3,6))
Action(ActionType.BLOCK, position=Square(4,7))
Action(ActionType.SELECT_DEFENDER_DOWN)
Action(ActionType.FOLLOW_UP)
Action(ActionType.BLOCK, position=Square(4,8))
Action(ActionType.END_PLAYER_TURN)
```

But how do we know which actions that are allowed in the current step of the game? The game object contains a list of the available action choices in the state:

```
game.state.available_actions
```

This is a list of possible action choices that can be performed with some additional information about them, such as the required dice roll to make. An example of this list, formatted in json, looks like this:

```
"available_actions": [
     {
         "action_type": "MOVE", 
         "positions": [{"x": 12, "y": 6}, {"x": 14, "y": 6}, {"x": 12, "y": 7}, {"x": 12, "y": 8}], 
         "team_id": "human-1", 
         "rolls": [], 
         "block_rolls": [], 
         "agi_rolls": [[3], [5], [3], [3]], 
         "player_ids": [], 
         "disabled": false
     }, 
     {
         "action_type": "BLOCK", 
         "positions": [{"x": 14, "y": 7}, {"x": 14, "y": 8}], 
         "team_id": "human-1", 
         "rolls": [], 
         "block_rolls": [1, 1], 
         "agi_rolls": [[], []], 
         "player_ids": [], 
         "disabled": false
     }, 
     {
         "action_type": "END_PLAYER_TURN", 
         "positions": [], 
         "team_id": "human-1", 
         "rolls": [], 
         "block_rolls": [], 
         "agi_rolls": [], 
         "player_ids": [], 
         "disabled": false
     }
 ]
```
which are the available actions in this situation:

!["A human lineman has taken a Blitz action and can thus both move and block."](screenshots/block_situation.png?raw=true "Block situation")

By iterating the available actions, we can easily select one that we like. For our random bot, we first sample a random ```ÀctionType```:

```
action_choice = self.rnd.choice(game.state.available_actions)
```

We do not want to sample the ActionType.PLACE_PLAYER, which is used during the setup phase, as we don’t want to rely in our bot to randomly come up with a valid starting formation. Instead, we allow it to select one of the built-in starting formations that are available as actions. After selecting an action type, we can sample a position or player if it is needed:

```
pos = self.rnd.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
player = self.rnd.choice(action_choice.players) if len(action_choice.players) > 0 else None
```

Finally, we can instantiate the Action object and return it:

```
action = Action(action_choice.action_type, pos=pos, player=player)
return action
```

To play against you agent in the web interface, add the following the your bot script, and start a new server.

```
register_bot('my-random-bot', MyRandomBot)
server.start_server(debug=True, use_reloader=False)
```

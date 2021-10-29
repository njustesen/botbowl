# Scripted Bots III: Setup

Kick-off formations are incredibly important for both the offense and defense. It is, however, also incredibly difficult to write a script the does this successfully for several reasons. First of all, the task of setting up players before kick-off 
requires extensive long-term planning. The offense team needs to consider of there is a need for scoring quickly, slowly, or perhaps not at all but rather just maintain the possession of the ball. Additionally, the offensive team must consider the context in great detail. How many players do I have left? Which player should pickup the ball? Who should make the first blitz 
action? Who should move into scoring range? The defensive team has an additional challenge: adaptation. While the offense more or less can use the same setup formation every time (with some players potentially missing), the defense has to react to the offense formation. Such adaptation is perhaps one of the greatest challenges in game AI. One can easily write a script for a single 
setup formation While it is infeasible to describe a ruleset that can adapt the defensive setup to any posible offensive formation.

In this tutorial, we do not attempt to solve the challenges of adapting to the opponent. Instead, we start developing a technique for fixed setup formations that, however, can adapt to missing players as some will be injured or knocked out during the game.

## Formation syntax
We use a simple syntax to describe kick-off formations in a similar style to [https://fumbbl.com/help:Offensive+Setups](FUMBBL's setup guides).
An example of an offensive setup formation looks like this:

```
-------------
-------------
-----------m-
-------------
------------x
------------S
------------x
-----s---0--S
------------x
------------S
------------x
-------------
-----------m-
-------------
-------------
```

A setup is always seen from the left-field perspective and is always 13x15 (half of the field) unless a smaller board size is used. 

The characters in the syntax means:

- **-**: No player should be placed here.
- **S**: Prioritize players with high strength and block and avoid player with __Sure Hands__.
- **s**: Prioritize players with high strength.
- **m**: Prioritize players with high movement allowance.
- **a**: Prioritize players with high agility.
- **v**: Prioritize players with high armour value.
- **s**: Prioritize players with __Sure Hands__.
- **p**: Prioritize players with __Pass__.
- **c**: Prioritize players with __Catch__.
- **b**: Prioritize players with __Block__.
- **d**: Prioritize players with __Dodge__.
- **0**: Prioritize players with no skills.
- **x**: Prioritize players with __Block__ and avoid __Pass__/__Catch__ players.

## The Formation class
botbowl comes with an implementation of a Formation class with a constructor that takes a 2D array of strings, similar to the syntax above. 

Its ```actions(self, game, team)``` function will then return a list of actions to execute the described setup formation. 
botbowl also has a loader if you prefer to write formations in a text file exactly as above. Load a formation from a file like 
this:

````python
formation = load_formation(filename, directory=directory_path)
actions = formation.actions(game, my_team)
````

botbowl also comes with two offensive and two defensive formations that you can use out of the box like this:

````python
off_formation_line = load_formation("off_line")
def_formation_wedge = load_formation("off_wedge")
def_formation_spread = load_formation("def_spread")
def_formation_spread = load_formation("def_spread")
````
Notice, that if no directory is specified, the default location inside botbowl is used.

## Custom formations
Our scripted bot from the previous tutorial we will use a fixed offensive and a fixed defensive formation:

```python
self.off_formation = [
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "m", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "S"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x"],
    ["-", "-", "-", "-", "-", "s", "-", "-", "-", "0", "-", "-", "S"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "S"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "m", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
]

self.def_formation = [
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "b", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "S", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "0"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "0"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "0"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "S", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "b", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
    ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
]
self.off_formation = Formation("Wedge offense", self.off_formation)
self.def_formation = Formation("Zone defense", self.def_formation)
self.setup_actions = []
```

We can then use the formations in the ```Setup()``` function of our bot like this:

```python
def setup(self, game):
    self.my_team = game.get_team_by_id(self.my_team.team_id)
    self.opp_team = game.get_opp_team(self.my_team)
    if self.setup_actions:
        action = self.setup_actions.pop(0)
        return action
    else:
        if game.get_receiving_team() == self.my_team:
            self.setup_actions = self.off_formation.actions(game, self.my_team)
            self.setup_actions.append(Action(ActionType.END_SETUP))
        else:
            self.setup_actions = self.def_formation.actions(game, self.my_team)
            self.setup_actions.append(Action(ActionType.END_SETUP))
```

Try experimenting with a few different setup formations or implement a larger repository of formations for different situations.
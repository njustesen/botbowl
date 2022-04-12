# Monte Carlo Tree Search

In the previous tutorial, we implemented a simple one-step look-ahead search. 
Let's extend that into the famous algorithm Monte Carlo Tree Search (MCTS). 
If you are unfamiliar with the algorithm, check out the seminal paper introducing the algorithm ([Monte-carlo Tree Search: A New Framework for Game AI](https://ojs.aaai.org/index.php/AIIDE/article/download/18700/18475)) and perhaps this survey paper ([A Survey of Monte Carlo Tree Search Methods](http://repository.essex.ac.uk/4117/1/MCTS-Survey.pdf)).

The code for this tutorial can be found in [examples/mcts_example.py](https://github.com/njustesen/botbowl/blob/main/examples/mcts_example.py) and [examples/hash_example.py](https://github.com/njustesen/botbowl/blob/main/examples/hash_example.py).

## Nodes

The first thing we will do is to create two types of nodes; ActionNode and ChanceNode. 
An ActionNode represents a state wherein an agent can take actions and a ChanceNode represents the branching of possible random outcomes.
The search tree our MCTS implementation will construct will have alternating ActionNodes and ChanceNodes. 
An agent takes an action resulting in one or more outcomes, then the same or another agent takes an action resulting in one or more outcomes again. 

```python
class Node:
    def __init__(self, action=None, parent=None):
        self.parent = parent
        self.children = []
        self.action = action
        self.evaluations = []

    def num_visits(self):
        return len(self.evaluations)

    def visit(self, score):
        self.evaluations.append(score)

    def score(self):
        return np.average(self.evaluations)
```

## The Bot
Now we will implement a bot that relies its decision making on a one-step search with a simple heuristic.

```python
class SearchBot(botbowl.Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.rnd = np.random.RandomState(seed)

    def new_game(self, game, team):
        self.my_team = team

    def act(self, game):
        game_copy = deepcopy(game)
        game_copy.enable_forward_model()
        game_copy.home_agent.human = True
        game_copy.away_agent.human = True

        root_step = game_copy.get_step()
        root_node = Node()
        for action_choice in game_copy.get_available_actions():
            if action_choice.action_type == botbowl.ActionType.PLACE_PLAYER:
                continue
            for player in action_choice.players:
                root_node.children.append(Node(Action(action_choice.action_type, player=player), parent=root_node))
            for position in action_choice.positions:
                root_node.children.append(Node(Action(action_choice.action_type, position=position), parent=root_node))
            if len(action_choice.players) == len(action_choice.positions) == 0:
                root_node.children.append(Node(Action(action_choice.action_type), parent=root_node))

        best_node = None
        print(f"Evaluating {len(nodes)} nodes")
        t = time.time()
        for node in nodes:
            game_copy.step(node.action)
            while not game.state.game_over and len(game.state.available_actions) == 0:
                game_copy.step()
            score = self._evaluate(game)
            node.visit(score)
            print(f"{node.action.action_type}: {node.score()}")
            if best_node is None or node.score() > best_node.score():
                best_node = node

            game_copy.revert(root)

        print(f"{best_node.action.action_type} selected in {time.time() - t} seconds")

        return best_node.action

    def _evaluate(self, game):
        return random.random()

    def end_game(self, game):
        pass
```

Let's walk through the interesting parts of the code.

```python
game_copy = deepcopy(game)
game_copy.enable_forward_model()
game_copy.home_agent.human = True
game_copy.away_agent.human = True
```

Here, we first make a copy of the entire game object so we don't manipulate with the actual game we are playing. In a competition, you wouldn't need this because you are already handed a copy of the game, but in our little example here it is important. Then we enable the forward model, meaning that state changes will be tracked so they can be reverted. The next part looks a little weird. We set the two agents in the game as *humans*. This just means that we can take steps in the game without agents being asked by the engine to take the next actions. We have basically detached the agents from our copy of the game. 

After this, we make a root node representing the current state of the game and create child nodes for each available action.

```python
root_step = game_copy.get_step()
root_node = Node()
for action_choice in game_copy.get_available_actions():
    if action_choice.action_type == botbowl.ActionType.PLACE_PLAYER:
        continue
    for player in action_choice.players:
        root_node.children.append(Node(Action(action_choice.action_type, player=player), parent=root_node))
    for position in action_choice.positions:
        root_node.children.append(Node(Action(action_choice.action_type, position=position), parent=root_node))
    if len(action_choice.players) == len(action_choice.positions) == 0:
        root_node.children.append(Node(Action(action_choice.action_type), parent=root_node))
```

We then evaluate each child node and pick the action of the highest valued node.

```python
best_node = None
print(f"Evaluating {len(root_node.children)} nodes")
t = time.time()
for node in root_node.children:
    game_copy.step(node.action)
    while not game.state.game_over and len(game.state.available_actions) == 0:
        game_copy.step()
    score = self._evaluate(game)
    node.visit(score)
    print(f"{node.action.action_type}: {node.score()}")
    if best_node is None or node.score() > best_node.score():
        best_node = node

    game_copy.revert(root_step)
```

Notice that the state evluation is just a random value between 0 and 1. 

```python
def _evaluate(self, game):
    return random.random()
```

## Next Steps
You find the script in [examples/search_example.py](../examples/search_example.py).

- Implement the ```_evaluate(game)``` function so it actually evaluates the game state.
- Try with pathfinding enabled to search among pathfinding-assisted move actions.
- Can you extend this example into a [Monte-Carlo Tree Search](https://www.aaai.org/Papers/AIIDE/2008/AIIDE08-036.pdf)?

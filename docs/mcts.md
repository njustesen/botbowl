# Monte Carlo Tree Search

In [the previous tutorial](https://github.com/njustesen/botbowl/blob/main/docs/search-based.md), we implemented a simple one-step look-ahead search. 
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

    def __init__(self):
        self.evaluations = []

    def num_visits(self):
        return len(self.evaluations)

    def visit(self, score):
        self.evaluations.append(score)

    def score(self):
        return np.average(self.evaluations)

    def print(self):
        raise NotImplementedError


class ActionNode(Node):

    def __init__(self, game, hash_key, opp=False):
        super().__init__()
        self.hash_key = hash_key
        self.available_actions = self._extract_actions(game)
        self.children: List[ChangeNode] = []
        self.opp = opp  # Is it the opponent turn to move?
        self.terminal = game.state.game_over

    def _extract_actions(self, game):
        actions = []
        for action_choice in game.get_available_actions():
            if action_choice.action_type == botbowl.ActionType.PLACE_PLAYER:
                continue
            if action_choice.action_type == botbowl.ActionType.END_SETUP:
                continue
            if len(action_choice.players) > 0:
                for player in action_choice.players:
                    actions.append(Action(action_choice.action_type, position=None, player=player))
            elif len(action_choice.positions) > 0:
                for position in action_choice.positions:
                    actions.append(Action(action_choice.action_type, position=position))
            else:
                actions.append(Action(action_choice.action_type))
        return actions

    def is_fully_expanded(self):
        return len(self.children) == len(self.available_actions)

    def print(self, tabs=0):
        ...


class ChangeNode(Node):

    def __init__(self, parent, action):
        super().__init__()
        self.parent = parent
        self.action = action
        self.outcomes = {}  # hash: Node
        self.evaluations = []
        self.terminal = False

    def print(self, tabs=0):
        ...
```

A few interesting things are going on here. 

First, notice that we don't copy and store game state in nodes. 
This would be too expensive because game states are big in Blood Bowl and they contain many objects.
Instead we store an action in each ChanceNode and we will step forward in our forward model to recreate already seen states.

First, notice that we extract all the different actions as individual Action objects. 
Because setup is tricky, we ignore PLACE_PLAYER and END_SETUP actions, leaving out the built-in formation actions.
Later, we make sure that we automatically perform the END_SETUP action right after performing a formation action.

Second, notice the `is_fully_expanded()` function. 
It will check if we have tried every action at least once which is important for our tree policy later.

Finally, the recursive `print()` functions will come in handy if we want to inspect the search tree.

## MCTS
Now, let's write the actual search algorithm.

```python
class MCTS:

    def __init__(self, game, agent, tree_policy, action_policy, heuristic):
        self.game = game
        self.agent = agent
        self.tree_policy = tree_policy
        self.action_policy = action_policy
        self.heuristic = heuristic

    def run(self, seconds):
        t = time.time()
        hash_key = gamestate_hash(self.game)
        root = ActionNode(self.game, hash_key)
        step = self.game.get_step()
        while time.time() < t + seconds:
            tree_trajectory = self._select_and_expand(root)
            self._rollout()
            value = self.heuristic(self.game, self.agent)
            self._backpropagate(tree_trajectory, value)
            self.game.revert(step)
            # root.print()  # <- use this to print the tree at every step
        return root
```

If you made yourself familiar with MCTS this shouldn't be too surprising. 

The constructor takes a few interesting arguments, including `tree_policy` (used to traverse the existing tree), 
`action_policy` (used to sample actions during rollouts), and `heuristic` (used to evaluate game states so rollouts don't have to run until the end of the game).

The `gamestate_hash(self.game)` is important, as it is used to check if two states are similar. 
This is used when an action of a ChanceNode is performed and we need to check if the outcome is new or if we have seen it before.

We store the trajectory which is used during backpropagation in contrast to most implementations of MCTS that just follow the `parent` property until the root is reached.
We could easily do that here but in case you want to extend this implementation to use transpositions, this is the preferred way.

## Selection

````python
    def _select_and_expand(self, root):
        node = root
        trajectory = [root]
        while node.is_fully_expanded():
            if node.terminal:
                return trajectory
            best_child = self.tree_policy(node)
            self.game.step(best_child.action)
            if "SETUP_" in best_child.action.action_type.name:
                self.game.step(Action(botbowl.ActionType.END_SETUP))
            trajectory.append(best_child)
            hash_key = gamestate_hash(self.game)
            if hash_key not in best_child.outcomes:
                node = ActionNode(self.game, hash_key, self.game.actor == self.agent)
                best_child.outcomes[hash_key] = node
                trajectory.append(node)
                return trajectory
            else:
                node = best_child.outcomes[hash_key]
            trajectory.append(node)
        new_chance_node, new_node = self._expand(node)
        trajectory.append(new_chance_node)
        trajectory.append(new_node)
        return trajectory
````

This function selects a node from the existing tree to expand from and perform a rollout from.

Notice, how the `tree_policy` is used to select the most promising node.

````python
def ucb1(node, c=0.707):
    best_node = None
    best_score = None
    maximize = True
    if type(node) is ActionNode and node.opp():
        maximize = False
    for child in node.children:
        mean_score = child.score() if maximize else 1-child.score()
        ucb_score = mean_score + 2*c * np.sqrt((2 * np.log(node.num_visits())) / child.num_visits())
        if best_score is None or ucb_score > best_score:
            best_node = child
            best_score = ucb_score
    return best_node
````

The default choice is UCB1 and we use a simple modification that minimizes the score if the opponent is making a choice.

### Expansion

When a non-fully expanded node is reached in the selection phase, the `_expand` function will be called. 

```python
def _expand(self, node: ActionNode):
    next_action_idx = len(node.children)
    action = node.available_actions[next_action_idx]
    chance_node = ChangeNode(node, action)
    node.children.append(chance_node)
    self.game.step(action)
    if "SETUP_" in action.action_type.name:
        self.game.step(Action(botbowl.ActionType.END_SETUP))
    hash_key = gamestate_hash(self.game)
    node = ActionNode(self.game, hash_key, self.game.actor == self.agent)
    chance_node.outcomes[node.hash_key] = node
    return chance_node, node
```

Here, we perform the next non-visited action, add a `ChanceNode` to represent that action, and a new `ActionNode` representing the outcome.

### Simulation

From the expanded node, we perform a rollout until the next turn.

```python
def _rollout(self):
    turns = self.game.state.home_team.state.turn + self.game.state.away_team.state.turn
    while not self.game.state.game_over and turns == self.game.state.home_team.state.turn + self.game.state.away_team.state.turn:
        action = self.action_policy(self.game, self.game.get_agent_team(self.agent))
        self.game.step(action)
        if "SETUP_" in action.action_type.name:
            self.game.step(Action(botbowl.ActionType.END_SETUP))
```

Remember, that because we removed the `END_SETUP` action from the action space, we must perform it right after performing a formation action (they always have the "SETUP_" prefix").

At the end of the rollout, we apply a heuristic. We implement a super simple one that considers the score and material values.

```python
def simple_heuristic(game: botbowl.Game, agent:botbowl.Agent):
    own_team = game.get_agent_team(agent)
    opp_team = game.get_opp_team(own_team)
    own_score = own_team.state.score
    opp_score = opp_team.state.score
    own_kos = len(game.get_knocked_out(own_team))
    opp_kos = len(game.get_knocked_out(opp_team))
    own_cas = len(game.get_casualties(own_team))
    opp_cas = len(game.get_casualties(opp_team))
    own_stunned = len([p for p in game.get_players_on_pitch(own_team, up=False) if p.state.stunned])
    opp_stunned = len([p for p in game.get_players_on_pitch(opp_team, up=False) if p.state.stunned])
    own_down = len([p for p in game.get_players_on_pitch(own_team, up=False) if not p.state.stunned])
    opp_down = len([p for p in game.get_players_on_pitch(opp_team, up=False) if not p.state.stunned])
    own_ejected = len(game.get_dungeon(own_team))
    opp_ejected = len(game.get_dungeon(opp_team))
    own_has_ball = False
    opp_has_ball = False
    ball_carrier = game.get_ball_carrier()
    if ball_carrier is not None:
        own_has_ball = 1 if ball_carrier.team == own_team else 0
        opp_has_ball = 1 if ball_carrier.team == opp_team else 0
    own = own_score/10 + own_has_ball/20 - (own_cas + own_ejected)/30 - own_kos/50 - own_stunned/100 - own_down/200
    opp = opp_score/10 + opp_has_ball/20 - (opp_cas + opp_ejected)/30 - opp_kos/50 - opp_stunned/100 - opp_down/200
    if game.state.game_over:
        if game.get_winner() == agent:
            return 1
        elif game.get_winner() is None:
            return 0.5
        else:
            return -1
    return 0.5 + own - opp
```

We aim to have the value returned by the heuristic to be between 0 and 1, which is an assumption made by UCB1, such that 1 is a guaranteed win and 0 is a guaranteed loss. 
Our heuristic here is, however, a bit wonky but loosely follows that idea.

### Backpropagation
After the rollout, we only need the backpropagation phase which is very simple. 

```python
def _backpropagate(self, trajectory, score):
    for node in reversed(trajectory):
        node.visit(score)
```

### MCTS Agent

Now that we have an MCTS implementation, we just need to wrap it into an Agent implementation so we can use it in the botbowl framework.

````python
class MCTSBot(botbowl.Agent):

    def __init__(self,
                 name,
                 tree_policy=ucb1,
                 action_policy=random_policy,
                 final_policy=most_visited,
                 heuristic=simple_heuristic,
                 seconds=2,
                 seed=None):
        super().__init__(name)
        self.my_team = None
        self.rng = np.random.RandomState(seed)
        self.tree_policy = tree_policy
        self.action_policy = action_policy
        self.final_policy = final_policy
        self.heuristic = heuristic
        self.seconds = seconds
        self.next_action = None

    def new_game(self, game, team):
        self.my_team = team

    def act(self, game):
        if self.next_action is not None:
            action = self.next_action
            self.next_action = None
            return action
        game_copy = deepcopy(game)
        game_copy.enable_forward_model()
        game_copy.home_agent.human = True
        game_copy.away_agent.human = True
        mcts = MCTS(game_copy,
                    self,
                    tree_policy=self.tree_policy,
                    action_policy=self.action_policy,
                    heuristic=self.heuristic)
        root = mcts.run(self.seconds)
        # root.print()
        best_node = self.final_policy(root)
        # print(f"Found action {best_node.action.action_type} with {root.num_visits()} rollouts.")
        action = best_node.action
        if "SETUP_" in action.action_type.name:
            self.next_action = Action(botbowl.ActionType.END_SETUP)
        else:
            self.next_action = None
        return action

    def end_game(self, game):
        pass
````

Two things are important to note here. 

First, we implement a feature to fix the next action, which we will use to perform the `END_SETUP` action after a formation action.

Second, is the use of the `final_policy` that is used to select the action to perform after the search is over.
Among common strategies are selecting the action with the highest value, highest visit count, or a combination.  
Here, we select the most visited action, which is a conservative but simple strategy, and add the ucb1 score as a tie breaker.

````python
def most_visited(node):
    return max(node.children, key=lambda x: x.num_visits() + x.score())
````

## Performance

Let's see how well our MCTS bot performs on various board sizes. 
We played two versions of MCTS with a time budget of 5 seconds per decision, 10 times against the Random bot as home and 10 times as away.
The first version uses rollouts until the next turn and the second version uses no rollouts.

| Rollouts | Env | Home Wins | Home TDs | Away Wins | Away TDs | Total Wins | Total TDs | AVG Wins | AVG TDs | Iterations |
|----------|----:|----------:|---------:|----------:|---------:|-----------:|----------:|---------:|--------:|-----------:|
| Yes      |   1 |      9/10 |       35 |     10/10 |       41 |      19/20 |        76 |     0.95 |     3.8 |       1592 |
| No       |   1 |     10/10 |       50 |      8/10 |       48 |      18/20 |        98 |     0.90 |     4.9 |       2038 |
| Yes      |   3 |      9/10 |       11 |      8/10 |        9 |      17/20 |        20 |     0.85 |       1 |        688 |
| No       |   3 |     10/10 |       26 |      9/10 |       18 |      19/20 |        45 |     0.85 |    2.25 |       1735 |
| Yes      |   5 |      6/10 |        7 |      0/10 |        0 |       6/20 |         7 |     0.30 |    0.35 |        555 |
| No       |   5 |     10/10 |       23 |      5/10 |        5 |      15/20 |        28 |     0.75 |     1.4 |       1299 |
| Yes      |  11 |      2/10 |        2 |      0/10 |        0 |       2/20 |         2 |     0.10 |     0.1 |        361 |
| No       |  11 |      7/10 |       12 |      0/10 |        0 |       7/20 |        12 |     0.35 |     0.6 |        951 |

We see that MCTS is able to score and win against random on all the board sizes. 
It is, however, striking that MCTS plays a lot better as the home team than the away team.
This is because actions are expanded left to right and thus actions that moves players towards the away team's endzone are prioritized first.
To improve our MCTS agent when playing as away, we either need to flip the board or have a better move ordering when we expand.

Our results gives us another key insight: on medium and large board sizes, MCTS is better when we don't do rollouts.
This is probably because a) rollouts are expensive in Blood Bowl and b) doing random actions is Blood Bowl is risky and more often than not results in failed dodges.

The number of TDs is not quite on par with our Reinforcement Learning agents on the smaller variants while our MCTS is able to score on the full board!

Considering that this is almost a vanilla implementation of MCTS, with just a few small enhancements, it already looks promising. 

## Search Tree Inspection

Let's take a look at how MCTS does in the following game situation, playing as the blue team.

![MCTS game](img/mcts-game.png?raw=true "MCTS Game")

```xml
<ActionNode p='max' visits=483 score=0.4984368530020704 actions=7>
	<ChanceNode p='max' visits=68 score=0.4952450980392156 action='{'action_type': 'START_MOVE', ...}'/>
	<ChanceNode p='max' visits=69 score=0.5004347826086958 action='{'action_type': 'START_BLOCK', ...}'>
		<ActionNode p='max' visits=69 score=0.5004347826086958 actions=3>
			<ChanceNode p='max' visits=23 score=0.5006521739130435 action='{'action_type': 'BLOCK', ...}'>
				<ActionNode p='max' visits=3 score=0.5033333333333333 actions=2/>
				<ActionNode p='max' visits=3 score=0.49833333333333335 actions=2/>
				<ActionNode p='max' visits=3 score=0.5083333333333333 actions=2/>
				<ActionNode p='max' visits=7 score=0.4985714285714286 actions=2/>
				<ActionNode p='max' visits=7 score=0.4992857142857143 actions=2/>
			</ChanceNode>
			<ChanceNode p='max' visits=23 score=0.5049275362318841 action='{'action_type': 'END_PLAYER_TURN', ...}'/>
		</ActionNode>
	</ChanceNode>
	<ChanceNode p='max' visits=70 score=0.5015238095238095 action='{'action_type': 'START_BLITZ', 'position': None, 'player_id': '91df3c08-bb32-11ec-ae08-acde48001122'}'>
		<ActionNode p='max' visits=70 score=0.5015238095238095 actions=13>
			<ChanceNode p='max' visits=5 score=0.49400000000000005 action='{'action_type': 'MOVE', 'position': {'x': 1, 'y': 1}, ...}'/>
			<ChanceNode p='max' visits=5 score=0.49700000000000005 action='{'action_type': 'MOVE', 'position': {'x': 1, 'y': 2}, ...}'/>
			<ChanceNode p='max' visits=5 score=0.496 action='{'action_type': 'MOVE', 'position': {'x': 1, 'y': 3}, ...}'/>
			<ChanceNode p='max' visits=5 score=0.49800000000000005 action='{'action_type': 'MOVE', 'position': {'x': 2, 'y': 1}, ...}'/>
			<ChanceNode p='max' visits=6 score=0.505 action='{'action_type': 'MOVE', 'position': {'x': 2, 'y': 3}, ...}'/>
			<ChanceNode p='max' visits=5 score=0.496 action='{'action_type': 'MOVE', 'position': {'x': 3, 'y': 1}, ...}'/>
			<ChanceNode p='max' visits=5 score=0.496 action='{'action_type': 'MOVE', 'position': {'x': 3, 'y': 3}, ...}'/>
			<ChanceNode p='max' visits=6 score=0.5066666666666667 action='{'action_type': 'MOVE', 'position': {'x': 4, 'y': 1}, 'player_id': None}'/>
			<ChanceNode p='max' visits=6 score=0.5325000000000001 action='{'action_type': 'MOVE', 'position': {'x': 4, 'y': 2}, 'player_id': None}'>
				<ActionNode p='max' visits=1 score=0.55 actions=2/>
				<ActionNode p='max' visits=2 score=0.55 actions=2>
					<ChanceNode p='max' visits=1 score=0.55 action='{'action_type': 'USE_REROLL', 'position': None, 'player_id': None}'>
						<ActionNode p='max' visits=1 score=0.55 actions=12/>
					</ChanceNode>
				</ActionNode>
				<ActionNode p='max' visits=3 score=0.515 actions=12>
					<ChanceNode p='max' visits=1 score=0.495 action='{'action_type': 'MOVE', 'position': {'x': 1, 'y': 1}, ...}'>
						<ActionNode p='max' visits=1 score=0.495 actions=2/>
					</ChanceNode>
					<ChanceNode p='max' visits=1 score=0.495 action='{'action_type': 'MOVE', 'position': {'x': 1, 'y': 2}, ...}'>
						<ActionNode p='max' visits=1 score=0.495 actions=2/>
					</ChanceNode>
				</ActionNode>
			</ChanceNode>
			<ChanceNode p='max' visits=5 score=0.4923333333333334 action='{'action_type': 'MOVE', 'position': {'x': 4, 'y': 3}, .../>
			<ChanceNode p='max' visits=5 score=0.497 action='{'action_type': 'BLOCK', 'position': {'x': 2, 'y': 2}, .../>
			<ChanceNode p='max' visits=6 score=0.5025 action='{'action_type': 'END_PLAYER_TURN', ...}'/>
		</ActionNode>
	</ChanceNode>
	<ChanceNode p='max' visits=68 score=0.4944362745098039 action='{'action_type': 'START_PASS', ...}'/>
	<ChanceNode p='max' visits=69 score=0.4978743961352657 action='{'action_type': 'START_HANDOFF', ...}'/>
	<ChanceNode p='max' visits=69 score=0.4978743961352657 action='{'action_type': 'START_FOUL', ...}'/>
	<ChanceNode p='max' visits=70 score=0.5014761904761904 action='{'action_type': 'END_TURN', ...}'/>
</ActionNode>
Found action {'action_type': 'START_BLITZ', 'position': None, 'player_id': '91df3c08-bb32-11ec-ae08-acde48001122'} with 483 rollouts.
```

By inspecting the game tree we notice that it prefers to start a Blitz action and then move towards the ball. 
We can further see that it did a few rollouts trying to move to the endzone but it probably failed every time. 
The alternative move it seriously considers is blocking the opponent but it's not clear that it will gain an advantage.

## Next Steps
Here are some suggestions to further improve the MCTS bot:

- Better move ordering or action sampling, e.g. with a learned policy model. 
- Better time management. 
- Better heuristics, e.g. with a learned game state evaluation model.
- Apply parallelization techniques?
- Why not try out AlphaZero?
- There are possibly several performance improvements to be made
- Tweak and tune the parameters such as the exploration constant `c`
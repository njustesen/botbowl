# Reinforcement Learning I: OpenAI Gym Environment
This tutorial will introduce you to botbowl's implementations of the [Open AI Gym interface](https://arxiv.org/pdf/1606.01540.pdf) that will allow for easy integration of reinforcement learning algorithms. 

You can run [examples/gym_example.py](examples/gym_example.py) to se a random agent play Blood Bowl through the botbowl Gym environment. The rendering is simplified for faster execution and looks like this:
![botbowl Gym GUI](https://njustesen.github.io/botbowl/img/gym.png?raw=true "botbowl Gym GUI botbowl-3")

[examples/gym_example.py](examples/multi_gym_example.py) demonstrates how you can run multiple instance of the environment in parallel. Notice, that the render() function doesn't work across multiple processes. Instead a custom renderer is used in this example.

Agents receive numerical observations from the botbowl environment at every step and sends back and action with an action type and in some cases a position. Along with the observations, the environment also sends a scalar reward value to the agent. We will describe the structure of the three components: observations, actions, and rewards.

## The step function 
Now we will talk environments step() function which is called like this: 
```python
spatial_obs, reward, done, info = env.step(action) 
```

### Spatial observation
`spatial_obs` is a 3 dimensional numpy array with `shape=(num_layer, height, width)`. This is all the features layers stack together. 
The default feature layers are: 

0. OccupiedLayer()
1. OwnPlayerLayer()
2. OppPlayerLayer()
3. OwnTackleZoneLayer()
4. OppTackleZoneLayer()
5. UpLayer()
6. StunnedLayer()
7. UsedLayer()
8. AvailablePlayerLayer()
9. AvailablePositionLayer()
10. RollProbabilityLayer()
11. BlockDiceLayer()
12. ActivePlayerLayer()
13. TargetPlayerLayer()
14. MALayer()
15. STLayer()
16. AGLayer()
17. AVLayer()
18. MovemenLeftLayer()
19. BallLayer()
20. OwnHalfLayer()
21. OwnTouchdownLayer()
22. OppTouchdownLayer()
23. SkillLayer(Skill.BLOCK)
24. SkillLayer(Skill.DODGE)
25. SkillLayer(Skill.SURE_HANDS)
26. SkillLayer(Skill.CATCH)
27. SkillLayer(Skill.PASS)

A layer is a 2-D array of scalars in [0,1] with the size of the board including __crowd__ padding. Some layers have binary values, e.g. indicating whether a square is occupied by player (```OccupiedLayer()```), a standing player (```UpLayer()```), or a player with the __Block__ skill (```SkillLayer(Skill.BLOCK)```). Other layers contain normalized values such as ```OwnTackleZoneLayer()``` that represents the number of friendly tackle zones squares are covered by divided by 8, or ```MALayer()``` where the values are equal to the movement allowence of players divided by 10.

botbowl environments have the above 45 layers by defaults. Custom layers can, however, be implemented by implementing the ```FeatureLayer```:

```python
from botbowl.ai import FeatureLayer
class MyCustomLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        for y in range(len(game.state.pitch.board)):
            for x in range(len(game.state.pitch.board[0])):
                player = game.state.pitch.board[y][x]
                out[y][x] = 1.0 if player is not None and player.role.cost > 80000 else 0.0
        return out

    def name(self):
        return "expensive players"
```

Layers can then be added to an environment like this this:

```python
env.layers.append(MyCustomLayer())
```

To visualize the feature layers, use the ```feature_layers``` option when calling ```render()```:

```python
env.render(feature_layers=True)
```

![botbowl Gym Feature Layers](img/gym_layers.png?raw=true "botbowl Gym Feature Layers")

### Reward 
`reward` is by default always zero. But we will later down discuss gym wrappers that help us define a reward function.

### Non spatial observation
`info['non_spatial_obs']` provides us with non-spatial observables in a numpy array with `shape=(116,)` and is divided into three parts. The first part is the state
and contains normailized values for folliwng 50 features:

0. 'half'
1. 'round'
2. 'is sweltering heat'
3. 'is very sunny'
4. 'is nice'
5. 'is pouring rain'
6. 'is blizzard'
7. 'is own turn'
8. 'is kicking first half'
9. 'is kicking this drive'
10. 'own reserves'
11. 'own kods'
12. 'own casualites'
13. 'opp reserves'
14. 'opp kods'
15. 'opp casualties'
16. 'own score'
17. 'own turns'
18. 'own starting rerolls'
19. 'own rerolls left'
20. 'own ass coaches'
21. 'own cheerleaders'
22. 'own bribes'
23. 'own babes'
24. 'own apothecary available'
25. 'own reroll available'
26. 'own fame'
27. 'opp score'
28. 'opp turns'
29. 'opp starting rerolls'
30. 'opp rerolls left'
31. 'opp ass coaches'
32. 'opp cheerleaders'
33. 'opp bribes'
34. 'opp babes'
35. 'opp apothecary available'
36. 'opp reroll available'
37. 'opp fame'
38. 'is blitz available'
39. 'is pass available'
40. 'is handoff available'
41. 'is foul available'
42. 'is blitz'
43. 'is quick snap'
44. 'is move action'
45. 'is block action'
46. 'is blitz action'
47. 'is pass action'
48. 'is handoff action'
49. 'is foul action'

Some values are boolean, either 0 or 1, while others are normalized.

The second part of the non-spatial observation is the 19 procedures represented in the one-hot vector:

0. StartGame,
1. CoinTossFlip,
2. CoinTossKickReceive,
3. Setup,
4. PlaceBall,
5. HighKick,
6. Touchback,
7. Turn,
8. MoveAction,
9. BlockAction,
10. BlitzAction,
11. PassAction,
12. HandoffAction,
13. FoulAction,
14. ThrowBombAction,
15. Block,
16. Push,
17. FollowUp,
18. Apothecary,
19. PassAttempt,
20. Interception,
21. Reroll,
22. Ejection

The final and third part of the non-spatial observation is the action types. Actions consist of 43 action types. Some action types, denoted by `<position>` also requires a position and `<player>` requires a player to be specified. If an action is available in the game state the corresponding value is 1.0 else 0.0. 

0. ```ActionType.START_GAME```,
1. ```ActionType.HEADS```,
2. ```ActionType.TAILS```,
3. ```ActionType.KICK```,
4. ```ActionType.RECEIVE```,
5. ```ActionType.END_PLAYER_TURN```,
6. ```ActionType.USE_REROLL```,
7. ```ActionType.DONT_USE_REROLL```,
8. ```ActionType.END_TURN```,
9. ```ActionType.END_SETUP```,
10. ```ActionType.STAND_UP```,
11. ```ActionType.SELECT_ATTACKER_DOWN```,
12. ```ActionType.SELECT_BOTH_DOWN```,
13. ```ActionType.SELECT_PUSH```,
14. ```ActionType.SELECT_DEFENDER_STUMBLES```,
15. ```ActionType.SELECT_DEFENDER_DOWN```,
16. ```ActionType.SELECT_NONE```,
17. ```ActionType.PLACE_PLAYER<Player, Position>``` (player to place and position of the target),,
18. ```ActionType.PLACE_BALL<Position>``` (position of the player),
19. ```ActionType.PUSH<Position>``` (position of the target),
20. ```ActionType.FOLLOW_UP```,
21. ```ActionType.SELECT_PLAYER<Position>``` (position of the player),
22. ```ActionType.MOVE<Position>``` (position of the target),
23. ```ActionType.BLOCK<Position>``` (position of the target),
24. ```ActionType.PASS<Position>``` (position of the target),
25. ```ActionType.FOUL<Position>``` (position of the target),
26. ```ActionType.HANDOFF<Position>``` (position of the target),
27. ```ActionType.LEAP<Position>``` (position of the target),
28. ```ActionType.STAB<Position>``` (position of the target),
29. ```ActionType.START_MOVE<Position>``` (position of the player),
30. ```ActionType.START_BLOCK<Position>``` (position of the player),
31. ```ActionType.START_BLITZ<Position>``` (position of the player),
32. ```ActionType.START_PASS<Position>``` (position of the player),
33. ```ActionType.START_FOUL<Position>``` (position of the player),
34. ```ActionType.START_HANDOFF<Position>```,
35. ```ActionType.USE_SKILL```,
36. ```ActionType.DONT_USE_SKILL```,
37. ```ActionType.SETUP_FORMATION_WEDGE```,
38. ```ActionType.SETUP_FORMATION_LINE```,
39. ```ActionType.SETUP_FORMATION_SPREAD```,
40. ```ActionType.SETUP_FORMATION_ZONE```,
41. ```ActionType.USE_BRIBE```,
42. ```ActionType.DONT_USE_BRIBE```

## Actions and action mask
The action space is discrete (`gym.spaces.Discrete(num_actions)`) where the number of actions is the number of non-positional actions plus number of positional actions times number of squares on the board. 

There is an action mask in the info object `info['action_mask]` that is a numpy array with `shape=(num_actions)`. With entries at 1.0 for allowed actions and zero for disallowed actions. 

## Environments
botbowl comes with five environments with various difficulty:

- **botbowl-v2:** 11 players on a 26x15 pitch (traditional size)
- **botbowl-7-v2:** 7 players on a 20x11 pitch
- **botbowl-5-v2:** 5 players on a 16x8 pitch
- **botbowl-3-v2:** 3 players on a 12x5 pitch
- **botbowl-1-v2:** 1 player on a 4x3 pitch

![A rendering of __botbowl-3-v2__.](img/gym_3.png?raw=true "A rendering of __botbowl-3-v2__.")

## Explore the Observation Space
Try running [examples/gym_example.py](examples/gym_example.py) while debugging in your favorite IDE (e.g. [PyCharm](https://www.jetbrains.com/pycharm/)). Set a break point in the line where the step function is called and investigate the obs object. If you run with the rendering enabled it is easier to analyze the values in the feature layers.

In the next tutorial, we will start developing a reinforcement learning agent.

## Wrappers 
By wrapping the environment in different wrappers we can change the behavior of the environement without modifying the internal parts of the environment. 
Wrappers are applied like below, and then used as normal.  

```python
env = BotBowlEnv()
env = NuffleWrapper(env, argument1=1337)
```

There are currently three wrappers, we will talk about two here: RewardWrapper and ScriptedActionWrapper
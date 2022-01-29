# Reinforcement Learning I: OpenAI Gym Environment
This tutorial will introduce you to botbowl's implementations of the [Open AI Gym interface](https://arxiv.org/pdf/1606.01540.pdf) 
that will allow for easy integration of reinforcement learning algorithms. 

You can run [examples/gym_example.py](examples/gym_example.py) to see a random agent play Blood Bowl through the botbowl 
Gym environment. The rendering is simplified for faster execution and looks like this:
![botbowl Gym GUI](https://njustesen.github.io/botbowl/img/gym.png?raw=true "botbowl Gym GUI botbowl-3")

[examples/multi_gym_example.py](examples/multi_gym_example.py) demonstrates how you can run multiple instance of the 
environment in parallel. Notice, that the render() function doesn't work across multiple processes. Instead a custom 
renderer is used in this example.

Agents receive numerical observations from the botbowl environment at every step and sends back and action with an action 
type and in some cases a position. Along with the observations, the environment also sends a scalar reward value to the 
agent. We will describe the structure of the three components: observations, actions, and rewards.

## botbowl gym environment 
Here is a short code example to demonstrate the core behavior. `BotBowlEnv()` creates a blood bowl game between two human teams. The home team is controlled by you by invoking the 
step()-function, the away team is controlled by RandomBot which takes random actions. We'll later see how we can 
modify this by providing arguments and using wrappers.  
```python
from botbowl.ai.env import BotBowlEnv, EnvConf
env_conf = EnvConf(size=11)
env = BotBowlEnv(env_conf) 
env.reset() 
spatial_obs, non_spatial_obs, action_mask = env.get_state()  
action_idx = 0 
assert action_mask[action_idx] == True  # check that the action we choose is allowed.   
spatial_obs, reward, done, info = env.step(action_idx) 
non_spatial_obs = info['non_spatial_obs'] 
action_mask = info['action_mask'] 
```

Now let's look at the types. In the next sections we'll dive deeper into these objects. 

* **spatial_obs** in a numpy array with shape `shape=(num_layer, height, width)` which in this case will be `(44, 17, 28)`
* **non_spatial_obs** is a numpy array with `shape=(116,)`
* **action_mask** is a numpy array with `dtype=bool` and `shape=(8117,)`
* **action_idx** is an int.
* **reward** is a float.
* **done** is a bool.
* **info** is a dict with two items

Try running [examples/gym_example.py](examples/gym_example.py) while debugging in your favorite IDE 
(e.g. [PyCharm](https://www.jetbrains.com/pycharm/)). Set a break point in the line where the step function is called 
and investigate the obs object. If you run with the rendering enabled it is easier to analyze the values in the feature 
layers.

### Action space
The action space is discrete, the action is an int in the range `0 <= action_idx < len(action_mask)`. The environment 
will translate `action_idx` into an action type. There are two kinds of action types: **simple action types** and 
**positional action types**. The positional action types needs a position too, but don't worry the environment takes care of this.

![Single-branch action space](img/action-space.png?raw=true "Single-branch action space")

Default simple action types:

 0. START_GAME,
 1. HEADS,
 2. TAILS,
 3. KICK,
 4. RECEIVE,
 5. END_SETUP,
 6. END_PLAYER_TURN,
 7. USE_REROLL,
 8. DONT_USE_REROLL,
 9. USE_SKILL,
 10. DONT_USE_SKILL,
 11. END_TURN,
 12. STAND_UP,
 13. SELECT_ATTACKER_DOWN,
 14. SELECT_BOTH_DOWN,
 15. SELECT_PUSH,
 16. SELECT_DEFENDER_STUMBLES,
 17. SELECT_DEFENDER_DOWN,
 18. SELECT_NONE,
 19. USE_BRIBE,
 20. DONT_USE_BRIBE,

Default positional action types: 

1. PLACE_BALL,
2. PUSH,
3. FOLLOW_UP,
4. MOVE,
5. BLOCK,
6. PASS,
7. FOUL,
8. HANDOFF,
9. LEAP,
10. STAB,
11. SELECT_PLAYER,
12. START_MOVE,
13. START_BLOCK,
14. START_BLITZ,
15. START_PASS,
16. START_FOUL, 
17. START_HANDOFF

By having an unrolled action space it becomes easy to use state-of-the-art algorithms, but it's worth considering that 
compared to many of the standard reinforcement learning benchmarks we have orders of magnitude larger action space. 

Unrolling is not necessarily the only way. Action spaces could also be split into __multiple branches__, such that one action is 
sampled from each branch. In botbowl, we would sample an action type, and then for some action types also a position. 
This can be achieved using three branches; 1) action type, 2) x-coordinate, and 3) y-coordinate. This approach was e.g. 
applied in preliminary results presented in [StarCraft II: A New Challenge for Reinforcement Learning](https://arxiv.org/pdf/1708.04782.pdf) 
since StarCraft has a similar action space.


### Spatial observation
`spatial_obs` is all the features layers stack together with `shape=(num_feature_layers, height, width)`. 

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

In addition, there's a `AvailablePositionLayer()` for each positional action type.  

A layer is a 2-D array of scalars in [0,1] with the size of the board including __crowd__ padding. Some layers have 
binary values, e.g. indicating whether a square is occupied by player (```OccupiedLayer()```), a standing player 
(```UpLayer()```), or a player with the __Block__ skill (```SkillLayer(Skill.BLOCK)```). Other layers contain normalized 
values such as ```OwnTackleZoneLayer()``` that represents the number of friendly tackle zones squares are covered by 
divided by 8, or ```MALayer()``` where the values are equal to the movement allowence of players divided by 10.

botbowl environments have the above 44 layers by defaults. Custom layers can, however, be implemented by implementing 
the ```FeatureLayer```:

```python
from botbowl.ai.layers import FeatureLayer
from botbowl.ai.env import BotBowlEnv, EnvConf
import numpy as np 

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

# Layers can then be added to an environment 

env_conf = EnvConf(size=11, extra_feature_layers=[MyCustomLayer])
env = BotBowlEnv(env_conf)
```

To visualize the feature layers, use the ```feature_layers``` option when calling ```render()```:

```python
env.render(feature_layers=True)
```

![botbowl Gym Feature Layers](img/gym_layers.png?raw=true "botbowl Gym Feature Layers")

### Non spatial observation
`info['non_spatial_obs']` provides us with non-spatial observables in a numpy array with `shape=(116,)` and is divided 
into three parts. The first part is the state and contains normailized values for folliwng 50 features:

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

The second part of the non-spatial observation is the 23 procedures, if the procedure is on the game state's stack, the 
is generates 1.0 else 0. 

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

The final and third part of the non-spatial observation is the action types, listed and discussed above. Here we don't make a 
difference between simple and positional action types. If an action type is available, it gets value 1.0 else 0. 

### Reward 
`reward` is by default always zero. We will later discuss how gym wrappers can help us define a reward function.

## Environment configuration
 `BotBowlWrapper` can be used to add additional behavior 
to the environment. 

### EnvConf 
The `EnvConf` class can be used to modify the default list above. There are three options: 
1. use the argument when creating the `EnvConf`-object.  
2. modify the default lists directly in [botbowl/ai/env.py](botbowl/ai/env.py). 
3. modify the `EnvConf`-object after creation but before creating the `BotBowlEnv`. 

#### Pitch size 
botbowl comes with five environments with various difficulty, use `EnvConf(size=3)` to get a 3 player pitch:

- **botbowl-v4:** 11 players on a 26x15 pitch (traditional size)
- **botbowl-7-v4:** 7 players on a 20x11 pitch
- **botbowl-5-v4:** 5 players on a 16x8 pitch
- **botbowl-3-v4:** 3 players on a 12x5 pitch
- **botbowl-1-v4:** 1 player on a 4x3 pitch

![A rendering of __botbowl-3-v2__.](img/gym_3.png?raw=true "A rendering of __botbowl-3-v4__.")

#### Custom formations 
This environment only supports fixed formations in its action space. By default there are only a few formations. 
You can easily add custom formations: 
```python 
EnvConf(extra_formations=[MyCustomFormation1, MyCustomFormation2])
``` 
follow [**Scripted bot III - Formation**](bots-iii.md) on how to create formations. 

### Wrappers 
By wrapping the environment in different wrappers we can change the behavior of the environement without modifying its 
internals code. Here's the code for a wrapper that can add scripted behavior inside the env, it's located in 
[botbowl/ai/env.py](botbowl/ai/env.py). 

```python
class ScriptedActionWrapper(BotBowlWrapper):
    def __init__(self, env, scripted_func: Callable[[Game], Optional[Action]]):
        super().__init__(env)
        self.scripted_func = scripted_func

    def step(self, action: int, skip_observation: bool = False):
        self.env.step(action, skip_observation=True)
        self.do_scripted_actions()
        return self.root_env.get_step_return(skip_observation)

    def reset(self):
        self.env.reset()
        self.do_scripted_actions()
        obs, _, _ = self.root_env.get_state()
        return obs

    def do_scripted_actions(self):
        game = self.game
        while not game.state.game_over and len(game.state.stack.items) > 0:
            action = self.scripted_func(game)
            if action is None:
                break
            game.step(action)
```
And here is how it can be used. We define an oversimplified function to choose block dice. 

```python
from botbowl import Game, Action, ActionType, BotBowlEnv, ScriptedActionWrapper
import botbowl.core.procedure as procedure 
from typing import Optional 

def my_scripted_actions(game: Game) -> Optional[Action]: 
    proc = game.get_procedure()
    if type(proc) is procedure.Block or (type(proc) is procedure.Reroll and type(proc.context) is procedure.Block): 
        for action_choice in game.get_available_actions(): 
            if action_choice.action_type is ActionType.SELECT_DEFENDER_DOWN: 
                return Action(ActionType.SELECT_DEFENDER_DOWN) 
            if action_choice.action_type is ActionType.SELECT_DEFENDER_STUMBLES: 
                return Action(ActionType.SELECT_DEFENDER_STUMBLES) 
    return None 

env = BotBowlEnv()
env = ScriptedActionWrapper(env, my_scripted_actions)
```
We will talk more about wrappers in the next tutorial where we will start developing a reinforcement learning agent. 

Next tutorial: [**Reinforcement Learning II: A2C**](a2c.md) 

# Reinforcement Learning IV: Self-play

In the previous tutorials we trained an agent (with some success) to beat the random baseline. Training against such an 
agent will, however, will most likey result in an overfitted strategy that only works against this very simplistic opponent. 
In the previous tutorial, the agent learned to charge forward with the ball, wait util the next turn while standing near several 
opponent players, and then move forward again. A human player, or a smarter bot, would of course punish this behavior by 
surrounding and blocking the ball carrier to force a fumble. 

Open AI has recently shared some very impressive work on competitive self-play in RL. Self-play was also used in their Dota 2 bot
that competed against professional players. If you haven't read their [blog post](https://openai.com/blog/competitive-self-play/) 
and [paper](https://arxiv.org/abs/1710.03748), I strongly recommend you to check it out.
Similarly, [AlphaStar](https://deepmind.com/blog/article/alphastar-mastering-real-time-strategy-game-starcraft-ii) (by DeepMind) used a variation of self-play called AlphaLeague where several diverse agents competed against each other during training. 
Based on these findings, and our own shortcomings, it seems likely that we need some sort of self-play to master Blood Bowl with RL.
This tutorial will demonstrate how we can do that in the small 1v1 variant of the game.

## Approach

First, we specify the settings of our self-play session:

```python
make_agent_from_model = partial(A2CAgent, env_conf=env_conf, scripted_func=a2c_scripted_actions)

selfplay = True  # Use this to enable/disable self-play
selfplay_window = 1
selfplay_save_steps = int(num_steps / 10)
selfplay_swap_steps = selfplay_save_steps


``` 

```selfplay=False``` by default, so remember to set it to ```True```.
```selfplay_window``` controls the size of the sampling window when we swap the opponent policy. A higher window size allows for more 
diversity in the policy pool.  
```selfplay_save_steps``` specifies how often the current policy is added to the pool.
```selfplay_swap_steps``` specifies how often the opponent is swapped by a uniformly sampled policy from the pool.

Before training we save our initial policy and use it as the opponent:

```python
# self-play
selfplay_next_save = selfplay_save_steps
selfplay_next_swap = selfplay_swap_steps
selfplay_models = 0
if selfplay:
    model_name = f"{exp_id}_selfplay_0.nn"
    model_path = os.path.join(model_dir, model_name)
    torch.save(ac_agent, model_path)
    self_play_agent = make_agent_from_model(name=model_name, filename=model_path)
    envs.swap(self_play_agent)
    selfplay_models += 1
```

To swap the opponent policy, our worker process needs to handle the swap commands:

```python
def worker(remote, parent_remote, env, worker_id):
    ...
    next_opp = botbowl.make_bot('random')
    while True:
        command, data = remote.recv()
        if command == 'step':
            ...    
            if done or steps >= reset_steps:
                ...
                env.root_env.away_agent = next_opp
                ...
            remote.send((obs, reward, reward_shaped, tds_scored, tds_opp_scored, done, info))
        ...
        elif command == 'swap':
            next_opp = data
```

In this way, we do not swap the opponent directly but only at the beginning of a new episode.

After each step, we check if it's time to save the current policy into our pool of policies:

```python
...
# Self-play save
if selfplay and all_steps >= selfplay_next_save:
    selfplay_next_save = max(all_steps+1, selfplay_next_save+selfplay_save_steps)
    model_path = f"models/{model_name}_selfplay_{selfplay_models}"
    print(f"Saving {model_path}")
    torch.save(ac_agent, model_path)
    selfplay_models += 1
```

We also check if it is time to swap the model. Note, that saving and swapping can happen at different rates. 

```python
# Self-play swap
if selfplay and all_steps >= selfplay_next_swap:
    selfplay_next_swap = max(all_steps + 1, selfplay_next_swap+selfplay_swap_steps)
    lower = max(0, selfplay_models-1-(selfplay_window-1))
    i = random.randint(lower, selfplay_models-1)
    model_name = f"{exp_id}_selfplay_{i}.nn"
    model_path = os.path.join(model_dir, model_name)
    print(f"Swapping opponent to {model_path}")
    envs.swap(make_agent_from_model(name=model_name, filename=model_path))
```

## Results

We try running [examples/a2c/a2c_example.py](https://github.com/njustesen/botbowl/blob/master/examples/a2c/a2c_example.py) on the 1v1 variant ```botbowl-1-v2``` with the default self-play from above.
Additionally, we set ```num_steps=5000000```.

The results are quite interesting. We see that the agent's touchdown rate (blue line) gets beyond that when it trained against a random agent (~12 vs ~7.5). 
This is most likely because we now have two scoring agents; the best strategy is simply to attempt score rather than trying to protect the ball (there are no teammates to hide behind), and since both agents continually score, 
the ball possession quickly switches side making it easier to score a lot of touchdowns. This is something we also see in Blood Bowl 
when agile teams, such as Skaven, play against each other, where the score typically reach something like 4-4 (which is a lot on the full board).

![Self-play on botbowl-1-v2](img/botbowl-1-v2_selfplay.png?raw=true "Self-play on botbowl-1-v2")

The Touchdown plot also shows the opponent's touchdowns in red. It is noticeable that the red and blue lines doesn't meet. Since the agents 
should see the board in the exact same way, because we simply mirror it, they should be equally good if the agent reaches the optimal strategy. 
This might indicate that there are not just one optimal strategy but several, and whenever we swap the opponent, the learning agent quickly 
learns to adjust it's playing style. Another option is that there is a small bug somewhere but so far I haven't noticed any anomalies in the stored 
policies.

When we test the final policy against the random baseline, we see that it is still good but a bit worse than when we trained it directly against 
random. In 100 test games, the selfplay agent achieved a 5.72 TD rate against random vs. ~7.5 by our vanilla A2C agent.

## Things to try
By only playing against the recently saved policy, there is a risk of continuously switching between two or more overfitted strategies rather than learning a general strategy that 
works well against many strategies. In games where there is a multitude of different strategies, it's often best to maintain a pool of policies that reflects the diversity of the strategy space. 
Try running the experiment with a larger window size to allow sampling of policies other than the most recent. It is possible, that there 
aren't that many different strategies in Blood Bowl (how many can you think of?) but how about 3v3 or 11v11 Blood Bowl? 

Would it make more sense to deploy a different opponent models on each worker process, such that our policy is updated basd on batches containing 
a variety of opponent playing styles?

After that, how would you actually go about applying this to the full board? You could perhaps seed the self-play phase with the agent from the 
previous [tutorial](a2c-full.md)? Maybe you have an even better ideas.

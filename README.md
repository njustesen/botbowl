# Gotebot
I wish to thank Niels Justesen for providing not only the FFAI framework but also a complete implementation of Advantage Actor Critic (A2C) for FFAI on top of which Gotebot was built and trained. 
## Introduction 
In this post I will lay out the details of Gotebot and some of my thoughts for the future of ML bots for Blood Bowl. When describing Gotebot, I will mostly compare to Niels Justesen’s Reinforcement learning tutorial. The main difference to his tutorial is a curriculum learning which is described in depth below. Some changes were also made in how the reward and discounted reward were calculated. Finally, the network structure was also modified, see appendix C. 
## Reward shape 
Rewards were calculated based on game outcomes reported by the environment. The weights are detailed in Appendix A. Furthermore, three non-outcome based rewards were calculated: 

Ball progression reward 0.005 per square, same as in the tutorial. 

Dominance around the ball. In the area two squares from the ball (i.e. 5-by-5 squares with ball in the middle) reward: 
0.01 per tacklezone. 
0.05 per player up to 3 players 
0.05 per player marking the ball / ball carrier. 
	Reward is subtracted with same coefficients for opponent’s players and tackle zones. 

Move in the right direction. 
If the bot is carrying the ball. Each player’s step towards opponent endzone is rewarded with 0.001
Else each step toward the ball is rewarded with 0.001 

In both cases, moves in the opposite direction results in the same size negative reward. 

Finally, the discounted reward was not calculated until the end of the game as in the tutorial. Instead it was ended at a touchdown or end of the half. This means  that an episode ends at a touchdown or at the end of the half. This is further discussed in the final section. 

## Progressive curriculum learning
During training a fixed number of processes were dedicated to playing several different lectures. The purpose of a lecture is to teach the policy about a specific reward that the bot is otherwise unlikely to randomly stumble upon, for example scoring, knocking down the opponent ball carrier, picking up the ball. A summary of all lectures can be found in Appendix B.

All lectures were designed to have progressive difficulty. At the first level only one or two correct actions are needed to reach the state where the environment gives the intended reward. This design ensures that a random policy can solve the first level of any lecture with probability 1-10%. As the policy learns and the success rate improves the difficulty is increased and more correct actions are needed to reach the intended end state. 

After each step in the environment the lecture checks if the end condition of the lecture is satisfied. If so, the lecture then evaluates if the agent has passed. For example, the Scoring-lecture checked if the policy scored a touchdown. Finally the environment is reset to the next lecture in the queue. 

A lecture is not able to give the policy a reward, instead the lecture needs to be tailored to a reward specified in the environment. For example, if the purpose of a lecture is to teach the policy to mark the ball carrier, then the environment needs to reward that, otherwise the policy can’t learn it. 

Finally, to ensure that the policy generalises, the lectures were created with as much randomness as possible. The players not involved in the lecture scenario were randomly placed all over the board except for the lecture area. 

To avoid forgetting learned features all lectures have a non-zero probability of being picked. 
## Training the policy
Training was divided into three phases. 
1) Pre training with lectures
2) Training against almost-random bot 
3) Self play 

The AlmostRandomBot takes random actions except it always places the ball in the center at kick off to avoid that the policy builds a strategy reliant on touchbacks.

Pre training with lectures used six of eight processes exclusively for lecture training. The other two processes played normal games against the almost random bot. This training ended when the policy reached the highest level of all lectures. 

The second phase of training used two processes for lectures and six for normal games against the AlmostRandomBot. I ended the phase when the policy seemed to converge at three touchdowns per game and ~100% win rate . See Figure 1. 


Figure 1 - results from the second phase of training shows a rapid increase in performance against the AlmostRandomBot. 

This final training phase was self play. Again two processes were used for lectures and six for normal self play games. The self play agent was switched once at approximately 1.5 million step.. I decided to end this phase at 3 million training steps when I didn’t see significant progress. See Figure 2. 


Figure 2 - Self play results from the final training phase. Very noisy and any progress is difficult to see. 

## Discussion on training
During self play the policy forgot how to reliably beat the RandomBot. I think the policy became too dependent on its opponent bringing the ball forward in a risky manner. I think it’s important to come up with ways to prevent forgetting important features. It could be to play a big fraction of the games against the opponent which the policy is training against and a small fraction of games against old opponents against which the policy can already reliably beat. If the policy starts drawing or losing games against the old opponents, this fraction can be increased. 

Towards the end of training I noticed that the policy was still doing an unnecessary amount of GFIs. I think it’s very difficult for the policy to learn how they work and when to use them since they are only punished in 17% of the cases. A remedy to this stochastic reward problem is discussed in the next section. 
## Conclusion and future
## Curriculum 
I think Gotebot proved that progressive curriculum learning is an efficient way to learn simple tactics and that combined with normal games a neural network can be trained to weave the learned tactics into very simple strategies. Such as picking up the ball at kickoff, moving it forward along a flank while avoiding opponent tackle zones and finally scoring. However, handcrafting lectures with a progressive level system is time consuming and requires a large degree of domain knowledge. The effort grows as one wishes to consider more roasters and skills. Furthermore, claiming that a behavior that optimizes rewards in the curriculum training setting are also viable blood bowl strategies is questionable at best. Therefore it is probably best to only use lectures for learning during a pre-training phase and policy evaluation in the later training phases. . 
## Stochastic reward 
Gotebot did not show any sign of understanding risk management or ability to anticipate its opponent’s actions. Behavior such as screening or caging the ball carrier or even moving a player to assist a block. I found it very challenging to design lectures that could teach the policy to take actions that would increase the probability of important future actions, as when moving a player to assist a block. For example, without re-rolls and skills, the probability of a successful knockdown increases from 33% to 55%  when you go from 1d block to 2d block. Though a significant increase, the policy did not seem to learn to move in assists. I think this is another part of the stochastic reward problem. 

To address this I propose the following solution. At each action that can result in multiple outcomes the reward takes the probabilities of the different outcomes into account: 

R = p(outcome #1) Routcome #1 + p(outcome #2) Routcome #2+ ... 

One of the outcomes is chosen for continued play as usual. The reward of the other outcome(s) is calculated in a dream. Since the reward of an action is the accumulated reward of all future states the dream needs to take several steps. 

- Figure 3 - 

To decrease the computational load of this strategy, the actions that are branched can be limited to those that cause turnovers. There are certainly more trade offs to experiment with to optimize training performance. 

To further stabilize the reward signal one could also consider branching at the same type of actions in the dream too thus creating a dream within a dream (which is very appealing from a popular culture point of view!). I think at least one dream branching is feasible to account for a re-roll choice. 
Limiting episode length 
Finally I want to discuss what goals a policy could have during a game. Experienced coaches urge new coaches to consider the goal of a game and of a drive (and each turn but I’ll leave that for now). Until our bots start playing in a league format without full recovery between games, the goal of every game is to win at any price. Currently we are not giving the policy any reward for the win per se but touchdowns are significantly rewarded. What about the goal of the drive? There are viable Blood Bowl strategies where a coach forces their opponent to score early in the half to be able to score themselves in turn eight. Setting up a reward scheme to award this behavior is very detailed and usually not appropriate. Instead I propose to limit the policy’s goal in the drive to scoring and preventing the opponent from scoring. This is done by ending the episode at a touchdown or at the end of the half if no team scored. This reduces the noise of the discounted reward signal by becoming independent of whatever happens on the line of scrimmage blocks in the next drive. The game can still continue but the training episode ends. 

The disadvantage of this is that it inhibits the policy from learning multi drive strategies such as stalling, two-one grind and bashing the opponent into submission. However, looking at the performance of the current RL policies, it’s very far away..!  

Gotebot was trained with this setting and has a long way to go before the lack of multi drive strategies is the main reason it is not winning. I think this should be the standard setting until the ML bots have proved to give human coaches a proper challenge! 

Appendixes removed until further notice. 

# Scripted Bots II: Pathfinding and Probabilities

In this tutorial, we will walk you through the code of a fully-fledged bot that is able to consistently beat the random bot
from the previous tutorial. To achieve this, we will make heavy use of the pathfinding and probability features of FFAI.
There will be plenty things to improve on after this tutorial while some intelligent behaviors will emerge.

## Architecture
We will use the procedure-based bot that was introduced in the previous tutorial. This will allow us to implement employ simple
rules for most of the procedures so we can focus on the few ones that require important planning and decision-making. The
bot that we will describe follows a prioritized lists of decisions, starting with simple and safe player actions in the beginning and
ending with more risky and complicated decisions. The list will be traversed before each player takes a decision, such that a risky
action taken by one player can open up several simple actions by others. This approach stand in contrast to the design of Grodbot, the
winnner of Bot Bowl I, wherein many possible actions a evaluated fore every player at the same time and then the most promising action
is taken. Grodbots approach is possibly more potent while it is tricky to evaluate all types of actions. Here, a simple list allows us
to implement easily understood heuristics.

The list of priorities are:
1. Stand up and end turn with players that are marked and down
2. Move the ball carrier safely towards the endzone
3. Execute safe blocks
4. Pickup the ball if it's on the ground
5. Move receivers into scoring range
6. Make a blitz move
7. Make cage around ball carrier
8. Move non-marked players to assist
9. Move towards the ball
10. Make risky blocks
11. End turn

The bot plans out a complete player turn at each step during its turn. A list of actions is maintained and execute in order.
When a player ends its turn, a function is called that will plan a new list of actions for a new player until no reasonable player
actions are lef.

The rest of this tutorial will walk you through these eleven decisions.

## 1. Stand up

We simply iterate through all players on our team and if we find a player that is on the field (player.position is not None),
is not standing up, not stunned, and not used, then we plan a move action, stand the player up. We return to stop the planning
execute the planned actions.

```python
for player in self.my_team.players:
    if player.position is not None and not player.state.up and not player.state.stunned and not player.state.used:
        if game.num_tackle_zones_in(player) > 0:
            self.actions.append(Action(ActionType.START_MOVE, player=player))
            self.actions.append(Action(ActionType.STAND_UP))
            return
```

## 2. Move the ball carrier

This is perhaps the most important step, as we wouldn't be able to win against any opponent without it. If we have the
ball, we should move towards the endzone, especially I we can score. However, we shouldn't do any risky moves that will
cause us to loose the ball. Moving the ball carrier is split up in three parts:

```python
if ball_carrier is not None and ball_carrier.team == self.my_team and not ball_carrier.state.used:
    # 2.1 Can ball carrier score with high probability
    ...

    # 2.2 Hand-off action to a scoring player
    ...

    # 2.3 Move safely towards the endzone
    ...
```

Notice, that for simplicity the ball carrier can perform a hand-off action but not a pass action. We will start by
describing the steps needed to move the playing into the endzone if it's safe to do so.

```python
# 2.1 Can ball carrier score with high probability
td_path = pf.get_safest_scoring_path(game, ball_carrier)
if td_path is not None and td_path.prob >= 0.7:
    self.actions.append(Action(ActionType.START_MOVE, player=ball_carrier))
    for step in td_path.steps:
        self.actions.append(Action(ActionType.MOVE, position=step))
    return
```

First, we use the pathfinding module (ffai.ai.pathfinding) to get the safest (and then shortest) path for the ball carrier
the opponent's endzone. If ```pf.get_safest_scoring_path(game, ball_carrier)``` returns None, then no path was found, which
means that either the path is blocked or the player doesn't have enough moves left. If a path was found, the Path instance
will have a probability in [0,1] and a list of steps in the path. Here, we decide to move the ball carrier to the endzone,
if the probability of success is at least 70%. This is absolutely not a perfect heuristic, especially because we haven't
yet checked if we can make any blitzes to clear the path for a safer option. Nevertheless, we will settle for this simple
solution.

If we can't score with the ball carrier, or it is too risky, we will check to see if we can hand off the ball to another
player in scoring range.

```python
# 2.2 Hand-off action to a scoring player
if game.is_handoff_available():

    # Get players in scoring range
    unused_teammates = []
    for player in self.my_team.players:
        if player.position is not None and player != ball_carrier and not player.state.used and player.state.up:
            unused_teammates.append(player)

    # Find other players in scoring range
    handoff_p = None
    handoff_path = None
    handoff_player = None
    for player in unused_teammates:
        if game.get_distance_to_endzone(player) > player.get_ma() + 3:
            continue
        td_path = pf.get_safest_scoring_path(game, player)
        if td_path is None:
            continue
        path_from_ball_carrier = pf.get_safest_path_to_player(game, ball_carrier, player)
        if path_from_ball_carrier is None:
            continue
        p_catch = game.get_catch_prob(player, handoff=True, allow_catch_reroll=True)
        p = td_path.prob * path_from_ball_carrier.prob * p_catch
        if handoff_p is None or p > handoff_p:
            handoff_p = p
            handoff_path = path_from_ball_carrier
            handoff_player = player

    # Hand-off if high probability or last turn
    if handoff_path is not None and (handoff_p >= 0.7 or self.my_team.state.turn == 8):
        self.actions = [Action(ActionType.START_HANDOFF, player=ball_carrier)]
        for step in handoff_path.steps:
            self.actions.append(Action(ActionType.MOVE, position=step.x))
        self.actions.append(Action(ActionType.HANDOFF, position=handoff_player.position))
        return
```

First, we check if a hand-off actions is available. If not, a we scan the field for unused players on our team that are
standing. We check if each player is in scoring distance and saves the safest of the available choice. The probability of
success is computed as a product of three components: 1) the probability of success for moving the ball carrier to the target player,
2) the probability for catching the ball, and 3) the probability of success for moving the target player to the endzone.
If the hand-off is successful, step 2.1 from before will make sure that we the target player will attempt to score.
One catch with this approach is, that the target player's path to endzone is computed in a state where that current ball carrier is
not necessarily adjacent, i.e. before the ball carriers is moved. This could mean that the ball carrier will block the scoring
path of the target player. This an effect of not doing perfect forward planning but it should be okay in most situations.

The last step for the ball carrier, if none of the above was possible or safe enough, is to just move the ball carrier
safely towards the endzone. We do this by calling ```pf.get_safest_scoring_path(game, ball_carrier, max_search_distance=30)```
with a max_searc_distance of 30, which will override the player's movement allowance. When a path is found, we follow it
until a step will cause it to become adjacent to an opponent or a GFI roll is needed.

```python
if game.num_tackle_zones_in(ball_carrier) == 0:
    td_path = pf.get_safest_scoring_path(game, ball_carrier, max_search_distance=30)
    if td_path is not None:
        steps = []
        for step in td_path.steps:
            if game.num_tackle_zones_at(ball_carrier, step) > 0:
                break
            if len(steps) >= ball_carrier.num_moves_left():
                break
            steps.append(step)
        if len(steps) > 0:
            self.actions.append(Action(ActionType.START_MOVE, player=ball_carrier))
            for step in steps:
                self.actions.append(Action(ActionType.MOVE, position=step))
            return
```

## 3. Safe blocks

If we can't score, we might just execute a few safe blocks. The code below will first find the safest block in terms of not
knocking down the attacker. Additionally, it will not select a block where there is a chance of loosing the ball (if we have
possession of the ball). If the safest block has more than 94% chance of staying upright and no chance of loosing the ball, then
we will perform the block immediately.

```python
attacker, defender, p_self_up, p_opp_down, block_p_fumble_self, block_p_fumble_opp = self._get_safest_block(game)
if attacker is not None and p_self_up > 0.94 and block_p_fumble_self == 0:
    self.actions.append(Action(ActionType.START_BLOCK, player=attacker))
    self.actions.append(Action(ActionType.BLOCK, position=defender.position))
    return
```

Without describing in detail how ```self._get_safest_block(game)``` works (it is pretty straigforward), is useful to know
that it makes use of FFAI's ```game.get_block_probs(attacker, defender)```, that returns a tuple contains four probabilites:
1) the probability for the attacker to be knocked down, 2) the probability for the defender to be knocked down, 3) the
probability of the attacker to fumble the ball, and 4) the probability for the defender to fumble the ball. This function
makes use of lower-level functions such as ```game.num_block_dice(attacker, defender)``` and ```game.get_push_squares(attacker, defender)```.
Notice, that the use of team-rerolls is not included in the computation of the probability.

## 4. Pickup the ball

We iterate all unused players on the field to see if they are in reach of the ball. We use FFAI's function
```game.get_pickup_prob(player, game.get_ball_position(), allow_pickup_reroll=True)``` to compute the probability for the player
to pick up the ball. The function includes the use of the Pickup skill but does not include team re-rolls by default.
We then multiply the pick-up probability with the probability of success for moving to the ball. If there is more than 33% chance
of success, we will perform the move. After the move, we will move towards the endzone until a dodge or GFI roll or we would move into
an opponent tackle zone.

```python
if game.get_ball_carrier() is None:
    pickup_p = None
    pickup_player = None
    pickup_path = None
    for player in self.my_team.players:
        if player.position is not None and not player.state.used:
            path = pf.get_safest_path(game, player, game.get_ball_position())
            if path is not None:
                p = game.get_pickup_prob(player, game.get_ball_position(), allow_pickup_reroll=True)
                p = path.prob * p
                if pickup_p is None or p > pickup_p:
                    pickup_p = p
                    pickup_player = player
                    pickup_path = path
    if pickup_player is not None and pickup_p > 0.33:
        self.actions.append(Action(ActionType.START_MOVE, player=pickup_player))
        if not pickup_player.state.up:
            self.actions.append(Action(ActionType.STAND_UP))
        for step in pickup_path.steps:
            self.actions.append(Action(ActionType.MOVE, position=step))
        # Find safest path towards endzone
        if game.num_tackle_zones_at(pickup_player, game.get_ball_position()) == 0:
            td_path = pf.get_safest_scoring_path(game, pickup_player, from_position=game.get_ball_position(), num_moves_used=len(pickup_path.steps), max_search_distance=30)
            if td_path is not None:
                steps = []
                for step in td_path.steps:
                    if game.num_tackle_zones_at(pickup_player, step) > 0:
                        break
                    if len(steps) + len(pickup_path.steps) >= pickup_player.get_ma():
                        break
                    steps.append(step)
                if len(steps) > 0:
                    self.actions.append(Action(ActionType.START_MOVE, player=ball_carrier))
                    for step in steps:
                        self.actions.append(Action(ActionType.MOVE, position=step))
        return
```

## 5. Move receivers into scoring range
As we allow the ball carrier to make hand-off actions to players in scoring range, it makes sense to move some players
close to the endzone. First, we scan the field for unused players on the field that are not in opponent tackle zones.
Then, if the have the catch skill and aren't the ball carrier, we will move one of them towards the endzone until they
are in scoring range, would move into an opponent tackle zone, or would have to make a GFI roll.

```python
open_players = []
for player in self.my_team.players:
    if player.position is not None and not player.state.used and game.num_tackle_zones_in(player) == 0:
        open_players.append(player)

for player in open_players:
    if player.has_skill(Skill.CATCH) and player != ball_carrier:
        if game.get_distance_to_endzone(player) > player.num_moves_left():
            continue
        td_path = pf.get_safest_scoring_path(game, player, max_search_distance=30)
        steps = []
        for step in td_path.steps:
            if len(steps) >= player.get_ma() + (3 if not player.state.up else 0):
                break
            if game.num_tackle_zones_at(player, step) > 0:
                break
            if step.distance(td_path.steps[-1]) < player.get_ma():
                break
            steps.append(step)
        if len(steps) > 0:
            self.actions.append(Action(ActionType.START_MOVE, player=player))
            if not player.state.up:
                self.actions.append(Action(ActionType.STAND_UP))
            for step in steps:
                self.actions.append(Action(ActionType.MOVE, position=step))
            return
```

## 6. Make a blitz move

Deciding where to make a blitz action requires a lot of planning. Each player can move to and block any opponent player
in range from several different angles. We will prioritize a good amount of time for this decision by checking many of
these options. We will, however, ignore potential blitzing players that do not have the Block skill. Each avaible blitz
move is rated using the following summation: ```score = p_self_up + p_opp + p_fumble_opp - p_fumble_self```, which is the
probability of staying upright plus the probability of knocking down the opponent plus the probability of causing the opponent
to fumble minus the probability of loosing the ball possession. We will execute the highest scoring blitz if it has a score
of 1.25 or higher, e.g. 0.92 + 0.44 + 0.0 - 0.08 = 1.28, which is a pretty good blitz to take.

```python
if game.is_blitz_available():
    best_blitz_attacker = None
    best_blitz_defender = None
    best_blitz_score = None
    best_blitz_path = None
    for blitzer in open_players:
        if blitzer.position is not None and not blitzer.state.used and blitzer.has_skill(Skill.BLOCK):
            for opp_player in self.opp_team.players:
                if opp_player.position is None or opp_player.position.distance(blitzer.position) > blitzer.num_moves_left():
                    continue
                if opp_player.state.up and not (opp_player.has_skill(Skill.BLOCK) and opp_player != ball_carrier):
                    for adjacent_position in game.get_adjacent_squares(opp_player.position, occupied=False):
                        path = pf.get_safest_path(game, blitzer, adjacent_position, num_moves_used=1)
                        if path is None:
                            continue
                        # Include an addition GFI to block if needed
                        moves = blitzer.get_ma() if blitzer.state.up or blitzer.has_skill(Skill.JUMP_UP) else blitzer.get_ma() + 3
                        if len(path.steps) > moves:
                            path.prob = path.prob * (5.0/6.0)
                        p_self, p_opp, p_fumble_self, p_fumble_opp = game.get_blitz_probs(blitzer, adjacent_position, opp_player)
                        p_self_up = path.prob * (1-p_self)
                        p_opp = path.prob * p_opp
                        p_fumble_opp = p_fumble_opp * path.prob
                        if blitzer == game.get_ball_carrier():
                            p_fumble_self = path.prob + (1 - path.prob) * p_fumble_self
                        score = p_self_up + p_opp + p_fumble_opp - p_fumble_self
                        if best_blitz_score is None or score > best_blitz_score:
                            best_blitz_attacker = blitzer
                            best_blitz_defender = opp_player
                            best_blitz_score = score
                            best_blitz_path = path
    if best_blitz_attacker is not None and best_blitz_score >= 1.25:
        self.actions.append(Action(ActionType.START_BLITZ, player=best_blitz_attacker))
        if not best_blitz_attacker.state.up:
            self.actions.append(Action(ActionType.STAND_UP))
        for step in best_blitz_path.steps:
            self.actions.append(Action(ActionType.MOVE, position=step))
        self.actions.append(Action(ActionType.BLOCK, position=best_blitz_defender.position))
        return
```

## 7. Make a cage around the ball carrier

If we have possession of the ball, we would like to protect the ball carrier and if the opponent has the ball, we would
like to surround the ball carrier in as similar way. Ideally, we would like to surround the ball carrier in different ways
depending on which team it's on and the position on the board. To make it simple, we will attempt to make a classic Blood
Bowl cage around the ball carrier regardless of these other factors. A cage is a formation where several players (ideally four)
are diagonally adjacent to the ball carrier. The way this is done should be all familiar with you know, besides an
additional check to see of a cage position is out of bounds.

```python
cage_positions = [
    Square(game.get_ball_position().x - 1, game.get_ball_position().y - 1),
    Square(game.get_ball_position().x + 1, game.get_ball_position().y - 1),
    Square(game.get_ball_position().x - 1, game.get_ball_position().y + 1),
    Square(game.get_ball_position().x + 1, game.get_ball_position().y + 1)
]
if ball_carrier is not None:
    for cage_position in cage_positions:
        if game.get_player_at(cage_position) is None and not game.is_out_of_bounds(cage_position):
            for player in open_players:
                if player == ball_carrier or player.position in cage_positions:
                    continue
                if player.position.distance(cage_position) > player.num_moves_left():
                    continue
                if game.num_tackle_zones_in(player) > 0:
                    continue
                path = pf.get_safest_path(game, player, cage_position, max_search_distance=20)
                if path is not None and path.prob > 0.94:
                    self.actions.append(Action(ActionType.START_MOVE, player=player))
                    if not player.state.up:
                        self.actions.append(Action(ActionType.STAND_UP))
                    for step in path.steps:
                        self.actions.append(Action(ActionType.MOVE, position=step))
                    return
```

# Move non-marked players to assist

If we can't move the ball nor make any safe blocks, we should move players to assisting positions. First, we scan
the field for possible assisting position by iterating the opponent players.

```python
assist_positions = []
for player in game.get_opp_team(self.my_team).players:
    if player.position is None or not player.state.up:
        continue
    opponents = game.get_adjacent_opponents(player, down=False)
    for opponent in opponents:
        att_str, def_str = game.get_block_strengths(player, opponent)
        if def_str == att_str:
            for open_position in game.get_adjacent_squares(opponent.position, occupied=False):
                if len(game.get_adjacent_players(open_position, team=self.opp_team, down=False)) == 1:
                    assist_positions.append(open_position)

```

We make use of the function ```game.get_block_strengths(player, opponent)``` that returns a tuple containing the attacker's
and defender's strength values (including assists). Here, we employ a simple heuristic that adds a position to the list of
potential assist positions if it adjacent to and opponent in a block situation with equal strength, if the position doesn't
have any other adjacent opponents. This should raise the attackers strength to our favor. When such positions are found,
we scan for players than can safely move to one of them.

```python
for player in open_players:
    for assist_position in assist_positions:
        path = pf.get_safest_path(game, player, assist_position)
        if path is not None and path.prob == 1:
            self.actions.append(Action(ActionType.START_MOVE, player=player))
            if not player.state.up:
                self.actions.append(Action(ActionType.STAND_UP))
            for step in path.steps:
                self.actions.append(Action(ActionType.MOVE, position=step))
            return
```

## 9. Move towards the ball

The rest of the open players we have, we will just move safely towards the ball. Much smarter defensive moves could be
made, such as screening the backfield. This approach is, however, simpler and will put direct pressure on the opponent
ball carrier, block passage towards our own ball carrier, or just move players towards a free ball on the ground. The
code to do this will use concepts that we have already covered.

```python
for player in open_players:
    if player == ball_carrier:
        continue
    if game.num_tackle_zones_in(player) > 0:
        continue
    if ball_carrier is None:
        path = pf.get_safest_path(game, player, game.get_ball_position(), max_search_distance=30)
    else:
        path = pf.get_safest_path_to_player(game, player, ball_carrier)
    if path is not None:
        steps = []
        for step in path.steps:
            if len(steps) >= player.get_ma() + (3 if not player.state.up else 0):
                break
            if ball_carrier is not None and ball_carrier.team == self.my_team and step in game.get_adjacent_squares(ball_carrier.position):
                break
            steps.append(step)
            if game.num_tackle_zones_at(player, step) > 0:
                break
        if len(steps) > 0:
            self.actions.append(Action(ActionType.START_MOVE, player=player))
            if not player.state.up:
                self.actions.append(Action(ActionType.STAND_UP))
            for step in steps:
                self.actions.append(Action(ActionType.MOVE, position=step))
            return
```

## 10. Execute risky blocks

Why not block with players, where there is a higher chance of knocking down the defender than knocking down the attacker.
If there is a chance that we can cause a fumble, we should also do the block regardless of the risk.

```python
attacker, defender, p_self_up, p_opp_down, block_p_fumble_self, block_p_fumble_opp = self._get_safest_block(game)
if attacker is not None and (p_opp_down > (1-p_self_up) or block_p_fumble_opp > 0):
    self.actions.append(Action(ActionType.START_BLOCK, player=attacker))
    self.actions.append(Action(ActionType.BLOCK, position=defender.position))
    return
```

## 11. End turn

If none of the other parts resulted in an action, we should just end the turn.

## Evaluation

This bot wins consistently against the random baseline from the previous tutorial. In 10 games, this bot won all of them
with an average of 3.5 touchdowns per game. The video below shows an example of such a game. To run this evaluation yourself,
uncomment the code at the end of [https://github.com/njustesen/ffai/blob/master/examples/scripted_bot_example.py](examples/scripted_bot_example.py) and run it.

## Next steps

While this bot is good against the random baseline, it can easily be exploited by smarter bots or by human players. Try
a game against it to identify its weaknesses, and then see if you can improve it. You can run the web server to play against
the bot by running python script in [https://github.com/njustesen/ffai/blob/master/examples/scripted_bot_example.py](examples/scripted_bot_example.py). Remember to comment out the evaluation part first, if
you have activated it.

In the next tutorial, we will take a look at kick-off formations (coming soon).
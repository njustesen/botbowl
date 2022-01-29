# Scripted Bots II: Pathfinding and Probabilities

In this tutorial, we will walk you through the code of a fully-fledged bot that is able to consistently beat the random bot
from the previous tutorial. To achieve this, we will make heavy use of the pathfinding and probability features of botbowl.
There will be plenty things to improve on after this tutorial while some intelligent behaviors will emerge. The complete code of the scripted bot that we will describe is in [https://github.com/njustesen/botbowl/blob/master/examples/scripted_bot_example.py](examples/scripted_bot_example.py).

## Architecture
We will use the procedure-based bot that was introduced in the previous tutorial. This will allow us to implement simple rules for most of the procedures so we can focus on the few procedures that require important planning and decision-making. The
bot that we will describe follows a prioritized lists of decisions, starting with simple and safe player actions in the beginning and
ending with more risky and complicated actions. The list will be traversed before each player takes a decision, such that a risky action taken by one player can open up several safe actions by others. This approach stands in contrast to the design of Grodbot, the
winnner of Bot Bowl I, wherein many possible actions are evaluated for all players at the same time and then the most promising action
is taken. Grodbots approach is possibly more potent while it is difficult to assign numerical values to all types of actions. Our approach implements the following list of priorities, which allows us
to implement easily understood and isolated heuristics.

The list of priorities are:
1. Stand players up that are down and marked, and then end their turn
2. Move the ball carrier safely towards the endzone
3. Make safe blocks
4. Pickup the ball if it's on the ground
5. Move receivers into scoring range
6. Make a blitz move
7. Make cage around ball carrier
8. Move non-marked players to assist
9. Move towards the ball
10. Make risky blocks
11. End turn

The bot plans out a complete player turn at each step during its turn. A list of actions is maintained and executed in order.
When a player ends its turn, the ```_make_plan(game, ball_carrier)``` function is called that will plan a new list of actions for a new player until no reasonable player actions are left.

The rest of this tutorial will walk you through the eleven decisions listed above.

## 1. Stand up

We simply iterate through all players on our team and if we find a player that is on the field (```player.position is not None```),
is not standing up, not stunned, and not used, then we plan a move action, stand the player up. We return to stop planning which will execute the actions we have appended to ```self.actions````.

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

Notice, that our bot can perform a hand-off action but not a pass action. We leave this as an exercise for you to implement in the end. 

We will start by describing the steps needed to move the playing into the endzone if it's safe to do so.

```python
# 2.1 Can ball carrier score with high probability
td_path = pf.get_safest_path_to_endzone(game, ball_carrier, allow_team_reroll=True)
if td_path is not None and td_path.prob >= 0.7:
    self.actions.append(Action(ActionType.START_MOVE, player=ball_carrier))
    self.actions.extend(path_to_move_actions(game, td_path))
    return
```

First, we use the pathfinding module (botbowl.ai.pathfinding) to get the safest (and then shortest) path for the ball carrier to the opponent's endzone. If ```pf.get_safest_scoring_path(game, ball_carrier)``` returns ```None```, then no path was found, which
means that either the path is blocked or the player doesn't have enough moves left. If a path was found, the Path instance
will have a probability of success in [0,1] and a list of steps. Here, we decide to move the ball carrier to the endzone if the probability of success is at least 0.7 (70%). This is absolutely not a perfect heuristic, especially because we haven't
yet checked if we can make any blitzes to clear the path for a safer option. Nevertheless, we will settle for this simple
solution.

Finally we are introduced to the function `path_to_move_actions()` which takes the game object and a path as arguments. It returns a list of actions corresponding to the steps in the path.  

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
    for player in unused_teammates:
        if game.get_distance_to_endzone(player) > player.num_moves_left():
            continue
        td_path = pf.get_safest_path_to_endzone(game, player, allow_team_reroll=True)
        if td_path is None:
            continue
        handoff_path = pf.get_safest_path(game, ball_carrier, player.position, allow_team_reroll=True)
        if handoff_path is None:
            continue
        p_catch = game.get_catch_prob(player, handoff=True, allow_catch_reroll=True, allow_team_reroll=True)
        p = td_path.prob * handoff_path.prob * p_catch
        if handoff_p is None or p > handoff_p:
            handoff_p = p
            handoff_path = handoff_path

    # Hand-off if high probability or last turn
    if handoff_path is not None and (handoff_p >= 0.7 or self.my_team.state.turn == 8):
        self.actions.append(Action(ActionType.START_HANDOFF, player=ball_carrier))
        self.actions.extend(path_to_move_actions(game, handoff_path))
        return
```

First, we check if a hand-off actions is available. If not, we scan the field for unused players on our team that are standing. We check if each player is in scoring distance and save the safest of the available choices. The probability of
success is computed as a product of three components: 
1) the probability of success for moving the ball carrier to the target player,
2) the probability for catching the ball, and 
3) the probability of success for moving the target player to the endzone.

If the hand-off is successful, step 2.1 from before will make sure that the target player will attempt to score.
The downside of this approach, is that the target player's path to endzone is computed in a state where that current ball carrier is
not necessarily adjacent, i.e. before the ball carriers is moved. This could mean that the ball carrier will block the scoring
path of the target player. This an effect of not doing perfect forward planning but it should be okay in most situations. To overcome this potential issue, we would need to carefully move the ball carrier to the new square, run pathfinding for the target player, and then place the ball carrier at its original position.

The last step for the ball carrier, if none of the above were possible or safe enough, is to just move the safely towards the endzone. We do this by first calling ```pf.get_all_paths(game, ball_carrier)``` to find the safest paths to all the reachable squares of the player. This function is actually quite efficient and not much slower than finding the safest path to just one square.
After receiving the safest paths to all the reachable squares we pick the one nearest the opponent's endzone that doesn't require any dice rolls. 
```python
if game.num_tackle_zones_in(ball_carrier) == 0:
    paths = pf.get_all_paths(game, ball_carrier)
    best_path = None
    best_distance = 100
    target_x = game.get_opp_endzone_x(self.my_team)
    for path in paths:
        distance_to_endzone = abs(target_x - path.steps[-1].x)
        if path.prob == 1 and (best_path is None or distance_to_endzone < best_distance) and game.num_tackle_zones_at(ball_carrier, path.get_last_step()) == 0:
            best_path = path
            best_distance = distance_to_endzone
    if best_path is not None:
        self.actions.append(Action(ActionType.START_MOVE, player=ball_carrier))
        self.actions.extend(path_to_move_actions(game, best_path))
        #print(f"Move ball carrier {ball_carrier.role.name}")
        return
```

## 3. Safe blocks

If we can't score, we might just make a few safe blocks. The code below will first find the safest block in terms of avoiding an attacker down result. Additionally, it will not select a block where there is a chance of loosing the ball if we have possession of the ball. If for the safest block we have more than 94% chance of staying upright and there is no risk of loosing the ball, then
we will perform the block immediately.

```python
attacker, defender, p_self_up, p_opp_down, block_p_fumble_self, block_p_fumble_opp = self._get_safest_block(game)
if attacker is not None and p_self_up > 0.94 and block_p_fumble_self == 0:
    self.actions.append(Action(ActionType.START_BLOCK, player=attacker))
    self.actions.append(Action(ActionType.BLOCK, position=defender.position))
    return
```

Without describing in detail how ```self._get_safest_block(game)``` works (it is pretty straightforward), it is useful to know
that it makes use of botbowl's ```game.get_block_probs(attacker, defender)```, that returns a tuple contains four probabilites:

1) the probability of the attacker being knocked down, 
2) the probability of the defender being knocked down, 
3) the probability of the attacker to fumble the ball, and 
4) the probability of the defender to fumble the ball. 

This function makes use of lower-level functions such as ```game.num_block_dice(attacker, defender)``` and ```game.get_push_squares(attacker, defender)``` to check for __crowd surfing__.
Notice, that the use of team-rerolls is not included in the calculation of the probabilities. 

## 4. Pickup the ball

If the ball is on the ground we should pick it up! We first iterate all unused players on the field to see if they are in reach of the ball. We use botbowl's function
```game.get_pickup_prob(player, game.get_ball_position(), allow_team_reroll=True)``` to compute the probability for the player
to pick up the ball. This function includes the option to compute the probability using team re-rolls, which we use in this case.
We then multiply the pick-up probability with the probability of success for moving to the ball. If there is more than 33% chance
of success, we will perform the move. After the move, we will move towards the endzone until a dodge, a GFI roll, or we would move into an opponent tackle zone.
Notice, that we here use the ```from_position``` and ```num_moves_used``` and arguments in call to ```pf.get_all_paths()```. This moves the player to that position using ```num_moves_used``` moves and safely resets the player after the path has been computed.

```python
if game.get_ball_carrier() is None:
    pickup_p = None
    pickup_player = None
    pickup_path = None
    for player in self.my_team.players:
        if player.position is not None and not player.state.used:
            if player.position.distance(game.get_ball_position()) <= player.get_ma() + 2:
                path = pf.get_safest_path(game, player, game.get_ball_position())
                if path is not None:
                    p = path.prob
                    if pickup_p is None or p > pickup_p:
                        pickup_p = p
                        pickup_player = player
                        pickup_path = path
    if pickup_player is not None and pickup_p > 0.33:
        self.actions.append(Action(ActionType.START_MOVE, player=pickup_player))
        if not pickup_player.state.up:
            self.actions.append(Action(ActionType.STAND_UP))
        self.actions.extend(path_to_move_actions(game, pickup_path))
        #print(f"Pick up the ball with {pickup_player.role.name}, p={pickup_p}")
        # Find safest path towards endzone
        if game.num_tackle_zones_at(pickup_player, game.get_ball_position()) == 0 and game.get_opp_endzone_x(self.my_team) != game.get_ball_position().x:
            paths = pf.get_all_paths(game, pickup_player, from_position=game.get_ball_position(), num_moves_used=len(pickup_path))
            best_path = None
            best_distance = 100
            target_x = game.get_opp_endzone_x(self.my_team)
            for path in paths:
                distance_to_endzone = abs(target_x - path.steps[-1].x)
                if path.prob == 1 and (best_path is None or distance_to_endzone < best_distance) and game.num_tackle_zones_at(pickup_player, path.get_last_step()) == 0:
                    best_path = path
                    best_distance = distance_to_endzone
            if best_path is not None:
                self.actions.extend(path_to_move_actions(game, best_path, do_assertions=False))
                #print(f"- Move ball carrier {pickup_player.role.name}")
        return
```

Here we provide `path_to_move_actions()` with the optional argument `do_assertions=False`. By default, this function has a few sanity checks, by providing this argument we turn them off. 
We want to turn them off in this case because `best_path` was created assuming the `pickup_player` was at the ball's position. But when `path_to_move_actions()` is called, this is not yet the case. Again this is because we don't have perfect forward planning.     

## 5. Move receivers into scoring range
As we allow the ball carrier to make hand-off actions to players in scoring range, it makes sense to move some players
close to the endzone. First, we scan the field for unused players on the field that are not in opponent tackle zones.
Then, if the have the __Catch__ skill and aren't the ball carrier, we will move one of them towards the endzone until they
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
        paths = pf.get_all_paths(game, player)
        best_path = None
        best_distance = 100
        target_x = game.get_opp_endzone_x(self.my_team)
        for path in paths:
            distance_to_endzone = abs(target_x - path.steps[-1].x)
            if path.prob == 1 and (best_path is None or distance_to_endzone < best_distance) and game.num_tackle_zones_at(player, path.get_last_step()):
                best_path = path
                best_distance = distance_to_endzone
        if best_path is not None:
            self.actions.append(Action(ActionType.START_MOVE, player=player))
            if not player.state.up:
                self.actions.append(Action(ActionType.STAND_UP))
            self.actions.extend(path_to_move_actions(game, best_path))
            return
```

## 6. Make a blitz move

Deciding where to make a blitz action requires a lot of planning. Each player can move to and block any opponent player
in range from several different angles. We will prioritize a good amount of time for this decision by checking many of
these options. We will, however, ignore potential blitzing players that do not have the __Block__ skill. Each available blitz
move is rated using the following summation: ```score = p_self_up + p_opp + p_fumble_opp - p_fumble_self```, which is the
probability of staying upright plus the probability of knocking down the opponent plus the probability of causing the opponent
to fumble minus the probability of loosing the ball possession. We will execute the highest scoring blitz if it has a score
of 1.25 or higher, e.g. 0.92 + 0.44 + 0.0 - 0.08 = 1.28, which is a reasonably good blitz to take.

Here, we use ```pf.get_all_paths(game, blitzer, blitz=True)``` to find all path to blitz positions. Additionally, it will provide us 
with the probablity of moving these positions plus potential GFI attemps required to perform the block. 

```python
if game.is_blitz_available():

    best_blitz_attacker = None
    best_blitz_defender = None
    best_blitz_score = None
    best_blitz_path = None
    for blitzer in open_players:
        if blitzer.position is not None and not blitzer.state.used and blitzer.has_skill(Skill.BLOCK):
            blitz_paths = pf.get_all_paths(game, blitzer, blitz=True)
            for path in blitz_paths:
                defender = game.get_player_at(path.get_last_step())
                if defender is None:
                    continue
                from_position = path.steps[-2] if len(path.steps)>1 else blitzer.position
                p_self, p_opp, p_fumble_self, p_fumble_opp = game.get_blitz_probs(blitzer, from_position, defender)
                p_self_up = path.prob * (1-p_self)
                p_opp = path.prob * p_opp
                p_fumble_opp = p_fumble_opp * path.prob
                if blitzer == game.get_ball_carrier():
                    p_fumble_self = path.prob + (1 - path.prob) * p_fumble_self
                score = p_self_up + p_opp + p_fumble_opp - p_fumble_self
                if best_blitz_score is None or score > best_blitz_score:
                    best_blitz_attacker = blitzer
                    best_blitz_defender = defender
                    best_blitz_score = score
                    best_blitz_path = path
    if best_blitz_attacker is not None and best_blitz_score >= 1.25:
        self.actions.append(Action(ActionType.START_BLITZ, player=best_blitz_attacker))
        self.actions.extend(path_to_move_actions(game, best_blitz_path))
        return

```

## 7. Make a cage around the ball carrier

If we have possession of the ball, we would like to protect the ball carrier and if the opponent has the ball, we would
like to surround the ball carrier in as similar way. Ideally, we would like to surround the ball carrier in different ways
depending on whos has possession of the ball and the position on the board. To make it simple, we will just attempt to make a classic Blood Bowl cage around the ball carrier regardless of these other factors. A cage is a formation where several players (ideally four)
are diagonally adjacent to the ball carrier. The way we achieve this should now be familiar with you, besides an additional check to see of a cage position is out of bounds.

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
                path = pf.get_safest_path(game, player, cage_position)
                if path is not None and path.prob > 0.94:
                    self.actions.append(Action(ActionType.START_MOVE, player=player))
                    self.actions.extend(path_to_move_actions(game, path))
                    return

```

## 8. Move non-marked players to assist

If we can't move the ball nor make any safe blocks, we should move players to assisting positions. First, we scan
the field for possible assisting positions by iterating the opponent players.

```python
# Scan for assist positions
assist_positions = set()
for player in game.get_opp_team(self.my_team).players:
    if player.position is None or not player.state.up:
        continue
    for opponent in game.get_adjacent_opponents(player, down=False):
        att_str, def_str = game.get_block_strengths(player, opponent)
        if def_str >= att_str:
            for open_position in game.get_adjacent_squares(player.position, occupied=False):
                if len(game.get_adjacent_players(open_position, team=self.opp_team, down=False)) == 1:
                    assist_positions.add(open_position)
```

We make use of the function ```game.get_block_strengths(player, opponent)``` that returns a tuple containing the attacker's
and defender's strength values (including assists). Here, we employ a simple heuristic that adds a position to the list of
potential assist positions if it is adjacent to and opponent in a block situation where adding an assist would equalize the strength values or give us the favor. We thus also check if the position doesn't
have any other adjacent opponents. When such positions are found, we scan for players that can safely move to one of them.

```python
for player in open_players:
    for path in pf.get_all_paths(game, player): 
        if path.prob < 1.0 or path.get_last_step() not in assist_positions:
            continue
        self.actions.append(Action(ActionType.START_MOVE, player=player))
        self.actions.extend(path_to_move_actions(game, path))
        return
```

## 9. Move towards the ball

The rest of the open players we have, we will just move safely towards the ball unless we already have it. Much smarter defensive moves could be
made, such as screening the backfield. This approach is, however, simpler and will put direct pressure on the opponent
ball carrier or just move players towards a free ball on the ground. The
code to do this will use concepts that we have already covered.

```python
for player in open_players:
    if player == ball_carrier or game.num_tackle_zones_in(player) > 0:
        continue

    shortest_distance = None
    path = None

    if ball_carrier is None:
        for p in pf.get_all_paths(game, player):
            distance = p.get_last_step().distance(game.get_ball_position())
            if shortest_distance is None or (p.prob == 1 and distance < shortest_distance):
                shortest_distance = distance
                path = p
    elif ball_carrier.team != self.my_team:            
        for p in pf.get_all_paths(game, player):
            distance = p.get_last_step().distance(ball_carrier.position)
            if shortest_distance is None or (p.prob == 1 and distance < shortest_distance):
                shortest_distance = distance
                path = p

    if path is not None:
        self.actions.append(Action(ActionType.START_MOVE, player=player))
        self.actions.extend(path_to_move_actions(game, path))
        return
```

## 10. Execute risky blocks

At last we will block with players where there is a higher chance of knocking down the defender than knocking down the attacker.
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

This bot wins consistently against the random baseline from the previous tutorial. In 10 games, this bot won all of them with an average of 3.5 touchdowns per game. To run this evaluation yourself,
uncomment the code at the end of [examples/scripted_bot_example.py](https://github.com/njustesen/botbowl/blob/master/examples/scripted_bot_example.py) and run it.

## Next steps

While this bot is good against the random baseline, it can easily be exploited by smarter bots or by human players. Try a game against it to identify its weaknesses and  see if you can improve it. You can run the web server to play against the bot by running the python script in [https://github.com/njustesen/botbowl/blob/master/examples/scripted_bot_example.py](examples/scripted_bot_example.py). Remember to comment out the evaluation part of the code if you have activated it.

Some ideas for improvement:
- Passing actions
- Add follow up logic
- Improve reroll logic when blocking the opponent ball carrier
- Stay away from the sideline!
- Foul actions

In the next tutorial, we will take a look at [Kick-off formations](bots-iii.md).

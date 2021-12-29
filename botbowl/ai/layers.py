"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains the feature layers used by the gym implementation.
"""

from botbowl.core.procedure import *
from botbowl.core.game import Game


class FeatureLayer:

    def __init__(self):
        self.cache = {}

    def name(self):
        raise NotImplementedError("Must be overridden by subclass")

    def get(self, game: Game):
        """
        :return: a 2D 1-hot feature layer, possibly cached.
        """
        key = self.key(game)
        if key is not None and key in self.cache:
            return self.cache[key]
        layer = self.produce(game)
        if key is not None:
            self.cache[key] = layer
        return layer

    def key(self, game):
        """
        Override this to use caching. Any state layer with a key other than None will be stored and reused.
        :return: a unique key for each possible layer outcome.
        """
        return None
        
    def produce(self, game):
        """
        :param game:
        :return: a newly generated 2D 1-hot feature layer.
        """
        raise NotImplementedError("Must be overridden by subclass")
        

class OccupiedLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))

        for player in game.state.home_team.players + game.state.home_team.players:
            if player.position is not None:
                x = player.position.x
                y = player.position.y
                out[y][x] = 1.0 

        return out

    def key(self, game):
        return None

    def name(self):
        return "occupied"


class OwnPlayerLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        
        for p in active_team.players: 
            if p.position is not None: 
                x = p.position.x 
                y = p.position.y 
                out[y][x] = 1.0

        return out

    def key(self, game):
        return None

    def name(self):
        return "own players"


class OppPlayerLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        
        if active_team == game.state.home_team: 
            target_team = game.state.away_team 
        else: 
            target_team = game.state.home_team 
        
        for p in target_team.players: 
            if p.position is not None: 
                x = p.position.x 
                y = p.position.y 
                out[y][x] = 1.0

        return out

    def key(self, game):
        return None

    def name(self):
        return "opp players"


class OwnTackleZoneLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for player in active_team.players:
            if player.position is not None:
                if player.has_tackle_zone():
                    for square in game.get_adjacent_squares(player.position):
                        out[square.y][square.x] += 0.125
        return out

    def key(self, game):
        return None

    def name(self):
        return "own tackle zones"


class OppTackleZoneLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for player in game.get_opp_team(active_team).players:
            if player.position is not None:
                if player.has_tackle_zone():
                    for square in game.get_adjacent_squares(player.position):
                        out[square.y][square.x] += 0.125
        return out

    def key(self, game):
        return None

    def name(self):
        return "opp tackle zones"


class UsedLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for player in active_team.players:
            if player.position is not None:
                out[player.position.y][player.position.x] = 1.0 if player.state.used else 0.0
        for player in game.get_opp_team(active_team).players:
            if player.position is not None:
                out[player.position.y][player.position.x] = 1.0 if player.state.used else 0.0
        return out

    def key(self, game):
        return None

    def name(self):
        return "used players"


class UpLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for player in active_team.players:
            if player.position is not None:
                out[player.position.y][player.position.x] = 1.0 if player.state.up else 0.0
        for player in game.get_opp_team(active_team).players:
            if player.position is not None:
                out[player.position.y][player.position.x] = 1.0 if player.state.up else 0.0
        return out

    def key(self, game):
        return None

    def name(self):
        return "standing players"


class StunnedLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for player in active_team.players:
            if player.position is not None:
                out[player.position.y][player.position.x] = 1.0 if player.state.stunned else 0.0
        for player in game.get_opp_team(active_team).players:
            if player.position is not None:
                out[player.position.y][player.position.x] = 1.0 if player.state.stunned else 0.0
        return out

    def key(self, game):
        return None

    def name(self):
        return "stunned players"


class ActivePlayerLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))

        if game.state.active_player is None or game.state.active_player.position is None:
            return out

        out[game.state.active_player.position.y][game.state.active_player.position.x] = 1.0
        return out

    def key(self, game):
        return None

    def name(self):
        return "active players"


class TargetPlayerLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        target = None
        for i in reversed(range(game.state.stack.size())):
            proc = game.state.stack.items[i]
            if isinstance(proc, Block):
                target = proc.defender
                break
            if isinstance(proc, PassAttempt):
                target = proc.catcher
                break
            if isinstance(proc, Handoff):
                target = proc.catcher
                break
            if isinstance(proc, Foul):
                target = proc.defender
                break
        if target is not None and target.position is not None:
            out[target.position.y][target.position.x] = 1.0
        return out

    def key(self, game):
        return None

    def name(self):
        return "target player"


class AvailablePositionLayer(FeatureLayer):

    def __init__(self, action_type):
        super().__init__()
        self.action_type = action_type

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        for action_choice in game.state.available_actions:
            if action_choice.action_type != self.action_type:
                continue
            for position in action_choice.positions:
                if position is not None:
                    out[position.y][position.x] = 1.0
            if len(action_choice.positions) == 0:
                for player in action_choice.players:
                    if player.position is not None:
                        out[player.position.y][player.position.x] = 1.0
            break
        return out

    def key(self, game):
        return None

    def name(self):
        return f"{self.action_type.name.replace('_', ' ').lower()} positions"


class RollProbabilityLayer(FeatureLayer):

    # The probability of sum of two D6 rolls to be equal to or greater than roll_target. e.g. 9+ = 0.2777..
    accumulated_prob_2d_roll = (np.array([36, 36, 36, 35, 33, 30, 26, 21, 15, 10, 6, 3, 1])/36)

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for action_choice in game.state.available_actions:
            for i in range(len(action_choice.positions)):
                if action_choice.positions[i] is not None:
                    if len(action_choice.paths) == len(action_choice.positions):
                        out[action_choice.positions[i].y][action_choice.positions[i].x] = action_choice.paths[i].prob
                    elif i < len(action_choice.rolls):
                        # Convert to chance of succeeding
                        if action_choice.action_type == ActionType.FOUL:
                            # Use different probability calculation for 2D6 rolls
                            chance = self.accumulated_prob_2d_roll[action_choice.rolls[i][0]]
                        else:
                            chance = 1.0
                            for roll in action_choice.rolls[i]:
                                chance = chance * ((1+(6-roll)) / 6)
                        out[action_choice.positions[i].y][action_choice.positions[i].x] = chance
        return out

    def key(self, game):
        return None

    def name(self):
        return "roll probabilities"


class BlockDiceLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for action_choice in game.state.available_actions:
            for i in range(len(action_choice.positions)):
                if action_choice.positions[i] is not None:
                    roll = 0
                    if i < len(action_choice.block_dice):
                        roll = (action_choice.block_dice[i] + 3) / 6.0
                    elif i < len(action_choice.paths):
                        if action_choice.paths[i].block_dice is not None:
                            roll = (action_choice.paths[i].block_dice + 3) / 6.0
                    out[action_choice.positions[i].y][action_choice.positions[i].x] = roll

        return out

    def key(self, game):
        return None

    def name(self):
        return "block dice"


class MALayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for player in active_team.players:
            if player.position is not None:
                out[player.position.y][player.position.x] = player.get_ma() * 0.1
        for player in game.get_opp_team(active_team).players:
            if player.position is not None:
                out[player.position.y][player.position.x] = player.get_ma() * 0.1
        return out

    def key(self, game):
        return None

    def name(self):
        return "movement allowence"


class STLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for player in active_team.players:
            if player.position is not None:
                out[player.position.y][player.position.x] = player.get_st() * 0.1
        for player in game.get_opp_team(active_team).players:
            if player.position is not None:
                out[player.position.y][player.position.x] = player.get_st() * 0.1
        return out

    def key(self, game):
        return None

    def name(self):
        return "strength"


class AGLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for player in active_team.players:
            if player.position is not None:
                out[player.position.y][player.position.x] = player.get_ag() * 0.1
        for player in game.get_opp_team(active_team).players:
            if player.position is not None:
                out[player.position.y][player.position.x] = player.get_ag() * 0.1
        return out

    def key(self, game):
        return None

    def name(self):
        return "agility"


class AVLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for player in active_team.players:
            if player.position is not None:
                out[player.position.y][player.position.x] = player.get_av() * 0.1
        for player in game.get_opp_team(active_team).players:
            if player.position is not None:
                out[player.position.y][player.position.x] = player.get_av() * 0.1
        return out

    def key(self, game):
        return None

    def name(self):
        return "armor value"


class SkillLayer(FeatureLayer):

    def __init__(self, skill):
        super().__init__()
        self.skill = skill

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for player in active_team.players:
            if player.position is not None:
                out[player.position.y][player.position.x] = 1 if player.has_skill(self.skill) else 0.0
        for player in game.get_opp_team(active_team).players:
            if player.position is not None:
                out[player.position.y][player.position.x] = 1 if player.has_skill(self.skill) else 0.0
        return out

    def key(self, game):
        return None

    def name(self):
        return self.skill.name.replace("_", " ").lower()


class BallLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        for ball in game.state.pitch.balls:
            if ball.position is not None:
                out[ball.position.y][ball.position.x] = 1.0
        return out

    def key(self, game):
        return None

    def name(self):
        return "balls"


class OwnHalfLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        home = active_team == game.state.home_team
        tiles = TwoPlayerArena.home_tiles if home else TwoPlayerArena.away_tiles
        for y in range(len(game.arena.board)):
            for x in range(len(game.arena.board[0])):
                out[y][x] = 1.0 if game.arena.board[y][x] in tiles else 0.0
        return out
    
    def key(self, game):
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        home = active_team == game.state.home_team
        return str(home)

    def name(self):
        return "own half"


class OwnTouchdownLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        home = active_team == game.state.home_team
        tile = Tile.HOME_TOUCHDOWN if home else Tile.AWAY_TOUCHDOWN
        for y in range(len(game.arena.board)):
            for x in range(len(game.arena.board[0])):
                out[y][x] = 1.0 if game.arena.board[y][x] == tile else 0.0
        return out

    def key(self, game):
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        home = active_team == game.state.home_team
        return str(home)
        
    def name(self):
        return "own touchdown"


class OppTouchdownLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        home = active_team == game.state.home_team
        tile = Tile.HOME_TOUCHDOWN if not home else Tile.AWAY_TOUCHDOWN
        for y in range(len(game.arena.board)):
            for x in range(len(game.arena.board[0])):
                out[y][x] = 1.0 if game.arena.board[y][x] == tile else 0.0
        return out

    def key(self, game):
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        home = active_team == game.state.home_team
        return str(home)    
        
    def name(self):
        return "opp touchdown"


class CrowdLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        for y in range(len(game.arena.board)):
            for x in range(len(game.arena.board[0])):
                out[y][x] = 1.0 if game.arena.board[y][x] == Tile.CROWD else 0.0
        return out

    def key(self, game):
        return 0

    def name(self):
        return "opp crowd"


class MovementLeftLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for player in active_team.players + game.get_opp_team(active_team).players:
            if player.position is not None:
                out[player.position.y][player.position.x] = max(0, (player.get_ma() - player.state.moves) * 0.1)
        return out

    def name(self):
        return "movement left"


class GFIsLeftLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for player in active_team.players + game.get_opp_team(active_team).players:
            if player.position is not None:
                num_max_gfis = 3 if player.has_skill(Skill.SPRINT) else 2
                out[player.position.y][player.position.x] = \
                    min(0.1 * num_max_gfis, (player.get_ma() + num_max_gfis - player.state.moves) * 0.1)
        return out

    def name(self):
        return "gfi left"

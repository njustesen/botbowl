import numpy as np
from bb.core.table import *
from bb.core.model import *


class FeatureLayer:

    def produce(self, game):
        raise NotImplementedError("Must be overridden by subclass")

    def name(self):
        raise NotImplementedError("Must be overridden by subclass")


class OccupiedLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        for y in range(len(game.state.pitch.board)):
            for x in range(len(game.state.pitch.board[0])):
                out[y][x] = 1.0 if game.state.pitch.board[y][x] is not None else 0.0
        return out

    def name(self):
        return "occupied"


class OwnPlayerLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for y in range(len(game.state.pitch.board)):
            for x in range(len(game.state.pitch.board[0])):
                out[y][x] = 1.0 if game.state.pitch.board[y][x] is not None and \
                                   game.state.pitch.board[y][x].team == active_team is not None else 0.0
        return out

    def name(self):
        return "own players"


class OppPlayerLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for y in range(len(game.state.pitch.board)):
            for x in range(len(game.state.pitch.board[0])):
                out[y][x] = 1.0 if game.state.pitch.board[y][x] is not None and \
                                   game.state.pitch.board[y][x].team != active_team is not None else 0.0
        return out

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
                    for square in game.state.pitch.get_adjacent_squares(player.position):
                        out[square.y][square.x] += 0.125
        return out

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
                    for square in game.state.pitch.get_adjacent_squares(player.position):
                        out[square.y][square.x] += 0.125
        return out

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

    def name(self):
        return "ready players"


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

    def name(self):
        return "down players"


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

    def name(self):
        return "stunned players"


class ActivePlayerLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))

        if game.state.active_player is None or game.state.active_player.position is None:
            return out

        out[game.state.active_player.position.y][game.state.active_player.position.x] = 1.0
        return out

    def name(self):
        return "active players"


class AvailablePlayerLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for action_choice in game.state.available_actions:
            for player in action_choice.players:
                if player.position is not None:
                    out[player.position.y][player.position.x] = 1.0
        return out

    def name(self):
        return "available players"


class AvailablePositionLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for action_choice in game.state.available_actions:
            for position in action_choice.positions:
                if position is not None:
                    out[position.y][position.x] = 1.0
            for player in action_choice.players:
                if player.position is not None:
                    out[player.position.y][player.position.x] = 1.0
        return out

    def name(self):
        return "available positions"


class RollProbabilityLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for action_choice in game.state.available_actions:
            for i in range(len(action_choice.positions)):
                if action_choice.positions[i] is not None:
                    if i < len(action_choice.agi_rolls):
                        # Convert to chance of succeeding
                        chance = 1.0
                        for roll in action_choice.agi_rolls[i]:
                            chance = chance * ((1+(6-roll)) / 6)
                        out[action_choice.positions[i].y][action_choice.positions[i].x] = chance
        return out

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
                    if i < len(action_choice.block_rolls):
                        roll = (action_choice.block_rolls[i] + 3) / 6.0
                        out[action_choice.positions[i].y][action_choice.positions[i].x] = roll
        return out

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

    def name(self):
        return "armor value"


class SkillLayer(FeatureLayer):

    def __init__(self, skill):
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

    def name(self):
        return self.skill.name.replace("_", " ").lower()


class MovemenLeftLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        if active_team is None:
            return out
        for player in active_team.players:
            if player.position is not None:
                out[player.position.y][player.position.x] = (player.get_ma() - player.state.moves) * 0.1
        for player in game.get_opp_team(active_team).players:
            if player.position is not None:
                out[player.position.y][player.position.x] = (player.get_ma() - player.state.moves) * 0.1
        return out

    def name(self):
        return "movement left"


class BallLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        for ball in game.state.pitch.balls:
            if ball.position is not None:
                out[ball.position.y][ball.position.x] = 1.0
        return out

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

    def name(self):
        return "opp touchdown"


class CrowdLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        for y in range(len(game.arena.board)):
            for x in range(len(game.arena.board[0])):
                out[y][x] = 1.0 if game.arena.board[y][x] == Tile.CROWD else 0.0
        return out

    def name(self):
        return "opp touchdown"

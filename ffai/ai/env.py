"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains the FFAIEnv class; FFAI implementing the Open AI Gym interface.
"""

import gym
from gym import spaces
from ffai.core.game import *
from ffai.core.load import *
from ffai.ai.bots import RandomBot
from ffai.ai.layers import *
import uuid
import tkinter as tk
import math


class FFAIEnv(gym.Env):

    square_size = 16
    square_size_fl = 4
    top_bar_height = 42
    bot_bar_height = 80
    layer_text_height = 26
    black = '#000000'
    white = '#ffffff'
    crowd = '#113311'
    blue = '#2277cc'
    red = '#cc7722'
    ball = '#ff00cc'
    field = '#77cc77'
    wing = '#55aa55'
    scrimmage = '#338833'

    actions = [
        ActionType.START_GAME,
        ActionType.HEADS,
        ActionType.TAILS,
        ActionType.KICK,
        ActionType.RECEIVE,
        ActionType.END_SETUP,
        ActionType.END_PLAYER_TURN,
        ActionType.USE_REROLL,
        ActionType.DONT_USE_REROLL,
        ActionType.END_TURN,
        ActionType.STAND_UP,
        ActionType.SELECT_ATTACKER_DOWN,
        ActionType.SELECT_BOTH_DOWN,
        ActionType.SELECT_PUSH,
        ActionType.SELECT_DEFENDER_STUMBLES,
        ActionType.SELECT_DEFENDER_DOWN,
        ActionType.SELECT_NONE,
        ActionType.PLACE_PLAYER,
        ActionType.PLACE_BALL,
        ActionType.PUSH,
        ActionType.FOLLOW_UP,
        ActionType.INTERCEPTION,
        ActionType.SELECT_PLAYER,
        ActionType.MOVE,
        ActionType.BLOCK,
        ActionType.PASS,
        ActionType.FOUL,
        ActionType.HANDOFF,
        ActionType.START_MOVE,
        ActionType.START_BLOCK,
        ActionType.START_BLITZ,
        ActionType.START_PASS,
        ActionType.START_FOUL,
        ActionType.START_HANDOFF,
        ActionType.SETUP_FORMATION_WEDGE,
        ActionType.SETUP_FORMATION_LINE,
        ActionType.SETUP_FORMATION_SPREAD,
        ActionType.SETUP_FORMATION_ZONE
    ]

    simple_action_types = [
        ActionType.START_GAME,
        ActionType.HEADS,
        ActionType.TAILS,
        ActionType.KICK,
        ActionType.RECEIVE,
        ActionType.END_SETUP,
        ActionType.END_PLAYER_TURN,
        ActionType.USE_REROLL,
        ActionType.DONT_USE_REROLL,
        ActionType.END_TURN,
        ActionType.STAND_UP,
        ActionType.SELECT_ATTACKER_DOWN,
        ActionType.SELECT_BOTH_DOWN,
        ActionType.SELECT_PUSH,
        ActionType.SELECT_DEFENDER_STUMBLES,
        ActionType.SELECT_DEFENDER_DOWN,
        ActionType.SELECT_NONE
    ]

    formation_action_types = [
        ActionType.SETUP_FORMATION_WEDGE,
        ActionType.SETUP_FORMATION_LINE,
        ActionType.SETUP_FORMATION_SPREAD,
        ActionType.SETUP_FORMATION_ZONE
    ]

    positional_action_types = [
        ActionType.PLACE_PLAYER,
        ActionType.PLACE_BALL,
        ActionType.PUSH,
        ActionType.FOLLOW_UP,
        ActionType.INTERCEPTION,
        ActionType.SELECT_PLAYER,
        ActionType.MOVE,
        ActionType.BLOCK,
        ActionType.PASS,
        ActionType.FOUL,
        ActionType.HANDOFF
    ]

    player_action_types = [
        ActionType.INTERCEPTION,
        ActionType.SELECT_PLAYER,
        ActionType.START_MOVE,
        ActionType.START_BLOCK,
        ActionType.START_BLITZ,
        ActionType.START_PASS,
        ActionType.START_FOUL,
        ActionType.START_HANDOFF
    ]

    # Procedures that require actions
    procedures = [
        StartGame,
        CoinTossFlip,
        CoinTossKickReceive,
        Setup,
        PlaceBall,
        HighKick,
        Touchback,
        Turn,
        PlayerAction,
        Block,
        Push,
        FollowUp,
        Apothecary,
        PassAction,
        Catch,
        Interception,
        GFI,
        Dodge,
        Pickup
    ]

    def __init__(self, config, home_team, away_team, opp_actor=None):
        self.__version__ = "0.0.1"
        self.config = config
        self.game = None
        self.team_id = None
        self.ruleset = get_rule_set(config.ruleset, all_rules=False)
        self.home_team = home_team
        self.away_team = away_team
        self.actor = Agent("Gym Learner", human=True)
        self.opp_actor = opp_actor if opp_actor is not None else RandomBot("Random")
        self.seed()
        self.root = None
        self.cv = None
        self.last_obs = None

        self.layers = [
            OccupiedLayer(),
            OwnPlayerLayer(),
            OppPlayerLayer(),
            OwnTackleZoneLayer(),
            OppTackleZoneLayer(),
            UpLayer(),
            UsedLayer(),
            AvailablePlayerLayer(),
            AvailablePositionLayer(),
            RollProbabilityLayer(),
            BlockDiceLayer(),
            ActivePlayerLayer(),
            MALayer(),
            STLayer(),
            AGLayer(),
            AVLayer(),
            MovemenLeftLayer(),
            BallLayer(),
            OwnHalfLayer(),
            OwnTouchdownLayer(),
            OppTouchdownLayer(),
            SkillLayer(Skill.BLOCK),
            SkillLayer(Skill.DODGE),
            SkillLayer(Skill.SURE_HANDS),
            SkillLayer(Skill.PASS),
            SkillLayer(Skill.BLOCK)
        ]

        arena = get_arena(self.config.arena)

        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=1, shape=(arena.height, arena.width, len(self.layers))),
            'state': spaces.Box(low=0, high=1, shape=(44,)),
            'procedure':  spaces.Box(low=0, high=1, shape=(len(FFAIEnv.procedures),)),
        })

        self.actions = FFAIEnv.actions

        self.action_space = spaces.Dict({
            'action-type': spaces.Discrete(len(self.actions)),
            'x': spaces.Discrete(arena.width),
            'y': spaces.Discrete(arena.height)
        })

    def step(self, action):
        action_type = self.actions[action['action-type']] if action['action-type'] is not None else None
        if action_type is None:
            return
        p = Square(action['x'], action['y']) if action['x'] is not None and action['y'] is not None else None
        position = None
        player = None
        if action_type in self.player_action_types:
            if p is None:
                print("p is None")
            player = self.game.get_player_at(p)
            if player is None:
                print("player is None")
            action = None
            for a in self.game.state.available_actions:
                if a.action_type == action_type:
                    action = a
            if len(action.positions) == 1:
                position = action.positions[0]
        elif action_type in self.positional_action_types:
            position = p
        real_action = Action(action_type=action_type, pos=position, player=player)
        return self._step(real_action)

    def _step(self, action):
        self.game.step(action)
        reward = 1 if self.game.get_winner() == self.actor else 0
        team = self.game.state.home_team if self.team_id == self.home_team.team_id else self.game.state.away_team
        opp_team = self.game.state.home_team if self.team_id != self.home_team.team_id else self.game.state.away_team
        info = {
            'cas_inflicted': len(self.game.get_casualties(team)),
            'opp_cas_inflicted': len(self.game.get_casualties(opp_team)),
            'touchdowns': team.state.score,
            'opp_touchdowns': opp_team.state.score,
            'half': self.game.state.round,
            'round': self.game.state.round
        }
        return self._observation(self.game), reward, self.game.state.game_over, info

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**32)
        self.rnd = np.random.RandomState(seed)
        if isinstance(self.opp_actor, RandomBot):
            self.opp_actor.rnd = self.rnd
        return seed

    def get_game(self):
        return self.game

    def _observation(self, game):
        obs = {
            'board': {},
            'state': {},
            'procedure': np.zeros(len(FFAIEnv.procedures))
        }
        for layer in self.layers:
            obs['board'][layer.name()] = layer.produce(game)

        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        opp_team = game.get_opp_team(active_team) if active_team is not None else None

        # State
        obs['state']['half'] = game.state.half - 1.0
        obs['state']['round'] = game.state.round / 8.0
        obs['state']['sweltering heat'] = 1.0 if game.state.weather == WeatherType.SWELTERING_HEAT else 0.0
        obs['state']['very sunny'] = 1.0 if game.state.weather == WeatherType.VERY_SUNNY else 0.0
        obs['state']['nice'] = 1.0 if game.state.weather.value == WeatherType.NICE else 0.0
        obs['state']['pouring rain'] = 1.0 if game.state.weather.value == WeatherType.POURING_RAIN else 0.0
        obs['state']['blizzard'] = 1.0 if game.state.weather.value == WeatherType.BLIZZARD else 0.0

        obs['state']['own turn'] = 1.0 if game.state.current_team == active_team else 0.0
        obs['state']['kicking first half'] = 1.0 if game.state.kicking_first_half == active_team else 0.0
        obs['state']['kicking this drive'] = 1.0 if game.state.kicking_this_drive == active_team else 0.0
        obs['state']['own reserves'] = len(game.get_reserves(active_team)) / 16.0 if active_team is not None else 0.0
        obs['state']['own kods'] = len(game.get_kods(active_team)) / 16.0 if active_team is not None else 0.0
        obs['state']['own casualites'] = len(game.get_casualties(active_team)) / 16.0 if active_team is not None else 0.0
        obs['state']['opp reserves'] = len(game.get_reserves(game.get_opp_team(active_team))) / 16.0 if active_team is not None else 0.0
        obs['state']['opp kods'] = len(game.get_kods(game.get_opp_team(active_team))) / 16.0 if active_team is not None else 0.0
        obs['state']['opp casualties'] = len(game.get_casualties(game.get_opp_team(active_team))) / 16.0 if active_team is not None else 0.0

        obs['state']['own score'] = active_team.state.score / 16.0 if active_team is not None else 0.0
        obs['state']['own turns'] = active_team.state.turn / 8.0 if active_team is not None else 0.0
        obs['state']['own starting rerolls'] = active_team.state.rerolls_start / 8.0 if active_team is not None else 0.0
        obs['state']['own rerolls left'] = active_team.state.rerolls / 8.0 if active_team is not None else 0.0
        obs['state']['own ass coaches'] = active_team.state.ass_coaches / 8.0 if active_team is not None else 0.0
        obs['state']['own cheerleaders'] = active_team.state.cheerleaders / 8.0 if active_team is not None else 0.0
        obs['state']['own bribes'] = active_team.state.bribes / 4.0 if active_team is not None else 0.0
        obs['state']['own babes'] = active_team.state.babes / 4.0 if active_team is not None else 0.0
        obs['state']['own apothecary available'] = 1.0 if active_team is not None and active_team.state.apothecary_available else 0.0
        obs['state']['own reroll available'] = 1.0 if active_team is not None and not active_team.state.reroll_used else 0.0
        obs['state']['own fame'] = active_team.state.fame if active_team is not None else 0.0

        obs['state']['opp score'] = opp_team.state.score / 16.0 if opp_team is not None else 0.0
        obs['state']['opp turns'] = opp_team.state.turn / 8.0 if opp_team is not None else 0.0
        obs['state']['opp starting rerolls'] = opp_team.state.rerolls_start / 8.0 if opp_team is not None else 0.0
        obs['state']['opp rerolls left'] = opp_team.state.rerolls / 8.0 if opp_team is not None else 0.0
        obs['state']['opp ass coaches'] = opp_team.state.ass_coaches / 8.0 if opp_team is not None else 0.0
        obs['state']['opp cheerleaders'] = opp_team.state.cheerleaders / 8.0 if opp_team is not None else 0.0
        obs['state']['opp bribes'] = opp_team.state.bribes / 4.0 if opp_team is not None else 0.0
        obs['state']['opp babes'] = opp_team.state.babes / 4.0 if opp_team is not None else 0.0
        obs['state']['opp apothecary available'] = 1.0 if opp_team is not None and opp_team.state.apothecary_available else 0.0
        obs['state']['opp reroll available'] = 1.0 if opp_team is not None and not opp_team.state.reroll_used else 0.0
        obs['state']['opp fame'] = opp_team.state.fame if opp_team is not None else 0.0

        obs['state']['blitz available'] = 1.0 if game.is_blitz_available() else 0.0
        obs['state']['pass available'] = 1.0 if game.is_pass_available() else 0.0
        obs['state']['handoff available'] = 1.0 if game.is_handoff_available() else 0.0
        obs['state']['foul available'] = 1.0 if game.is_foul_available() else 0.0
        obs['state']['is blitz'] = 1.0 if game.is_blitz() else 0.0
        obs['state']['is quick snap'] = 1.0 if game.is_quick_snap() else 0.0

        # Procedure
        if game.state.stack.size() > 0:
            procedure = game.state.stack.peek()
            assert procedure.__class__ in FFAIEnv.procedures
            proc_idx = FFAIEnv.procedures.index(procedure.__class__)
            obs['procedure'][proc_idx] = 1.0

        self.last_obs = obs

        return obs

    def reset(self):
        if self.rnd.rand(1)[0] >= 0.5:
            self.team_id = self.home_team.team_id
            home_agent = self.actor
            away_agent = self.opp_actor
        else:
            self.team_id = self.away_team.team_id
            home_agent = self.opp_actor
            away_agent = self.actor
        seed = self.rnd.randint(0, 2**32)
        self.game = Game(game_id=str(uuid.uuid1()),
                         home_team=self.home_team,
                         away_team=self.away_team,
                         home_agent=home_agent,
                         away_agent=away_agent,
                         config=self.config,
                         ruleset=self.ruleset,
                         seed=seed)
        self.game.init()
        return self._observation(self.game)

    def available_action_types(self):
        if isinstance(self.game.state.stack.peek(), Setup):
            return [self.actions.index(action.action_type) for action in self.game.state.available_actions if
                    action.action_type in self.actions and action.action_type != ActionType.PLACE_PLAYER]
        return [self.actions.index(action.action_type) for action in self.game.state.available_actions if action.action_type in self.actions]

    def available_positions(self, action_type):
        action = None
        for a in self.game.state.available_actions:
            if a.action_type == self.actions[action_type]:
                action = a
        if action is None:
            return []
        if action.action_type == ActionType.PUSH:
            print("Push")
        if action.action_type in FFAIEnv.player_action_types:
            return [player.position for player in action.players if player.position is not None]
        if action.action_type in FFAIEnv.positional_action_types:
            return [position for position in action.positions if position is not None]
        return []

    def _available_players(self, action_type):
        action = None
        for a in self.game.state.available_actions:
            if a.action_type == self.actions[action_type]:
                action = a
        if action is None:
            return []
        return [player.position for player in action.players]

    def _draw_player(self, player, x, y):
        if player.team == self.game.state.home_team:
            fill = FFAIEnv.blue
        else:
            fill = FFAIEnv.red
        width = max(1, player.get_st() - 1)
        if player.has_skill(Skill.BLOCK):
            outline = 'red'
        elif player.has_skill(Skill.CATCH):
            outline = 'yellow'
        elif player.has_skill(Skill.PASS):
            outline = 'white'
        elif player.get_st() > 3:
            outline = 'green'
        else:
            outline = 'grey'
        self.cv.create_oval(x, y, x + FFAIEnv.square_size, y + FFAIEnv.square_size, fill=fill, outline=outline,
                            width=width)
        text_fill = 'grey' if player.state.used else 'black'
        self.cv.create_text(x + FFAIEnv.square_size / 2,
                            y + FFAIEnv.square_size / 2,
                            text=str(player.nr), fill=text_fill)

        # State
        if not player.state.up:
            self.cv.create_line(FFAIEnv.square_size * x, FFAIEnv.square_size * y + FFAIEnv.top_bar_height,
                                FFAIEnv.square_size * x + FFAIEnv.square_size,
                                FFAIEnv.square_size * y + FFAIEnv.square_size + FFAIEnv.top_bar_height, fill='white',
                                width=1)
        if player.state.stunned:
            self.cv.create_line(FFAIEnv.square_size * x + FFAIEnv.square_size,
                                FFAIEnv.square_size * y + FFAIEnv.top_bar_height,
                                FFAIEnv.square_size * x,
                                FFAIEnv.square_size * y + FFAIEnv.square_size + FFAIEnv.top_bar_height, fill='white',
                                width=1)

    def render(self, feature_layers=False):
        if self.root is None:
            self.root = tk.Tk()
            self.root.title("FFAI Gym")
            self.game_width = max(500, self.game.arena.width*FFAIEnv.square_size)
            self.game_height = self.game.arena.height*FFAIEnv.square_size + FFAIEnv.top_bar_height + FFAIEnv.bot_bar_height
            if feature_layers:
                self.cols = math.floor(math.sqrt(len(self.layers)))
                self.rows = math.ceil(math.sqrt(len(self.layers)))
                self.fl_width = (self.game.arena.width+1) * self.cols * FFAIEnv.square_size_fl + FFAIEnv.square_size_fl
                self.fl_height = ((self.game.arena.height+1) * FFAIEnv.square_size_fl + FFAIEnv.layer_text_height) * self.rows + FFAIEnv.square_size_fl
                self.cv = tk.Canvas(width=max(self.game_width, self.fl_width), height=self.fl_height + self.game_height)
            else:
                self.cv = tk.Canvas(width=self.game_width, height=self.game_height)

        self.cv.pack(side='top', fill='both', expand='yes')
        self.cv.delete("all")
        self.root.configure(background='black')

        if self.game is not None:
            # Squares
            for y in range(self.game.arena.height):
                for x in range(self.game.arena.width):
                    if self.game.arena.board[y][x] == Tile.CROWD:
                        fill = FFAIEnv.crowd
                    elif self.game.arena.board[y][x] in TwoPlayerArena.home_td_tiles:
                        fill = FFAIEnv.blue
                    elif self.game.arena.board[y][x] in TwoPlayerArena.away_td_tiles:
                        fill = FFAIEnv.red
                    elif self.game.arena.board[y][x] in TwoPlayerArena.wing_left_tiles or self.game.arena.board[y][x] in TwoPlayerArena.wing_right_tiles:
                        fill = FFAIEnv.wing
                    elif self.game.arena.board[y][x] in TwoPlayerArena.scrimmage_tiles:
                        fill = FFAIEnv.scrimmage
                    else:
                        fill = FFAIEnv.field
                    self.cv.create_rectangle(FFAIEnv.square_size*x, FFAIEnv.square_size*y + FFAIEnv.top_bar_height, FFAIEnv.square_size*x + FFAIEnv.square_size, FFAIEnv.square_size*y + FFAIEnv.square_size + FFAIEnv.top_bar_height, fill=fill, outline=FFAIEnv.black)

            self.cv.create_line(self.game.arena.width*FFAIEnv.square_size/2.0-1, FFAIEnv.top_bar_height, self.game.arena.width*FFAIEnv.square_size/2.0-1, self.game.arena.height*FFAIEnv.square_size + FFAIEnv.top_bar_height, fill=FFAIEnv.black, width=2)

            # Players
            for y in range(self.game.state.pitch.height):
                for x in range(self.game.state.pitch.width):
                    player = self.game.state.pitch.board[y][x]
                    if player is not None:
                        self._draw_player(player, FFAIEnv.square_size*x, FFAIEnv.square_size*y + FFAIEnv.top_bar_height)

            # Dugouts
            x = 4
            y = self.game.arena.height*FFAIEnv.square_size + FFAIEnv.top_bar_height + 4
            for player in self.game.get_reserves(self.game.state.away_team):
                self._draw_player(player, x, y)
                x += FFAIEnv.square_size
            x = 4
            y += FFAIEnv.square_size
            for player in self.game.get_kods(self.game.state.away_team):
                self._draw_player(player, x, y)
                x += FFAIEnv.square_size
            x = 4
            y += FFAIEnv.square_size
            for player in self.game.get_casualties(self.game.state.away_team):
                self._draw_player(player, x, y)
                x += FFAIEnv.square_size
            x = 4
            y += FFAIEnv.square_size
            for player in self.game.get_dungeon(self.game.state.away_team):
                self._draw_player(player, x, y)
                x += FFAIEnv.square_size

            x = self.game.arena.width*FFAIEnv.square_size - FFAIEnv.square_size
            y = self.game.arena.height * FFAIEnv.square_size + FFAIEnv.top_bar_height + 4
            for player in self.game.get_reserves(self.game.state.home_team):
                self._draw_player(player, x, y)
                x -= FFAIEnv.square_size
            x = self.game.arena.width * FFAIEnv.square_size - FFAIEnv.square_size
            y += FFAIEnv.square_size
            for player in self.game.get_kods(self.game.state.home_team):
                self._draw_player(player, x, y)
                x -= FFAIEnv.square_size
            x = self.game.arena.width * FFAIEnv.square_size - FFAIEnv.square_size
            y += FFAIEnv.square_size
            for player in self.game.get_casualties(self.game.state.home_team):
                self._draw_player(player, x, y)
                x -= FFAIEnv.square_size
            x = self.game.arena.width * FFAIEnv.square_size - FFAIEnv.square_size
            y += FFAIEnv.square_size
            for player in self.game.get_dungeon(self.game.state.home_team):
                self._draw_player(player, x, y)
                x -= FFAIEnv.square_size

            # Ball
            for ball in self.game.state.pitch.balls:
                self.cv.create_oval(FFAIEnv.square_size * ball.position.x + FFAIEnv.square_size/4,
                                    FFAIEnv.square_size * ball.position.y + FFAIEnv.square_size/4 + FFAIEnv.top_bar_height,
                                    FFAIEnv.square_size * ball.position.x + FFAIEnv.square_size - FFAIEnv.square_size/4,
                                    FFAIEnv.square_size * ball.position.y + FFAIEnv.square_size - FFAIEnv.square_size/4 + FFAIEnv.top_bar_height,
                                    fill=FFAIEnv.ball, outline=FFAIEnv.black, width=1)

            # Non-spatial
            self.cv.create_text(self.game.arena.width*FFAIEnv.square_size/2.0, 10, text='Half: {}, Weather: {}'.format(self.game.state.half, self.game.state.weather.name), fill='black')
            self.cv.create_text(self.game.arena.width*FFAIEnv.square_size/2.0, 34, text='{}: Score: {}, Turn: {}, RR: {}/{}, Bribes: {}'.format(
                self.game.state.away_team.name,
                self.game.state.away_team.state.score,
                self.game.state.away_team.state.turn,
                self.game.state.away_team.state.rerolls,
                self.game.state.away_team.state.rerolls_start,
                self.game.state.away_team.state.bribes), fill='blue')
            self.cv.create_text(self.game.arena.width * FFAIEnv.square_size / 2.0, 22,
                                text='{}: Score: {}, Turn: {}, RR: {}/{}, Bribes: {}'.format(
                                    self.game.state.home_team.name,
                                    self.game.state.home_team.state.score,
                                    self.game.state.home_team.state.turn,
                                    self.game.state.home_team.state.rerolls,
                                    self.game.state.home_team.state.rerolls_start,
                                    self.game.state.home_team.state.bribes), fill='red')

        # Feature layers
        if feature_layers:
            row = 0
            col = 0
            for name, grid in self.last_obs['board'].items():
                grid_x = col * (len(grid[0]) + 1) * FFAIEnv.square_size_fl + FFAIEnv.square_size_fl
                grid_y = row * (len(grid) + 1) * FFAIEnv.square_size_fl + self.game_height + FFAIEnv.square_size_fl + ((row+1) * FFAIEnv.layer_text_height)

                self.cv.create_text(grid_x + (len(grid[0]) * FFAIEnv.square_size_fl)/2, grid_y - FFAIEnv.layer_text_height/2, text=name)
                self.cv.create_rectangle(grid_x,
                                         grid_y,
                                         grid_x + len(grid[0]) * FFAIEnv.square_size_fl,
                                         grid_y + len(grid) * FFAIEnv.square_size_fl,
                                         fill='black', outline=FFAIEnv.black, width=2)
                for y in range(len(grid)):
                    for x in range(len(grid[0])):
                        value = 1 - grid[y][x]
                        fill = '#%02x%02x%02x' % (int(value * 255), int(value * 255), int(value * 255))
                        self.cv.create_rectangle(FFAIEnv.square_size_fl * x + grid_x,
                                                 FFAIEnv.square_size_fl * y + grid_y,
                                                 FFAIEnv.square_size_fl * x + grid_x + FFAIEnv.square_size_fl,
                                                 FFAIEnv.square_size_fl * y + grid_y + FFAIEnv.square_size_fl,
                                                 fill=fill, outline=FFAIEnv.black)
                col += 1
                if col >= self.cols:
                    col = 0
                    row += 1

        self.root.update_idletasks()
        self.root.update()

    def close(self):
        self.game = None
        self.root = None
        self.cv = None


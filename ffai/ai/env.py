"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains the FFAIEnv class; FFAI implementing the Open AI Gym interface.
"""

import gym
from gym import spaces
from ffai.core import Game, load_rule_set, load_arena
from ffai.ai.bots.random_bot import RandomBot
from ffai.ai.layers import *
import uuid
import tkinter as tk
import math
from copy import deepcopy


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
        ActionType.END_PLAYER_TURN,
        ActionType.USE_REROLL,
        ActionType.DONT_USE_REROLL,
        ActionType.END_TURN,
        ActionType.END_SETUP,
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
        ActionType.SELECT_PLAYER,
        ActionType.MOVE,
        ActionType.BLOCK,
        ActionType.PASS,
        ActionType.FOUL,
        ActionType.HANDOFF,
        ActionType.LEAP,
        ActionType.STAB,
        ActionType.START_MOVE,
        ActionType.START_BLOCK,
        ActionType.START_BLITZ,
        ActionType.START_PASS,
        ActionType.START_FOUL,
        ActionType.START_HANDOFF,
        ActionType.USE_SKILL,
        ActionType.DONT_USE_SKILL,
        ActionType.SETUP_FORMATION_WEDGE,
        ActionType.SETUP_FORMATION_LINE,
        ActionType.SETUP_FORMATION_SPREAD,
        ActionType.SETUP_FORMATION_ZONE,
        ActionType.USE_BRIBE,
        ActionType.DONT_USE_BRIBE
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
        ActionType.USE_SKILL,
        ActionType.DONT_USE_SKILL,
        ActionType.END_TURN,
        ActionType.STAND_UP,
        ActionType.SELECT_ATTACKER_DOWN,
        ActionType.SELECT_BOTH_DOWN,
        ActionType.SELECT_PUSH,
        ActionType.SELECT_DEFENDER_STUMBLES,
        ActionType.SELECT_DEFENDER_DOWN,
        ActionType.SELECT_NONE,
        ActionType.USE_BRIBE,
        ActionType.DONT_USE_BRIBE
    ]

    defensive_formation_action_types = [
        ActionType.SETUP_FORMATION_SPREAD,
        ActionType.SETUP_FORMATION_ZONE
    ]

    offensive_formation_action_types = [
        ActionType.SETUP_FORMATION_WEDGE,
        ActionType.SETUP_FORMATION_LINE
    ]

    positional_action_types = [
        ActionType.PLACE_BALL,
        ActionType.PUSH,
        ActionType.FOLLOW_UP,
        ActionType.MOVE,
        ActionType.BLOCK,
        ActionType.PASS,
        ActionType.FOUL,
        ActionType.HANDOFF,
        ActionType.LEAP,
        ActionType.STAB,
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
        MoveAction,
        BlockAction,
        BlitzAction,
        PassAction,
        HandoffAction,
        FoulAction,
        ThrowBombAction,
        Block,
        Push,
        FollowUp,
        Apothecary,
        PassAttempt,
        Interception,
        Reroll,
        Ejection
    ]

    def __init__(self, config, home_team, away_team, opp_actor=None):
        self.__version__ = "0.0.3"
        self.config = config
        self.config.competition_mode = False
        self.config.fast_mode = True
        self.game = None
        self.team_id = None
        self.ruleset = load_rule_set(config.ruleset, all_rules=False)
        self.home_team = home_team
        self.away_team = away_team
        self.actor = Agent("Gym Learner", human=True)
        self.opp_actor = opp_actor if opp_actor is not None else RandomBot("Random")
        self._seed = None
        self.seed()
        self.root = None
        self.cv = None
        self.last_obs = None
        self.last_report_idx = 0
        self.last_ball_team = None
        self.last_ball_x = None
        self.own_team = None
        self.opp_team = None

        self.layers = [
            OccupiedLayer(),
            OwnPlayerLayer(),
            OppPlayerLayer(),
            OwnTackleZoneLayer(),
            OppTackleZoneLayer(),
            UpLayer(),
            StunnedLayer(),
            UsedLayer(),
            RollProbabilityLayer(),
            BlockDiceLayer(),
            ActivePlayerLayer(),
            TargetPlayerLayer(),
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
            SkillLayer(Skill.CATCH),
            SkillLayer(Skill.PASS)
        ]

        for action_type in FFAIEnv.positional_action_types:
            self.layers.append(AvailablePositionLayer(action_type))

        arena = load_arena(self.config.arena)

        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=1, shape=(len(self.layers), arena.height, arena.width)),
            'state': spaces.Box(low=0, high=1, shape=(50,)),
            'procedures': spaces.Box(low=0, high=1, shape=(len(FFAIEnv.procedures),)),
            'available-action-types': spaces.Box(low=0, high=1, shape=(len(FFAIEnv.actions),))
        })

        self.action_space = spaces.Dict({
            'action-type': spaces.Discrete(len(FFAIEnv.actions)),
            'x': spaces.Discrete(arena.width),
            'y': spaces.Discrete(arena.height)
        })

    def step(self, action):
        if type(action['action-type']) is ActionType and action['action-type'] in FFAIEnv.actions:
            action_type = action['action-type']
        else:
            action_type = FFAIEnv.actions[int(action['action-type'])]
        p = Square(action['x'], action['y']) if action['x'] is not None and action['y'] is not None else None
        position = None
        player = None
        if action_type in self.positional_action_types:
            position = p
        real_action = Action(action_type=action_type, position=position, player=player)
        self.last_report_idx = len(self.game.state.reports)
        return self._step(real_action)

    def _step(self, action):
        self.game.step(action)
        if action.action_type in FFAIEnv.offensive_formation_action_types or action.action_type in FFAIEnv.defensive_formation_action_types:
            self.game.step(Action(ActionType.END_SETUP))
        reward = 0
        if self.game.get_winner() is not None:
            reward = 1 if self.game.get_winner() == self.actor else -1
        team = self.game.state.home_team if self.team_id == self.home_team.team_id else self.game.state.away_team
        opp_team = self.game.state.home_team if self.team_id != self.home_team.team_id else self.game.state.away_team
        ball_carrier = self.game.get_ball_carrier()
        ball_team = ball_carrier.team if ball_carrier is not None else None
        ball_position = self.game.get_ball_position()
        progression = 0
        if ball_team == team and self.last_ball_team:
            #print("From: ", self.last_ball_x, ", To: ", ball_position.x)
            if team == self.game.state.home_team:
                progression = self.last_ball_x - ball_position.x
            else:
                progression = ball_position.x - self.last_ball_x
            #print("Progression: ", progression)
        self.last_ball_x = ball_position.x if ball_position is not None else None
        self.last_ball_team = ball_team
        info = {
            'cas_inflicted': len(self.game.get_casualties(team)),
            'opp_cas_inflicted': len(self.game.get_casualties(opp_team)),
            'touchdowns': team.state.score,
            'opp_touchdowns': opp_team.state.score,
            'half': self.game.state.round,
            'round': self.game.state.round,
            'ball_progression': progression
        }
        return self._observation(self.game), reward, self.game.state.game_over, info

    def seed(self, seed=None):
        if seed is None:
            self._seed = np.random.randint(0, 2**31)
        self.rnd = np.random.RandomState(self._seed)
        if isinstance(self.opp_actor, RandomBot):
            self.opp_actor.rnd = self.rnd
        return self._seed

    def get_seed(self):
        return self._seed

    def get_game(self):
        return self.game

    def get_observation(self):
        return self._observation(self.game)

    def _observation(self, game):
        obs = {
            'board': {},
            'state': {},
            'procedures': {},
            'available-action-types': {}
        }
        for layer in self.layers:
            obs['board'][layer.name()] = layer.get(game)

        active_team = game.state.available_actions[0].team if len(game.state.available_actions) > 0 else None
        opp_team = game.get_opp_team(active_team) if active_team is not None else None

        # State
        obs['state']['half'] = game.state.half - 1.0
        obs['state']['round'] = game.state.round / 8.0
        obs['state']['is sweltering heat'] = 1.0 if game.state.weather == WeatherType.SWELTERING_HEAT else 0.0
        obs['state']['is very sunny'] = 1.0 if game.state.weather == WeatherType.VERY_SUNNY else 0.0
        obs['state']['is nice'] = 1.0 if game.state.weather.value == WeatherType.NICE else 0.0
        obs['state']['is pouring rain'] = 1.0 if game.state.weather.value == WeatherType.POURING_RAIN else 0.0
        obs['state']['is blizzard'] = 1.0 if game.state.weather.value == WeatherType.BLIZZARD else 0.0

        obs['state']['is own turn'] = 1.0 if game.state.current_team == active_team else 0.0
        obs['state']['is kicking first half'] = 1.0 if game.state.kicking_first_half == active_team else 0.0
        obs['state']['is kicking this drive'] = 1.0 if game.state.kicking_this_drive == active_team else 0.0
        obs['state']['own reserves'] = len(game.get_reserves(active_team)) / 16.0 if active_team is not None else 0.0
        obs['state']['own kods'] = len(game.get_knocked_out(active_team)) / 16.0 if active_team is not None else 0.0
        obs['state']['own casualites'] = len(game.get_casualties(active_team)) / 16.0 if active_team is not None else 0.0
        obs['state']['opp reserves'] = len(game.get_reserves(game.get_opp_team(active_team))) / 16.0 if active_team is not None else 0.0
        obs['state']['opp kods'] = len(game.get_knocked_out(game.get_opp_team(active_team))) / 16.0 if active_team is not None else 0.0
        obs['state']['opp casualties'] = len(game.get_casualties(game.get_opp_team(active_team))) / 16.0 if active_team is not None else 0.0

        obs['state']['own score'] = active_team.state.score / 16.0 if active_team is not None else 0.0
        obs['state']['own turns'] = active_team.state.turn / 8.0 if active_team is not None else 0.0
        obs['state']['own starting rerolls'] = active_team.state.rerolls_start / 8.0 if active_team is not None else 0.0
        obs['state']['own rerolls left'] = active_team.state.rerolls / 8.0 if active_team is not None else 0.0
        obs['state']['own ass coaches'] = active_team.state.ass_coaches / 8.0 if active_team is not None else 0.0
        obs['state']['own cheerleaders'] = active_team.state.cheerleaders / 8.0 if active_team is not None else 0.0
        obs['state']['own bribes'] = active_team.state.bribes / 4.0 if active_team is not None else 0.0
        obs['state']['own babes'] = active_team.state.babes / 4.0 if active_team is not None else 0.0
        obs['state']['own apothecaries'] = 0.0 if active_team is None else active_team.state.apothecaries / 2
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
        obs['state']['opp apothecaries'] = 0.0 if opp_team is None else active_team.state.apothecaries / 2
        obs['state']['opp reroll available'] = 1.0 if opp_team is not None and not opp_team.state.reroll_used else 0.0
        obs['state']['opp fame'] = opp_team.state.fame if opp_team is not None else 0.0

        obs['state']['is blitz available'] = 1.0 if game.is_blitz_available() else 0.0
        obs['state']['is pass available'] = 1.0 if game.is_pass_available() else 0.0
        obs['state']['is handoff available'] = 1.0 if game.is_handoff_available() else 0.0
        obs['state']['is foul available'] = 1.0 if game.is_foul_available() else 0.0
        obs['state']['is blitz'] = 1.0 if game.is_blitz() else 0.0
        obs['state']['is quick snap'] = 1.0 if game.is_quick_snap() else 0.0

        obs['state']['is move action'] = 1.0 if game.state.active_player is not None and game.get_player_action_type() == PlayerActionType.MOVE else 0.0
        obs['state']['is block action'] = 1.0 if game.state.active_player is not None and game.get_player_action_type() == PlayerActionType.BLOCK else 0.0
        obs['state']['is blitz action'] = 1.0 if game.state.active_player is not None and game.get_player_action_type() == PlayerActionType.BLITZ else 0.0
        obs['state']['is pass action'] = 1.0 if game.state.active_player is not None and game.get_player_action_type() == PlayerActionType.PASS else 0.0
        obs['state']['is handoff action'] = 1.0 if game.state.active_player is not None and game.get_player_action_type() == PlayerActionType.HANDOFF else 0.0
        obs['state']['is foul action'] = 1.0 if game.state.active_player is not None and game.get_player_action_type() == PlayerActionType.FOUL else 0.0

        # Procedure
        for procedure in FFAIEnv.procedures:
            proc_name = procedure.__name__
            obs['procedures'][proc_name] = 0.0

        for procedure in game.state.stack.items:
            if procedure.__class__ in FFAIEnv.procedures:
            # proc_idx = FFAIEnv.procedures.index(procedure.__class__)
                proc_name = procedure.__class__.__name__
                obs['procedures'][proc_name] = 1.0

        self.last_obs = obs

        # Actions
        for action_type in FFAIEnv.actions:
            obs['available-action-types'][action_type.name] = 0.0

        for action_choice in game.get_available_actions():
            if action_choice.action_type not in FFAIEnv.actions:
                continue
            # idx = FFAIEnv.actions.index(action_choice.action_type)
            action_name = action_choice.action_type.name
            # Ignore end setup action if setup is illegal
            if action_choice.action_type == ActionType.END_SETUP and not game.is_setup_legal(active_team):
                continue
            obs['available-action-types'][action_name] = 1.0

        return obs

    def reset(self):
        self.team_id = self.home_team.team_id
        home_agent = self.actor
        away_agent = self.opp_actor
        seed = self.rnd.randint(0, 2**31)
        self.game = Game(game_id=str(uuid.uuid1()),
                         home_team=deepcopy(self.home_team),
                         away_team=deepcopy(self.away_team),
                         home_agent=home_agent,
                         away_agent=away_agent,
                         config=self.config,
                         ruleset=self.ruleset,
                         seed=seed)
        self.last_report_idx = len(self.game.state.reports)
        self.last_ball_team = None
        self.last_ball_x = None
        self.game.init()
        self.own_team = self.game.get_agent_team(self.actor)
        self.opp_team = self.game.get_agent_team(self.opp_actor)

        return self._observation(self.game)

    def get_outcomes(self):
        if self.last_report_idx == len(self.game.state.reports):
            return []
        return self.game.state.reports[self.last_report_idx:]

    def available_action_types(self):
        if isinstance(self.game.get_procedure(), Setup):
            if self.game.get_kicking_team().team_id == self.team_id:
                return [FFAIEnv.actions.index(action_type) for action_type in self.defensive_formation_action_types]
            else:
                return [FFAIEnv.actions.index(action_type) for action_type in self.offensive_formation_action_types]
        return [action.action_type for action in self.game.state.available_actions if action.action_type in FFAIEnv.actions]

    def available_positions(self, action_type):
        action = None
        for a in self.game.state.available_actions:
            if a.action_type == action_type:
                action = a
        if action is None:
            return []
        if action.action_type in FFAIEnv.positional_action_types:
            if action.positions:
                return [position for position in action.positions if position is not None]
            elif action.players:
                return [player.position for player in action.players if player is not None]
        return []

    def _available_players(self, action_type):
        action = None
        for a in self.game.state.available_actions:
            if a.action_type == FFAIEnv.actions[action_type]:
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
                self.cv = tk.Canvas(width=max(self.game_width, self.fl_width), height=self.fl_height + self.game_height, master=self.root)
            else:
                self.cv = tk.Canvas(width=self.game_width, height=self.game_height, master=self.root)

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
            for player in self.game.get_knocked_out(self.game.state.away_team):
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
            for player in self.game.get_knocked_out(self.game.state.home_team):
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


"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains the Game class, which is the main class and interface used to interact with a game in FFAI.
"""

from ffai.core.procedure import *
from ffai.core.load import *
from copy import deepcopy
import numpy as np


class Game:

    def __init__(self, game_id, home_team, away_team, home_agent, away_agent, config=None, arena=None, ruleset=None, state=None, seed=None, record=False):
        assert config is not None or arena is not None
        assert config is not None or ruleset is not None
        assert home_team.team_id != away_team.team_id
        self.replay = Replay(replay_id=game_id) if record else None
        self.game_id = game_id
        self.home_agent = home_agent
        self.away_agent = away_agent
        self.actor = None
        self.arena = load_arena(config.arena) if arena is None else arena
        self.config = config
        self.ruleset = load_rule_set(config.ruleset) if ruleset is None else ruleset
        self.state = state if state is not None else GameState(self, deepcopy(home_team), deepcopy(away_team))
        self.rnd = np.random.RandomState(seed)

        self.start_time = None
        self.end_time = None
        self.disqualified_agent = None
        self.last_request_time = None
        self.last_action_time = None
        self.forced_action = None

        self.action = None

    def to_json(self):
        return {
            'game_id': self.game_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'state': self.state.to_json(),
            'stack': self.get_procedure_names(),
            'home_agent': self.home_agent.to_json(),
            'away_agent': self.away_agent.to_json(),
            'squares_moved': self._squares_moved(),
            'arena': self.arena.to_json(),
            'ruleset': self.ruleset.name,
            'can_home_team_use_reroll': self.can_use_reroll(self.state.home_team),
            'can_away_team_use_reroll': self.can_use_reroll(self.state.away_team),
            'actor_id': self.actor.agent_id if self.actor is not None else None,
            'disqualified_agent_id': self.disqualified_agent.agent_id if self.disqualified_agent is not None else None,
            'time_limits': self.config.time_limits.to_json(),
            'active_other_player_id': self.get_other_active_player_id()
        }

    def safe_clone(self):
        # Make dummy agents for the clone
        home_agent = self.home_agent
        away_agent = self.away_agent
        self.home_agent = Agent(home_agent.name, agent_id=home_agent.agent_id)
        self.away_agent = Agent(away_agent.name, agent_id=away_agent.agent_id)
        clone = deepcopy(self)
        self.home_agent = home_agent
        self.away_agent = away_agent
        return clone

    def init(self):
        """
        Initialized the Game. The START_GAME action must still be called after this if humans are in the game.
        """
        EndGame(self)
        Pregame(self)
        if not self.away_agent.human:
            game_copy_away = self.safe_clone()
            self.actor = self.away_agent
            self.away_agent.new_game(game_copy_away, game_copy_away.state.away_team)
            self.actor = None
        if not self.home_agent.human:
            self.actor = self.home_agent
            game_copy_home = self.safe_clone()
            self.actor = None
            self.home_agent.new_game(game_copy_home, game_copy_home.state.home_team)

        self.set_available_actions()
        # Record state
        if self.replay is not None:
            self.replay.record_step(self)
        # Start game if no humans
        if not self.away_agent.human and not self.home_agent.human:
            start_action = Action(ActionType.START_GAME)
            # Record state
            if self.replay is not None:
                self.replay.record_action(start_action)
            self.step(start_action)

    def step(self, action=None):
        """
        Runs until an action from a human is required. If game requires an action to continue one must be given.
        :param action: Action to perform, can be None if game does not require any.
        :return:
        """

        # Ensure player points to player object
        if action is not None:
            action.player = self.get_player(action.player.player_id) if action.player is not None else None
            action.position = Square(action.position.x, action.position.y) if action.position is not None else None

        # Set action as a property so other methods can access it
        self.action = action

        # Update game
        while True:

            # Record state
            if self.replay is not None:
                self.replay.record_action(self.action)

            # Perform game step
            done = self._one_step(self.action)

            # Record state
            if self.replay is not None:
                self.replay.record_step(self)

            # Game over
            if self.state.game_over:
                self._end_game()
                return
            
            # if procedure is ready for input
            if done:

                # If human player - wait for input
                if self.actor.human:
                    return

                # Query agent for action
                self.last_request_time = time.time()
                self.action = self._safe_act()
                # self.action = self._safe_act()
                
                # Check if time limit was violated
                self.last_action_time = time.time()
                self.check_clocks()  # Might modify the action if clock was violated

                # Did the game terminate?
                if self.state.game_over:
                    return
                
            else:

                # If not in fast mode - wait for input before continuing
                if not self.config.fast_mode:
                    return
                
                # Else continue procedure with no action
                self.action = None

    def refresh(self):
        """
        Checks clocks and runs forced actions. Can be called in human games.
        """
        self.action = None
        self.check_clocks()
        if self.action is not None:
            self.step(self.action)

    def check_clocks(self):
        """
        Checks if clocks are done.
        """
        # Only use clocks in competition mode
        if not self.config.competition_mode:
            return

        # Timed out?
        if self.is_timed_out():
            print("Game timed out")
            self.action = None
            self.remove_clocks()
            self.disqualified_agent = self.actor
            self.report(Outcome(OutcomeType.END_OF_GAME_DISQUALIFICATION, team=self.get_agent_team(self.actor)))
            self._end_game()
            return

        # No time limit for this action
        if not self.has_agent_clock(self.actor):
            return

        # Agent too slow?
        clock = self.get_agent_clock(self.actor)
        if clock.is_done():
            
            # Disqualification? Relevant for hanging bots in competitions
            if clock.get_running_time() > clock.seconds + self.config.time_limits.disqualification:
                print(f"Time violation. {self.actor.name} will be disqualified!")
                self.action = None
                self.remove_clocks()
                self.disqualified_agent = self.actor
                self.report(Outcome(OutcomeType.END_OF_GAME_DISQUALIFICATION, team=self.get_agent_team(self.actor)))
                self._end_game()
                return

            # End the actor's turn
            done = True
            actor = self.actor
            clock = self.get_agent_clock(actor)
            while clock in self.state.clocks:
                
                # Request timout action
                if done:
                    action = self._forced_action()
                else:
                    action = None

                # Take action if it doesn't end the turn
                if self.action is None or self.action.action_type not in [ActionType.END_TURN, ActionType.END_SETUP]:
                    if self.config.debug_mode:
                        print("Forced step action")
                    done = self._one_step(action)
                else:
                    if self.config.debug_mode:
                        print(f"Forcing action: {self.action.to_json() if self.action is not None else 'None'}")
                    self.action = action
                    break
            
    def _end_game(self):
        '''
        End the game
        '''
        # Game ended when the last action was received - to avoid timout during finishing procedures
        self.end_time = self.last_action_time
        self.state.game_over = True

        # Let agents know that the game ended
        now = None
        if not self.home_agent.human:
            self.actor = self.home_agent  # In case it crashes or timeouts we have someone to blame
            if self.config.competition_mode:
                self.actor = self.home_agent
                clone = self.safe_clone()
                now = time.time()
                self.home_agent.end_game(clone)
                self.actor = None
            else:
                self.home_agent.end_game(self)
            # Disqualify if too long
            if self.config.competition_mode and time.time() > now + self.config.time_limits.end:
                print("End time violated by " + self.home_agent.name + ". Time used: " + time.time() - now)
                self.disqualified_agent = self.home_agent
        if not self.away_agent.human:
            self.actor = self.away_agent  # In case it crashes or timeouts we have someone to blame
            if self.config.competition_mode:
                self.actor = self.away_agent
                clone = self.safe_clone()
                now = time.time()
                self.away_agent.end_game(clone)
                self.actor = None
            else:
                self.away_agent.end_game(self)
            # Disqualify if too long
            if self.config.competition_mode and time.time() > now + self.config.time_limits.end:
                print("End time violated by " + self.away_agent.name + ". Time used: " + time.time() - now)
                self.disqualified_agent = self.away_agent

        # Record state
        if self.replay is not None:
            self.replay.record_step(self)
            self.replay.dump(self.game_id)

        # Record state
        if self.replay is not None:
            self.replay.record_step(self)
            self.replay.dump(self.game_id)

    def _is_action_allowed(self, action):
        """
        Checks whether the specified action is allowed by comparing to actions in self.state.available_actions.
        :param action:
        :return: True if the specified actions is allowed.
        """
        if action is None:
            return True
        for action_choice in self.state.available_actions:
            if action.action_type == action_choice.action_type:
                if type(action.action_type) != ActionType:
                    print("Illegal action type")
                    return False
                if action.player is not None and not isinstance(action.player, Player):
                    print("Illegal player type", action.action_type, action, self.state.stack.peek())
                    return False
                if action.position is not None and not isinstance(action.position, Square):
                    print("Illegal position type", action.position, action.action_type.name)
                    return False
                if len(action_choice.players) > 0 and action.player not in action_choice.players:
                    if action.player is None:
                        print("Player player: None")
                    else:
                        print("Illegal player", action.action_type, action.player, self.state.stack.peek())
                    return False
                if len(action_choice.positions) > 0 and action.position not in action_choice.positions:
                    if action.position is None:
                        print("Player position: None")
                    else:
                        print("Illegal position", action.position.to_json(), action.action_type.name)
                    return False
                return True
        return False

    def _safe_act(self):
        '''
        Clone the game before requesting an action so agent can't manipulate it.
        '''
        if self.config.competition_mode:
            action = self.actor.act(self.safe_clone())
            # Correct player object
            if action.player is not None:
                action.player = self.state.player_by_id[action.player.player_id]
            return action
        return self.actor.act(self)

    def _forced_action(self):
        '''
        Return action that prioritize to end the player's turn.
        '''
        # Take first negative action
        for action_type in [ActionType.END_TURN, ActionType.END_SETUP, ActionType.END_PLAYER_TURN, ActionType.SELECT_NONE, ActionType.HEADS, ActionType.KICK, ActionType.SELECT_DEFENDER_DOWN, ActionType.SELECT_DEFENDER_STUMBLES, ActionType.SELECT_ATTACKER_DOWN, ActionType.SELECT_PUSH, ActionType.SELECT_BOTH_DOWN, ActionType.DONT_USE_REROLL, ActionType.DONT_USE_APOTHECARY]:
            for action in self.state.available_actions:
                if action_type == ActionType.END_SETUP and not self.is_setup_legal(self.get_agent_team(self.actor)):
                    continue
                if action.action_type == action_type:
                    return Action(action_type)
        # Take random action
        action_choice = self.rnd.choice(self.state.available_actions)
        position = self.rnd.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
        player = self.rnd.choice(action_choice.players) if len(action_choice.players) > 0 else None
        return Action(action_choice.action_type, position=position, player=player)
    
    def _squares_moved(self):
        """
        :return: The squares moved by the active player in json - used by the web app.
        """
        '''
        for square in self.state.active_player.state.squares_moved:
            out.append(square)
        if proc.player is not None and proc.player.position is not None:
            if len(out) > 0 and out[-1] != proc.player.position:
                out = out[:-1]
        '''
        if self.state.active_player is None:
            return []
        out = [sq.to_json() for sq in self.state.active_player.state.squares_moved]
        return out
        #return []

    def _one_step(self, action):
        """
        Executes one step in the game if it is allowed.
        :param action: Action from agent. Can be None if no action is required.
        :return: True if game requires action or game is over, False if not
        """

        # Clear done procs
        while not self.state.stack.is_empty() and self.state.stack.peek().done:
            self.state.stack.pop()

        # Is game over
        if self.state.stack.is_empty():
            return False

        # Get proc
        proc = self.state.stack.peek()

        # If no action and action is required
        if action is None and len(self.state.available_actions) > 0:
            if self.config.debug_mode:
                print("None action is not allowed when actions are available")
            return True  # Game needs user input

        # If action but it's not available
        if action is not None:
            if action.action_type == ActionType.CONTINUE:
                if len(self.state.available_actions) == 0:
                    # Consider this as a None action
                    action = None
                else:
                    if self.config.debug_mode:
                        print("CONTINUE action is not allowed when actions are available")
                    return True  # Game needs user input
            else:
                # Only allowed actions
                if not self._is_action_allowed(action):
                    if self.config.debug_mode:
                        print(f"Action not allowed {action.to_json() if action is not None else 'None'}")
                    return True  # Game needs user input

        # Run proc
        if self.config.debug_mode:
            print("Proc={}".format(proc))
            print("Action={}".format(action.action_type if action is not None else "None"))

        proc.done = proc.step(action)

        if self.config.debug_mode:
            print("Done={}".format(proc.done))
            print(f"DONE={self.state.stack.peek().done}")

        # Used if players was accidently cloned
        if self.config.debug_mode:
            for y in range(len(self.state.pitch.board)):
                for x in range(len(self.state.pitch.board)):
                    assert self.state.pitch.board[y][x] is None or \
                        (self.state.pitch.board[y][x].position.x == x and self.state.pitch.board[y][x].position.y == y)

            for team in self.state.teams:
                for player in team.players:
                    if not (player.position is None or self.state.pitch.board[player.position.y][player.position.x] == player):
                        raise Exception("Player position violation")

        # Remove all finished procs
        while not self.state.stack.is_empty() and self.state.stack.peek().done:
            if self.config.debug_mode:
                print("--Proc={}".format(self.state.stack.peek()))
            self.state.stack.pop()

        # Is game over
        if self.state.stack.is_empty():
            return False  # Can continue without user input

        if self.config.debug_mode:
            print("-Proc={}".format(self.state.stack.peek()))

        # Initialize if not
        if not self.state.stack.peek().initialized:
            proc = self.state.stack.peek()
            proc.setup()
            proc.initialized = True

        # Update available actions
        self.set_available_actions()

        if self.config.debug_mode:
            print(f"{len(self.state.available_actions)} available actions")

        if len(self.state.available_actions) == 0:
            return False  # Can continue without user input

        # End player turn if only action available
        if len(self.state.available_actions) == 1 and self.state.available_actions[0].action_type == ActionType.END_PLAYER_TURN:
            self._one_step(Action(ActionType.END_PLAYER_TURN))
            return False  # We can continue without user input

        return True  # Game needs user input
    
    def remove_clocks(self):
        '''
        Remove all clocks.
        '''
        self.state.clocks.clear()

    def remove_secondary_clocks(self):
        '''
        Remove all secondary clocks and resume the primary clock - if any.
        '''
        self.state.clocks = [clock for clock in self.state.clocks if clock.is_primary]
        for clock in self.state.clocks:
            if not clock.is_running():
                clock.resume()

    def get_clock(self, team):
        '''
        Returns the clock belonging to the given team.
        '''
        for clock in self.state.clocks:
            if clock.team == team:
                return clock
        return None

    def get_agent_clock(self, agent):
        '''
        Returns the clock belonging to the given agent's team.
        '''
        for clock in self.state.clocks:
            if clock.team == self.get_agent_team(agent):
                return clock
        return None

    def has_clock(self, team):
        '''
        Returns true if the given team has a clock.
        '''
        for clock in self.state.clocks:
            if clock.team == team:
                return True
        return False

    def has_agent_clock(self, agent):
        '''
        Returns true if the given agent's team has a clock.
        '''
        for clock in self.state.clocks:
            if clock.team == self.get_agent_team(agent):
                return True
        return False

    def pause_clocks(self):
        '''
        Pauses all clocks.
        '''
        for clock in self.state.clocks:
            if clock.is_running():
                clock.pause()

    def add_secondary_clock(self, team):
        '''
        Adds a secondary clock for quick decisions.
        '''
        self.pause_clocks()
        assert team is not None and type(team) == Team
        clock = Clock(team, self.config.time_limits.secondary)
        self.state.clocks.append(clock)

    def add_primary_clock(self, team):
        '''
        Adds a primary clock that will be paused if secondary clocks are added.
        '''
        self.state.clocks.clear()
        assert team is not None and type(team) == Team
        clock = Clock(team, self.config.time_limits.turn, is_primary=True)
        self.state.clocks.append(clock)

    def get_seconds_left(self, team):
        '''
        Returns the number of seconds left on the clock for the given team and None if the given team has no clock.
        '''
        for clock in self.state.clocks:
            if clock.team == team:
                return clock.get_seconds_left()
        return None
        
    def get_team_agent(self, team):
        """
        :param team:
        :return: The agent who's controlling the specified team.
        """
        if team is None:
            return None
        if team == self.state.home_team:
            return self.home_agent
        return self.away_agent

    def get_agent_team(self, agent):
        """
        :param team:
        :return: The team controlled by the specified agent.
        """
        if agent is None:
            return None
        if agent == self.home_agent:
            return self.state.home_team
        return self.state.away_team

    def set_seed(self, seed):
        '''
        Sets the random seed of the game.
        '''
        self.seed = seed
        self.rnd = np.random.RandomState(self.seed)

    def set_available_actions(self):
        """
        Calls the current procedure's available_actions() method and sets the game's available actions to the returned
        list.
        """
        self.state.available_actions = self.state.stack.peek().available_actions()
        self.actor = None
        if len(self.state.available_actions) > 0:
            if self.state.available_actions[0].team == self.state.home_team:
                self.actor = self.home_agent
            elif self.state.available_actions[0].team == self.state.away_team:
                self.actor = self.away_agent
    
    def report(self, outcome):
        """
        Adds the outcome to the game's reports.
        """
        self.state.reports.append(outcome)

    def is_team_side(self, position, team):
        """
        :param position:
        :param team:
        :return: Returns True if pos is on team's side of the arena.
        """
        if team == self.state.home_team:
            return self.arena.board[position.y][position.x] in TwoPlayerArena.home_tiles
        return self.arena.board[position.y][position.x] in TwoPlayerArena.away_tiles

    def get_team_side(self, team):
        """
        :param team:
        :return: a list of squares on team's side of the arena.
        """
        tiles = []
        for y in range(len(self.arena.board)):
            for x in range(len(self.arena.board[y])):
                if self.arena.board[y][x] in (TwoPlayerArena.home_tiles if team == self.state.home_team else TwoPlayerArena.away_tiles):
                    tiles.append(Square(x, y))
        return tiles

    def is_scrimmage(self, position):
        """
        :param position:
        :return: Returns True if pos is on the scrimmage line.
        """
        return self.arena.board[position.y][position.x] in TwoPlayerArena.scrimmage_tiles

    def is_wing(self, position, right):
        """
        :param position:
        :param right: Whether to check on the right side of the arena. If False, it will check on the left side.
        :return: True if pos is on the arena's wing and on the specified side.
        """
        if right:
            return self.arena.board[position.y][position.x] in TwoPlayerArena.wing_right_tiles
        return self.arena.board[position.y][position.x] in TwoPlayerArena.wing_left_tiles

    def remove_balls(self):
        """
        Removes all balls from the arena.
        """
        self.state.pitch.balls.clear()

    def is_last_turn(self):
        """
        :return: True if this turn is the last turn of the game.
        """
        return self.get_next_team().state.turn == self.config.rounds and self.state.half == 2

    def is_last_round(self):
        """
        :return: True if this round is the las round of the game.
        """
        return self.state.round == self.config.rounds

    def get_next_team(self):
        """
        :return: The team who's turn it is next.
        """
        idx = self.state.turn_order.index(self.state.current_team)
        if idx+1 == len(self.state.turn_order):
            return self.state.turn_order[0]
        return self.state.turn_order[idx+1]

    def add_or_skip_turn(self, turns):
        """
        Adds or removes a number of turns from the current half. This method will raise an assertion error if the turn
        counter goes to a negative number.
        :param turns: The number of turns to add (if positive) or remove (if negative). 
        """
        for team in self.state.teams:
            team.state.turn += turns
            assert team.state.turn >= 0

    def get_player(self, player_id):
        """
        :param player_id: 
        :return: Returns the player with player_id
        """
        return self.state.player_by_id[player_id]

    def get_player_at(self, position):
        """
        :param position: 
        :return: Returns the player at pos else None.
        """
        return self.state.pitch.board[position.y][position.x]

    def set_turn_order_from(self, first_team):
        """
        Sets the turn order starting from first_team.
        :param first_team: The first team to start.
        """
        before = []
        after = []
        added = False
        if len(self.state.turn_order) == 0:
            self.state.turn_order = [team for team in self.state.teams]
        for team in self.get_turn_order():
            if team == first_team:
                added = True
            if not added:
                before.append(team)
            else:
                after.append(team)
        self.state.turn_order = after + before

    def set_turn_order_after(self, last_team):
        """
        Sets the turn order starting after last_team.
        :param last_team: The last team to start.
        """
        before = []
        after = []
        added = False
        if len(self.state.turn_order) == 0:
            self.state.turn_order = [team for team in self.state.teams]
        for team in self.get_turn_order():
            if not added:
                before.append(team)
            else:
                after.append(team)
            if team == last_team:
                added = True
        self.state.turn_order = after + before

    def get_turn_order(self):
        """
        :return: The list of teams sorted by turn order.
        """
        return self.state.turn_order

    def is_home_team(self, team):
        """
        :return: True if team is the home team.
        """
        return team == self.state.home_team

    def get_opp_team(self, team):
        """
        :param team: 
        :return: The opponent team of team.
        """
        return self.state.home_team if self.state.away_team == team else self.state.away_team

    def get_dugout(self, team):
        return self.state.dugouts[team.team_id]

    def get_reserves(self, team):
        """
        :param team: 
        :return: The reserves in the dugout of this team.
        """
        return self.get_dugout(team).reserves

    def get_knocked_out(self, team):
        """
        :param team:
        :return: The knocked out players in the dugout of this team.
        """
        return self.get_dugout(team).kod

    def get_casualties(self, team):
        """
        :param team:
        :return: The badly hurt, injured, and dead players in th dugout of this team.
        """
        return self.get_dugout(team).casualties

    def get_dungeon(self, team):
        """
        :param team:
        :return: The ejected players of this team, who's locked to a cell in the dungeon.
        """
        return self.get_dugout(team).dungeon

    def current_turn(self):
        """
        :return: The top-most Turn procedure in the stack.
        """
        for i in reversed(range(self.state.stack.size())):
            proc = self.state.stack.items[i]
            if isinstance(proc, Turn):
                return proc
        return None

    def can_use_reroll(self, team):
        """
        :param team:
        :return: True if the team can use reroll right now (i.e. this turn).
        """
        if not team.state.reroll_used and team.state.rerolls > 0 and self.state.current_team == team:
            current_turn = self.current_turn()
            if current_turn is not None and isinstance(current_turn, Turn):
                return not current_turn.blitz and not current_turn.quick_snap
        return False

    def get_kicking_team(self, half=None):
        """
        :param half: Set this to None if you want the team who's kicking this drive.
        :return: The team who's kicking in the specified half. If half is None, the team who's kicking this drive.
        """
        if half is None:
            return self.state.kicking_this_drive
        return self.state.kicking_first_half if half == 1 else self.state.receiving_first_half

    def get_receiving_team(self, half=None):
        """
        :param half: Set this to None if you want the team who's receiving this drive.
        :return: The team who's receiving in the specified half. If half is None, the team who's receiving this drive.
        """
        if half is None:
            return self.state.receiving_this_drive
        return self.state.receiving_first_half if half == 1 else self.state.kicking_first_half

    def has_ball(self, player):
        """
        :param player:
        :return: True if player has the ball.
        """
        ball = self.get_ball_at(player.position)
        return True if ball is not None and ball.is_carried else False

    def get_ball(self):
        """
        :return: A ball on the pitch or None.
        """
        for ball in self.state.pitch.balls:
            return ball.position

    def is_touchdown(self, player):
        """
        :param player:
        :return: True if player is in the opponent's endzone with the ball.
        """
        return self.arena.is_in_opp_endzone(player.position, player.team == self.state.home_team)

    def is_blitz_available(self):
        """
        :return: True if the current team can make a blitz this turn.
        """
        turn = self.current_turn()
        if turn is not None:
            return turn.blitz_available

    def is_pass_available(self):
        """
        :return: True if the current team can make a pass this turn.
        """
        turn = self.current_turn()
        if turn is not None:
            return turn.pass_available

    def is_handoff_available(self):
        """
        :return: True if the current team can make a handoff this turn.
        """
        turn = self.current_turn()
        if turn is not None:
            return turn.handoff_available

    def is_foul_available(self):
        """
        :return: True if the current team can make a foul this turn.
        """
        turn = self.current_turn()
        if turn is not None:
            return turn.foul_available

    def is_blitz(self):
        """
        :return: True if the current turn is a Blitz!
        """
        turn = self.current_turn()
        if turn is not None:
            return turn.blitz

    def is_quick_snap(self):
        """
        :return: True if the current turn is a Quick Snap!
        """
        turn = self.current_turn()
        if turn is not None:
            return turn.quick_snap

    def get_players_on_pitch(self, team, used=None, up=None):
        """
        :param team: The team of the players.
        :param used: If specified, filter by ther players used state.
        :param up: If specified, filter by ther players up state.
        :return: Players on the pitch who's on team.
        """
        players = []
        for y in range(len(self.state.pitch.board)):
            for x in range(len(self.state.pitch.board[y])):
                player = self.state.pitch.board[y][x]
                if player is not None and player.team == team and (used is None or used == player.state.used) and (up is None or up == player.state.up):
                    players.append(player)
        return players

    def pitch_to_reserves(self, player):
        """
        Moves player from the pitch to the reserves section in the dugout.
        :param player:
        """
        self.remove(player)
        self.get_reserves(player.team).append(player)
        player.state.used = False
        player.state.up = True

    def reserves_to_pitch(self, player, position):
        """
        Moves player from the reserves section in the dugout to the pitch.
        :param player:
        """
        self.get_reserves(player.team).remove(player)
        player_at = self.get_player_at(position)
        if player_at is not None:
            self.pitch_to_reserves(player_at)
        self.put(player, position)
        player.state.up = True

    def pitch_to_kod(self, player):
        """
        Moves player from the pitch to the KO section in the dugout.
        :param player:
        """
        self.remove(player)
        self.get_knocked_out(player.team).append(player)
        player.state.knocked_out = True
        player.state.up = True

    def kod_to_reserves(self, player):
        """
        Moves player from the KO section in the dugout to the pitch. This also resets the players knocked_out state.
        :param player:
        """
        self.get_knocked_out(player.team).remove(player)
        self.get_reserves(player.team).append(player)
        player.state.knocked_out = False
        player.state.up = True

    def pitch_to_casualties(self, player, casualty, effect, apothecary=False):
        """
        Moves player from the pitch to the CAS section in the dugout and applies the casualty and effect to the player.
        :param player:
        :param casualty:
        :param effect:
        :param apothecary: If True and effect == CasualtyEffect.NONE, player is moved to the reserves.
        :return:
        """
        self.remove(player)
        player.state.up = True
        if apothecary and effect == CasualtyEffect.NONE:
            # Apothecary puts badly hurt players in the reserves
            self.get_reserves(player.team).append(player)
        else:
            player.state.casualty = casualty
            player.state.casualty_effect = effect
            self.get_casualties(player.team).append(player)

    def pitch_to_dungeon(self, player):
        """
        Moves player from the pitch to the dungeon and ejects the player from the game.
        :param player:
        """
        self.remove(player)
        self.get_dungeon(player.team).append(player)
        player.state.ejected = True
        player.state.up = True

    def put(self, piece, position):
        """
        Put a piece on the board.
        :param piece: Ball or player
        :param position:
        """
        piece.position = Square(position.x, position.y)
        self.state.pitch.board[position.y][position.x] = piece

    def remove(self, piece):
        """
        Remove a piece from the board.
        :param piece:
        """
        assert piece.position is not None
        self.state.pitch.board[piece.position.y][piece.position.x] = None
        piece.position = None

    def move(self, piece, position):
        """
        Move a piece already on the board on the board. If the piece is a ball carrier, the ball is moved as well.
        :param piece:
        :param position:
        :return:
        """
        assert piece.position is not None
        assert self.state.pitch.board[position.y][position.x] is None
        for ball in self.state.pitch.balls:
            if ball.position == piece.position and ball.is_carried:
                ball.move_to(position)
        self.remove(piece)
        self.put(piece, position)

    def swap(self, piece_a, piece_b):
        """
        Swap two pieces on the board.
        :param piece_a:
        :param piece_b:
        :return:
        """
        assert piece_a.position is not None
        assert piece_b.position is not None
        pos_a = Square(piece_a.position.x, piece_a.position.y)
        pos_b = Square(piece_b.position.x, piece_b.position.y)
        piece_a.position = pos_b
        piece_b.position = pos_a
        self.state.pitch.board[pos_a.y][pos_a.x] = piece_b
        self.state.pitch.board[pos_b.y][pos_b.x] = piece_a

    def get_catch_modifiers(game, player, accurate=False, interception=False, handoff=False):
        """
        :param catcher:
        :param accurate: whether it is an accurate pass.
        :param interception: whether it is an interception catch.
        :param interception: whether it is a handoff catch.
        :return: the modifier to be added to the pass roll.
        """
        modifiers = 1 if accurate or handoff else 0
        modifiers = -2 if interception else modifiers
        if interception and player.has_skill(Skill.LONG_LEGS):
            modifiers += 1
        # opposing tackle zones
        if not player.has_skill(Skill.NERVES_OF_STEEL):
            modifiers -= game.num_tackle_zones_in(player)
        if game.state.weather == WeatherType.POURING_RAIN:
            modifiers -= 1
        if player.has_skill(Skill.EXTRA_ARMS):
            modifiers += 1
        return modifiers

    def get_pass_modifiers(self, passer, pass_distance):
        """
        :param passer:
        :param pass_distance: the PassDistance to the target.
        :return: the modifier to be added to the pass roll.
        """
        modifiers = Rules.pass_modifiers[pass_distance]
        # Opposing tackle zones
        tackle_zones = self.num_tackle_zones_in(passer)
        if not passer.has_skill(Skill.NERVES_OF_STEEL):
            modifiers -= tackle_zones

        # Weather
        if self.state.weather == WeatherType.VERY_SUNNY:
            modifiers -= 1

        if passer.has_skill(Skill.ACCURATE):
            modifiers += 1

        if passer.has_skill(Skill.STRONG_ARM):
            if pass_distance == PassDistance.SHORT_PASS or pass_distance == PassDistance.LONG_PASS or \
                    pass_distance == PassDistance.LONG_BOMB:
                modifiers += 1

        if passer.has_skill(Skill.STUNTY):
            modifiers -= 1

        return modifiers

    def get_dodge_modifiers(self, player, position):
        """
        :param player:
        :param position: The position the player is dodging to
        :return: the modifier to be added to the dodge roll.
        """
        modifiers = 1
        tackle_zones_to = self.num_tackle_zones_at(player, position)

        ignore_opp_mods = False
        if player.has_skill(Skill.STUNTY):
            modifiers = 1
            ignore_opp_mods = True
        if player.has_skill(Skill.TITCHY):
            modifiers += 1
            ignore_opp_mods = True
        if player.has_skill(Skill.TWO_HEADS):
            modifiers += 1

        prehensile_tailers = self.get_adjacent_opponents(player, skill=Skill.PREHENSILE_TAIL)
        if len(prehensile_tailers) > 0:
            modifiers -= len(prehensile_tailers)  # subtract 1 for each prehensile tail detractor

        if not ignore_opp_mods:
            modifiers -= tackle_zones_to

        return modifiers

    def get_pickup_modifiers(self, player, position):
        """
        :param player:
        :param position: the square of the ball.
        :return: the modifier to be added to the pickup roll.
        """
        modifiers = 1
        tackle_zones = self.num_tackle_zones_at(player, position)

        if not player.has_skill(Skill.BIG_HAND):
            modifiers -= tackle_zones

        # Weather
        if self.state.weather == WeatherType.POURING_RAIN:
            if not player.has_skill(Skill.BIG_HAND):
                modifiers -= 1

        # Extra arms
        if player.has_skill(Skill.EXTRA_ARMS):
            modifiers += 1

        return modifiers

    def num_tackle_zones_in(self, player):
        """
        :param player:
        :return: Number of opponent tackle zones the player is in.
        """
        return self.num_tackle_zones_at(player, player.position)

    def num_tackle_zones_at(self, player, position):
        """
        :param player:
        :param position:
        :return: Number of opponent tackle zones player would be in, if standing at position.
        """
        tackle_zones = 0
        for p in self.get_adjacent_players(position, team=self.get_opp_team(player.team)):
            if p is not None and p.has_tackle_zone():
                tackle_zones += 1
        return tackle_zones

    def is_setup_legal(self, team):
        """
        :param team:
        :return: Whether the team has set up legally.
        """
        if not self.is_setup_legal_count(team, max_players=self.config.pitch_max,
                                        min_players=self.config.pitch_min):
            return False
        elif not self.is_setup_legal_scrimmage(team, min_players=self.config.scrimmage_min):
            return False
        elif not self.is_setup_legal_wings(team, max_players=self.config.wing_max):
            return False
        return True

    def is_setup_legal_count(self, team, tile=None, max_players=11, min_players=3):
        """
        :param team:
        :param tile: The tile area to check.
        :param max_players: The maximum number of players in the area.
        :param min_players: The minimum number of players in the area.
        :return: True if team is setup legally in the specified tile area.
        """
        min_players_checked = min(min_players, len(self.get_reserves(team)) + len(self.get_players_on_pitch(team)))
        cnt = 0
        for y in range(len(self.state.pitch.board)):
            for x in range(len(self.state.pitch.board[y])):
                if not self.is_team_side(Square(x, y), team):
                    continue
                if tile is None or self.arena.board[y][x] == tile:
                    piece = self.state.pitch.board[y][x]
                    if isinstance(piece, Player) and piece.team == team:
                        cnt += 1
        if cnt > max_players or cnt < min_players_checked:
            return False
        return True

    def num_casualties(self, team=None):
        """
        :param team: If None, return the sum of both teams casualties.
        :return: The number of casualties suffered by team.
        """
        if team is not None:
            return len(self.get_casualties(team))
        else:
            return len(self.get_casualties(self.state.home_team)) + len(self.get_casualties(self.state.away_team))

    def get_winning_team(self):
        """
        :return: The team with most touchdowns, otherwise None.
        """
        if self.state.home_team.state.score > self.state.away_team.state.score:
            return self.state.home_team
        elif self.state.home_team.state.score < self.state.away_team.state.score:
            return self.state.away_team
        return None

    def is_setup_legal_scrimmage(self, team, min_players=3):
        """
        :param team:
        :param min_players:
        :return: True if team is setup legally on scrimmage.
        """
        if team == self.state.home_team:
            return self.is_setup_legal_count(team, tile=Tile.HOME_SCRIMMAGE, min_players=min_players)
        return self.is_setup_legal_count(team, tile=Tile.AWAY_SCRIMMAGE, min_players=min_players)

    def is_setup_legal_wings(self, team, min_players=0, max_players=2):
        """
        :param team:
        :param min_players:
        :param max_players:
        :return: True if team is setup legally on the wings.
        """
        if team == self.state.home_team:
            return self.is_setup_legal_count(team, tile=Tile.HOME_WING_LEFT, max_players=max_players, min_players=min_players) and \
                   self.is_setup_legal_count(team, tile=Tile.HOME_WING_RIGHT, max_players=max_players, min_players=min_players)
        return self.is_setup_legal_count(team, tile=Tile.AWAY_WING_LEFT, max_players=max_players, min_players=min_players) and \
               self.is_setup_legal_count(team, tile=Tile.AWAY_WING_RIGHT, max_players=max_players, min_players=min_players)

    def get_procedure_names(self):
        """
        :return: a list of procedure names in the stack.
        """
        procs = []
        for proc in self.state.stack.items:
            if isinstance(proc, Turn) and proc.quick_snap:
                procs.append("QuickSnap")
            elif isinstance(proc, Turn) and proc.blitz:
                procs.append("Blitz")
            else:
                procs.append(proc.__class__.__name__)
        return procs

    def get_player_action_type(self):
        """
        :param player:
        :return: the player ActionType if there is any on the stack.
        """
        if self.state.game_over:
            return None
        proc = self.state.stack.peek()
        if isinstance(proc, PlayerAction):
            return proc.player_action_type
        return None

    def remove_recursive_refs(self):
        """
        Removes recursive references. Must be called before serializing.
        """
        for team in self.state.teams:
            for player in team.players:
                player.team = None

    def add_recursive_refs(self):
        """
        Adds recursive references. Can be called after serializing.
        """
        for team in self.state.teams:
            for player in team.players:
                player.team = team

    def get_termination_time(self):
        """
        The time at which the current turn must be terminated - or the opponent's action choice (like selecting block die).
        """
        if self.state.termination_opp is not None:
            return self.state.termination_opp
        return self.state.termination_turn

    def is_timed_out(self):
        """
        Returns true if the game timed out - i.e. it was longer than the allowed self.config.time_limits.game - only if in competition mode.
        """
        if not self.config.competition_mode:
            return
        if self.end_time is None or self.start_time is None:
            return False
        return self.end_time - self.start_time > self.config.time_limits.game

    def get_team_by_id(self, team_id):
        """
        :param team_id:
        :return: returns the team with the id or None
        """
        if self.state.home_team.team_id == team_id:
            return self.state.home_team
        if self.state.away_team.team_id == team_id:
            return self.state.away_team
        return None

    def get_winner(self):
        """
        returns the winning agent of the game. None if it's a draw.
        If the game timed out the current player loses.
        A disqualified player will lose.
        If the game is over, the team with most TDs win.
        """
        # Disqualified players lose
        if self.disqualified_agent is not None:
            return self.get_other_agent(self.disqualified_agent)
        
        # If game timed out the current player lost
        if self.is_timed_out():
            if self.home_agent == self.actor:
                return self.away_agent
            elif self.away_agent == self.actor:
                return self.home_agent
        
        # If the game is over the player with most TDs wins
        if self.state.game_over:
            return self.get_team_agent(self.get_winning_team())
        
        return None

    def get_other_agent(self, agent):
        """
        Returns the other agent in the game.
        """
        if agent is None:
            return None
        if agent == self.home_agent:
            return self.away_agent
        return self.home_agent

    def get_other_active_player_id(self):
        """
        Returns the player id of the other player involved in current procedures - if any.
        """
        for proc in self.state.stack.items:
            if isinstance(proc, Block):
                if proc.defender is not None:
                    return proc.defender.player_id
            if isinstance(proc, PassAction):
                if proc.catcher is not None:
                    return proc.catcher.player_id
            if isinstance(proc, Handoff):
                if proc.catcher is not None:
                    return proc.catcher.player_id
            if isinstance(proc, Push):
                if proc.catcher is not None:
                    return proc.player.player_id
        return None

    def replace_home_agent(self, agent):
        """
        Replaces the home agent safely.
        :param agent:
        """
        if self.actor.agent_id == self.home_agent.agent_id:
            self.actor = agent
        self.home_agent = agent

    def replace_away_agent(self, agent):
        """
        Replaces the away agent safely.
        :param agent:
        """
        if self.actor.agent_id == self.away_agent.agent_id:
            self.actor = agent
        self.away_agent = agent

    def has_report_of_type(self, outcome_type, last=None):
        """
        :param outcome_type:
        :return: True if the the game has reported an outcome of the given type. If last is specified, only the recent number of reports are checked.
        """
        assert last is None or last > 0
        n = len(self.state.reports) if last is None else min(len(self.state.reports), last)
        j = len(self.state.reports) - n
        for report in self.state.reports[j:]:
            if report.outcome_type == outcome_type:
                return True
        return False

    def get_balls_at(self, position, in_air=False):
        """
        Assumes there is only one ball on the square
        :param position:
        :param in_air:
        :return: Ball or None
        """
        balls = []
        for ball in self.state.pitch.balls:
            if ball.position == position and (ball.on_ground or in_air):
                balls.append(ball)
        return balls

    def get_ball_at(self, position, in_air=False):
        """
        Assumes there is only one ball on the square
        :param position:
        :param in_air:
        :return: Ball or None
        """
        balls_at = self.get_balls_at(position, in_air)
        return balls_at[0] if balls_at else None

    def get_ball_positions(self):
        """
        :return: The position of the ball. If no balls are in the arena None is returned. If multiple balls are in the
        arena, the position of the first ball is return.
        """
        return [ball.position for ball in self.state.pitch.balls]

    def get_ball_position(self):
        """
        Assumes there is only one ball on the square
        :return: Ball or None
        """
        for ball in self.state.pitch.balls:
            return ball.position
        return None

    def get_ball_carrier(self):
        """
        :return: the ball carrier if any - otherwise None.
        """
        ball_position: Square = self.get_ball_position()
        if ball_position is None:
            return None
        else:
            return self.get_player_at(ball_position)

    def is_out_of_bounds(self, position):
        """
        :param position:
        :return: True if pos is out of bounds.
        """
        return position.x < 1 or position.x >= self.state.pitch.width-1 or position.y < 1 or position.y >= self.state.pitch.height-1

    def get_push_squares(self, from_position, to_position):
        """
        :param from_position: The position of the attacker.
        :param to_position: The position of the defender.
        :return: Possible square to push the player standing on pos_to on to.
        """
        squares_to = self.get_adjacent_squares(to_position, out=True)
        squares_empty = []
        squares_out = []
        squares = []
        #print("From:", pos_from)
        for square in squares_to:
            #print("Checking: ", square)
            #print("Distance: ", pos_from.distance(square, manhattan=False))
            include = False
            if from_position.x == to_position.x or from_position.y == to_position.y:
                if from_position.distance(square, manhattan=False) >= 2:
                    include = True
            else:
                if from_position.distance(square, manhattan=True) >= 3:
                    include = True
            #print("Include: ", include)
            if include:
                if self.is_out_of_bounds(square):
                    squares_out.append(square)
                elif self.get_player_at(square) is None:
                    squares_empty.append(square)
                squares.append(square)
        if len(squares_empty) > 0:
            return squares_empty
        if len(squares_out) > 0:
            return squares_out
        assert len(squares) > 0
        return squares

    def get_square(self, x, y):
        """
        Returns an existing square object for the given position to avoid a new instantiation. If the square object
        is out of bounds it may be instantiated.
        :param x:
        :param y:
        :return: A square with the position (x,y)
        """
        if 0 >= x < self.state.pitch.width and 0 >= y < self.state.pitch.height:
            return self.state.pitch.squares[y][x]
        return Square(x, y)

    def get_adjacent_squares(self, position, diagonal=True, out=False, occupied=True, distance=1):
        """
        Returns a list of adjacent squares from the position.
        :param position:
        :param diagonal: include diagonal
        :param out: include squares outside of the pitch and in the crowd.
        :param occupied: include occupied squares
        :param distance: distance of adjacency. E.g. use distance 2 when checking for leap.
        :return:
        """
        squares = []
        r = range(-distance, distance+1)
        for yy in r:
            for xx in r:
                if yy == 0 and xx == 0:
                    continue
                sq = self.get_square(position.x+xx, position.y+yy)
                if not out and self.is_out_of_bounds(sq):
                    continue
                if not occupied and self.get_player_at(sq) is not None:
                    continue
                if diagonal:
                    squares.append(sq)
                elif xx == 0 or yy == 0:
                    squares.append(sq)
        return squares

    def get_adjacent_opponents(self, player, diagonal=True, down=True, standing=True, stunned=True, skill=None):
        """
        Returns a list of adjacent opponents to the player it its current position.
        :param position:
        :param diagonal: Whether to include diagonally adjacent players.
        :param down: Whether to include down players.
        :param standing: Whether to include standing players.
        :param stunned: Whether to include stunned players.
        :param skill: Only include players with this skill.
        :return:
        """
        return self.get_adjacent_players(player.position, self.get_opp_team(player.team), diagonal, down, standing, stunned, skill=skill)

    def get_adjacent_teammates(self, player, diagonal=True, down=True, standing=True, stunned=True, skill=None):
        """
        Returns a list of adjacent teammates to the player it its current position.
        :param position:
        :param diagonal: Whether to include diagonally adjacent players.
        :param down: Whether to include down players.
        :param standing: Whether to include standing players.
        :param stunned: Whether to include stunned players.
        :param skill: Only include players with this skill.
        :return:
        """
        return self.get_adjacent_players(player.position, player.team, diagonal, down, standing, stunned, skill=skill)

    def get_adjacent_players(self, position, team=None, diagonal=True, down=True, standing=True, stunned=True, skill=None):
        """
        Returns a list of adjacent player to the position.
        :param position:
        :param team: Team of players to include. Includes all teams if None.
        :param diagonal: Whether to include diagonally adjacent players.
        :param down: Whether to include down players.
        :param standing: Whether to include standing players.
        :param stunned: Whether to include stunned players.
        :param skill: Only include players with this skill.
        :return:
        """
        players = []
        for square in self.get_adjacent_squares(position, diagonal=diagonal):
            player_at = self.get_player_at(square)
            if player_at is None:
                continue
            if not down and not player_at.state.up:
                continue
            if not standing and player_at.state.up:
                continue
            if not stunned and player_at.state.stunned:
                continue
            if team is not None and player_at.team != team:
                continue
            if skill is not None and not player_at.has_skill(skill):
                continue
            players.append(player_at)
        return players

    def get_assisting_players(self, player, opp_player, foul=False):
        """
        Returns a list of assisting players in a block between player and opp_player.
        :param player:
        :param opp_player:
        :param foul: Indicates whether it is a foul. The Guard skill is ignored if fould=True.
        :return: a list of assisting players.
        """
        assists = []
        for yy in range(-1, 2, 1):
            for xx in range(-1, 2, 1):
                if yy == 0 and xx == 0:
                    continue
                p = Square(opp_player.position.x+xx, opp_player.position.y+yy)
                if not self.is_out_of_bounds(p) and player.position != p:
                    player_at = self.get_player_at(p)
                    if player_at is not None:
                        if player_at.team == player.team:
                            if not player_at.can_assist():
                                continue
                            if (not foul and player_at.has_skill(Skill.GUARD)) or \
                                            self.num_tackle_zones_in(player_at) <= 1:
                                assists.append(player_at)
        return assists

    def num_block_dice(self, attacker, defender, blitz=False, dauntless_success=False):
        """
        :param attacker: 
        :param defender: 
        :param blitz: if it is a blitz
        :param dauntless_success: If a dauntless rolls was successful.
        :return: The number of block dice used in a block between the attacker and defender.
        """
        return self.num_block_dice_at(attacker, defender, attacker.position, blitz, dauntless_success)

    def num_block_dice_at(self, attacker, defender, position, blitz=False, dauntless_success=False):
        """
        :param attacker:
        :param defender:
        :param blitz: if it is a blitz
        :param dauntless_success: If a dauntless rolls was successful.
        :return: The number of block dice used in a block between the attacker and defender if the attacker block at the given position.
        """

        # Determine dice and favor
        st_for = attacker.get_st()
        st_against = defender.get_st()

        # Horns
        if blitz and attacker.has_skill(Skill.HORNS):
            st_for += 1

        # Dauntless
        if dauntless_success:
            st_for = max(st_for, st_against)

        # Find assists
        assists_for, assists_against = self.num_assists_at(attacker, defender, position, foul=False)

        st_for = st_for + assists_for
        st_against = st_against + assists_against

        # Determine dice and favor
        if st_for > 2 * st_against:
            return 3
        elif st_for > st_against:
            return 2
        elif st_for == st_against:
            return 1
        elif st_for*2 < st_against:
            return -3
        elif st_for < st_against:
            return -2

    def num_assists_at(self, attacker, defender, position, foul: bool = False):
        '''
        Return net assists for a block of player on opp_player when player has moved to position first.  Required for
        calculating assists after moving in a Blitz action.
        :param attacker: Player
        :param defender: Player
        :param position: Square
        :param ignore_guard: bool
        :return: int - Net # of assists
        '''

        # Note that because blitzing/fouling player may have moved, calculating assists for is slightly different to against.
        # Assists against
        opp_assisters = self.get_adjacent_players(position, team=self.get_opp_team(attacker.team), down=False)
        n_assist_against: int = 0
        for assister in opp_assisters:
            # For each opponent, check if they can assist
            if assister == defender:
                continue
            if not assister.can_assist():
                continue
            if not foul and assister.has_skill(Skill.GUARD):
                n_assist_against += 1
            else:
                # Check if in a tackle zone of anyone besides player (at either original square, or "position")
                adjacent_to_assisters = self.get_adjacent_opponents(assister, down=False)
                found_adjacent = False
                for adjacent_to_assister in adjacent_to_assisters:
                    # Need to make sure we take into account the blocking/blitzing player may be in a different square than currently represented on the board.
                    if adjacent_to_assister.position == position or adjacent_to_assister.position == attacker.position or not adjacent_to_assister.can_assist():
                        continue
                    else:
                        found_adjacent = True
                        break
                if not found_adjacent:
                    n_assist_against += 1
        # Assists for
        assisters = self.get_adjacent_opponents(defender, down=False)
        n_assists_for: int = 0
        for assister in assisters:
            if assister == attacker:
                continue
            if not foul and assister.has_skill(Skill.GUARD):
                n_assists_for += 1
            elif not assister.can_assist():
                continue
            else:
                adjacent_to_assisters = self.get_adjacent_opponents(assister, down=False)
                found_adjacent = False
                for adjacent_to_assister in adjacent_to_assisters:
                    if adjacent_to_assister == defender:
                        continue
                    if not adjacent_to_assister.can_assist():
                        continue
                    else:
                        found_adjacent = True
                        break
                if not found_adjacent:
                    n_assists_for += 1
        return (n_assists_for, n_assist_against)

    def get_pass_distances(self, passer):
        """
        :param passer:
        :param weather:
        :return: two lists (squares, distances) indicating the PassDistance to each square that the passer can pass to.
        """
        return self.get_pass_distances_at(passer, passer.position)

    def get_pass_distances_at(self, passer, position):
        """
        :param passer:
        :param weather:
        :return: two lists (squares, distances) indicating the PassDistance to each square that the passer can pass to if at the given position.
        """
        squares = []
        distances = []
        distances_allowed = [PassDistance.QUICK_PASS,
                             PassDistance.SHORT_PASS,
                             PassDistance.LONG_PASS,
                             PassDistance.LONG_BOMB,
                             PassDistance.HAIL_MARY] if Skill.HAIL_MARY_PASS in passer.get_skills() \
            else [PassDistance.QUICK_PASS, PassDistance.SHORT_PASS, PassDistance.LONG_PASS, PassDistance.LONG_BOMB]
        if self.state.weather == WeatherType.BLIZZARD:
            distances_allowed = [PassDistance.QUICK_PASS, PassDistance.SHORT_PASS]
        for y in range(len(self.state.pitch.board)):
            for x in range(len(self.state.pitch.board[y])):
                to_position = Square(x, y)
                if self.is_out_of_bounds(to_position) or position == to_position:
                    continue
                distance = self.get_pass_distance(position, to_position)
                if distance in distances_allowed:
                    squares.append(to_position)
                    distances.append(distance)
        return squares, distances

    def get_pass_distance(self, from_position, to_position):
        """
        :param from_position:
        :param to_position:
        :return: The PassDistance between the passer and the target position.
        """
        distance_x = abs(from_position.x - to_position.x)
        distance_y = abs(from_position.y - to_position.y)
        if distance_y >= len(Rules.pass_matrix) or distance_x >= len(Rules.pass_matrix[0]):
            return PassDistance.HAIL_MARY
        distance = Rules.pass_matrix[distance_y][distance_x]
        return PassDistance(distance)

    def get_interceptors(self, position_from, position_to, team):
        """
        Finds interceptors using the following rules:
        1) Find line x from a to b
        3) Find squares s where x intersects
        3) Find manhattan neighboring n squares of s
        4) Remove squares where distance to a is larger than dist(a,b)
        5) Remove squares without standing opponents with hands
        6) Determine players on squares
        :param position_from where the passer is
        :param position_to where the ball is passed to
        """

        # 1) Find line x from a to b
        x = get_line((position_from.x, position_from.y), (position_to.x, position_to.y))

        # 3) Find squares s where x intersects
        s = []
        for i in x:
            s.append(Square(i[0], i[1]))

        # 3) Include manhattan neighbors s into n
        # 4) Remove squares where distance to a is larger than dist(a,b)
        max_distance = position_from.distance(position_to)
        n = set()
        for square in s:
            for neighbor in self.get_adjacent_squares(square) + [square]:

                if neighbor in n:
                    continue

                # 4) Remove squares where distance to a is larger than dist(a,b)
                if neighbor.distance(position_from) > max_distance:
                    continue
                if neighbor.distance(position_to) > max_distance:
                    continue
                if neighbor.x > max(position_from.x, position_to.x) or neighbor.x < min(position_from.x, position_to.x):
                    continue
                if neighbor.y > max(position_from.y, position_to.y) or neighbor.y < min(position_from.y, position_to.y):
                    continue

                # 5) Remove squares without standing opponents with hands
                player_at = self.get_player_at(neighbor)
                if player_at is None:
                    continue
                if player_at.team != team:
                    continue
                if not player_at.can_catch():
                    continue
                if player_at.has_skill(Skill.NO_HANDS):
                    continue

                n.add(neighbor)

        if position_from in n:
            n.remove(position_from)
        if position_to in n:
            n.remove(position_to)

        players = []
        for square in n:
            players.append(self.get_player_at(square))

        return players

    def get_available_actions(self):
        """
        :return: a list of available action choices in the current state.
        """
        return self.state.available_actions

    def clear_board(self):
        for player in self.get_players_on_pitch(self.state.home_team):
            self.pitch_to_reserves(player)
        for player in self.get_players_on_pitch(self.state.away_team):
            self.pitch_to_reserves(player)
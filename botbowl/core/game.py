"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains the Game class, which is the main class and interface used to interact with a game in botbowl.
"""
from botbowl.core.load import *
from botbowl.core.procedure import *
from botbowl.core.forward_model import Trajectory, MovementStep

from copy import deepcopy
from typing import Optional, Tuple, List, Union, Any


class InvalidActionError(Exception):
    pass


class Game:
    replay: Optional[Replay]
    game_id: str
    home_agent: Agent
    away_agent: Agent
    arena: TwoPlayerArena
    config: Configuration
    ruleset: RuleSet
    state: GameState
    rnd: np.random.RandomState
    ff_map: Any #??
    start_time: Optional[float]
    end_time: Optional[float]
    last_request_time: Optional[float]
    last_action_time: Optional[float]
    action: Optional[Action]
    trajectory: Trajectory
    square_shortcut: List[List[Square]]

    def __init__(self, game_id, home_team: Team, away_team: Team, home_agent: Agent, away_agent: Agent,
                 config: Optional[Configuration] = None,
                 arena: Optional[TwoPlayerArena] = None,
                 ruleset: Optional[RuleSet] = None,
                 state: Optional[GameState] = None,
                 seed=None,
                 record: bool = False):
        assert config is not None or arena is not None
        assert config is not None or ruleset is not None
        assert home_team.team_id != away_team.team_id
        self.replay = Replay(replay_id=game_id) if record else None
        self.game_id = game_id
        self.home_agent = home_agent
        self.away_agent = away_agent
        self.arena = load_arena(config.arena) if arena is None else arena
        self.config = config
        self.ruleset = load_rule_set(config.ruleset) if ruleset is None else ruleset
        self.state = state if state is not None else GameState(self, deepcopy(home_team), deepcopy(away_team))
        self.rnd = np.random.RandomState(seed)
        self.ff_map = None
        self.start_time = None
        self.end_time = None
        self.last_request_time = None
        self.last_action_time = None
        self.action = None
        self.trajectory = Trajectory()
        self.square_shortcut = self.state.pitch.squares

    def to_json(self, ignore_reports: bool = False):
        return {
            'game_id': self.game_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'state': self.state.to_json(ignore_reports=ignore_reports),
            'stack': self.get_procedure_names(),
            'home_agent': self.home_agent.to_json(),
            'away_agent': self.away_agent.to_json(),
            'squares_moved': self._squares_moved(),
            'arena': self.arena.to_json(),
            'ruleset': self.ruleset.name,
            'can_home_team_use_reroll': self.can_use_reroll(self.state.home_team),
            'can_away_team_use_reroll': self.can_use_reroll(self.state.away_team),
            'actor_id': self.actor.agent_id if self.actor is not None else None,
            'time_limits': self.config.time_limits.to_json(),
            'active_other_player_id': self.get_other_active_player_id(),
            'rounds': self.config.rounds,
        }

    def enable_forward_model(self) -> None:
        """
        Enables the forward model. Should not be called before Game.init(). Can only be called once. Can't be undone
        """
        if self.trajectory.enabled:
            raise RuntimeError("Forward model already enabled")

        self.trajectory.enabled = True
        self.state.set_trajectory(self.trajectory)

    def get_step(self) -> int:
        """
        Returns an int that is the forward model step counter. The step counter can be used to revert the game state
        to this state with Game.revert()
        """
        return self.trajectory.current_step

    def revert(self, to_step: int) -> None:
        """
        Reverts the game state to how a the step to_step
        """
        assert self.trajectory.enabled
        self.trajectory.revert(to_step)

    @property
    def active_team(self) -> Optional[Team]:
        if len(self.state.available_actions) > 0:
            return self.state.available_actions[0].team
        else:
            return None

    @property
    def actor(self) -> Optional[Agent]:
        return self.get_team_agent(self.active_team)

    def init(self) -> None:
        """
        Initialized the Game. The START_GAME action must still be called after this if humans are in the game.
        """
        EndGame(self)
        Pregame(self)
        if not self.away_agent.human:
            self.away_agent.new_game(self, self.state.away_team)
        if not self.home_agent.human:
            self.home_agent.new_game(self, self.state.home_team)
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

    def step(self, action=None) -> None:
        """
        Runs until an action from a human is required. If game requires an action to continue one must be given.
        :param action: Action to perform, can be None if game does not require any.
        :return:
        """

        # Ensure player points to player object
        if action is not None:
            action.player = self.get_player(action.player.player_id) if action.player is not None else None

        # Set action as a property so other methods can access it
        self.action = action

        # Update game
        while True:

            # Perform game step
            done = self._one_step(self.action)

            # Game over
            if self.state.game_over:
                self._end_game()
                break

            if self.state.stack.is_empty():
                print("Somethings wrong")

            # if procedure is ready for input
            if done:

                # If human player - wait for input
                if self.actor is None or self.actor.human:
                    break

                # Query agent for action
                self.last_request_time = time.time()
                self.action = self._safe_act()

                # Check if time limit was violated
                self.last_action_time = time.time()

                # Check clocks if competition mode
                if self.config.competition_mode:
                    self._check_clocks()  # Might override the action if clock was violated

                # Did the game terminate?
                if self.state.game_over:
                    self._end_game()
                    break
            else:

                # If not in fast mode - wait for input before continuing
                if not self.config.fast_mode:
                    break

                # Else continue procedure with no action
                self.action = None

        self.trajectory.next_step()

    def refresh(self) -> None:
        """
        Checks clocks and runs forced actions. Useful in called in human games.
        """
        self.action = None
        if self.config.competition_mode:
            self._check_clocks()
        if self.action is not None:
            self.step(self.action)

    def _check_clocks(self) -> None:
        """
        Checks if clocks are done.
        """

        # No time limit for this action
        if not self.has_agent_clock(self.actor):
            return

        # Agent too slow?
        clock = self.get_agent_clock(self.actor)
        if clock.is_done():

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

        if not clock.is_running():
            clock.resume()

    def _end_game(self) -> None:
        """
        End the game
        """
        # Game ended when the last action was received - to avoid timout during finishing procedures
        self.end_time = self.last_action_time

        # Let agents know that the game ended
        if not self.home_agent.human:
            self.home_agent.end_game(self)
        if not self.away_agent.human:
            self.away_agent.end_game(self)

        # Record state
        if self.replay is not None:
            self.replay.record_step(self)
            self.replay.dump(self)

    def _is_action_allowed(self, action: Action) -> bool:
        """
        Checks whether the specified action is allowed by comparing to actions in self.state.available_actions.
        :param action:
        :return: True if the specified actions is allowed.
        """
        if action is None:
            return True
        for action_choice in self.state.available_actions:
            if action.action_type == action_choice.action_type:
                # Type checking
                if type(action.action_type) is not ActionType:
                    print("Illegal action type: ", type(action.action_type))
                    return False
                if action.player is not None and not isinstance(action.player, Player):
                    print("Illegal player type: ", type(action.action_type), action, self.state.stack.peek())
                    return False
                if action.position is not None and not isinstance(action.position, Square):
                    print("Illegal position type:", type(action.position), action.action_type.name)
                    return False
                # Check if player argument is used instead of position argument
                if len(action_choice.players) == 0 and action.player is not None and action.position is None:
                    action.position = action.player.position
                    # Check if player argument is used instead of position argument
                elif len(action_choice.positions) == 0 and action.position is not None and action.player is None:
                    action.player = self.get_player_at(action.position)
                # Check player argument
                if len(action_choice.players) > 1 and action.player not in action_choice.players:
                    if action.player is None:
                        print("Illegal player: None")
                    else:
                        print("Illegal player:", action.player.to_json(), action.action_type.name)
                    return False
                # Check position argument
                if len(action_choice.positions) > 0 and action.position not in action_choice.positions:
                    if action.position is None:
                        print("Illegal position: None")
                    else:
                        print("Illegal position:", action.position.to_json(), action.action_type.name)
                    return False
                return True
        return False

    def _safe_act(self) -> Optional[Action]:
        """
        Gets action from agent and sets correct player reference.
        """
        action = self.actor.act(self)
        if not type(action) == Action:
            return None
        # Correct player object
        if action.player is not None:
            if action.player.player_id not in self.state.player_by_id.keys():
                print(f"Unknown player id {action.player.player_id}")
                action.player = None
            else:
                action.player = self.state.player_by_id[action.player.player_id]
        return action

    def _forced_action(self) -> Action:
        """
        Return action that prioritize to end the player's turn.
        """
        # Take first negative action
        for action_type in [ActionType.END_TURN, ActionType.END_SETUP, ActionType.END_PLAYER_TURN,
                            ActionType.SELECT_NONE, ActionType.HEADS, ActionType.KICK, ActionType.SELECT_DEFENDER_DOWN,
                            ActionType.SELECT_DEFENDER_STUMBLES, ActionType.SELECT_ATTACKER_DOWN,
                            ActionType.SELECT_PUSH, ActionType.SELECT_BOTH_DOWN, ActionType.DONT_USE_REROLL,
                            ActionType.DONT_USE_APOTHECARY]:
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

    def _squares_moved(self) -> list:
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

    def _one_step(self, action: Action) -> bool:
        """
        Executes one step in the game if it is allowed.
        :param action: Action from agent. Can be None if no action is required.
        :return: True if game requires action or game is over, False if not
        """

        # Get proc
        proc = self.state.stack.peek()

        # If no action and action is required
        if action is None and len(self.state.available_actions) > 0:
            raise InvalidActionError("None action is not allowed when actions are available")

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

                    if type(action) is Action:
                        raise InvalidActionError(
                            f"Action not allowed {action.to_json() if action is not None else 'None'}")
                    else:
                        raise InvalidActionError(f"Action not allowed {action}")

        # Run proc
        if self.config.debug_mode:
            print("Proc={}".format(proc))
            print("Action={}".format(action.action_type if action is not None else "None"))
            print("Players on Field={}".format(len(self.get_players_on_pitch())))

        proc.done = proc.step(action)

        if self.config.debug_mode:
            print("Done={}".format(proc.done))
            print(f"DONE={self.state.stack.peek().done}")
            print("Players on Field={}".format(len(self.get_players_on_pitch())))

        # Used if players was accidentally cloned
        if self.config.debug_mode:
            for y in range(len(self.state.pitch.board)):
                for x in range(len(self.state.pitch.board)):
                    assert self.state.pitch.board[y][x] is None or \
                           (self.state.pitch.board[y][x].position.x == x and self.state.pitch.board[y][
                               x].position.y == y)

            for team in self.state.teams:
                for player in team.players:
                    if not (player.position is None or
                            self.state.pitch.board[player.position.y][player.position.x] == player):
                        raise Exception("Player position violation")

        # Remove all finished procs
        while not self.state.stack.is_empty() and self.state.stack.peek().done:
            if self.config.debug_mode:
                print("--Proc={}".format(self.state.stack.peek()))
            # Call end before removing
            proc_end = self.state.stack.peek()
            proc_end.end()
            self.state.stack.remove(proc_end)

        # Record state
        if self.replay is not None:
            if action is not None and action.action_type is not ActionType.PLACE_BALL:
                self.replay.record_step(self)

        # Is game over
        if self.state.stack.is_empty():
            return False  # Can end the game without user input

        if self.config.debug_mode:
            print("-Proc={}".format(self.state.stack.peek()))

        # Initialize if not
        if not self.state.stack.peek().started:
            proc = self.state.stack.peek()
            proc.start()
            proc.started = True

        # Update available actions
        self.set_available_actions()

        if self.config.debug_mode:
            print(f"{len(self.state.available_actions)} available actions")

        if len(self.state.available_actions) == 0:
            return False  # Can continue without user input

        # End player turn if only action available
        if len(self.state.available_actions) == 1 and \
                self.state.available_actions[0].action_type == ActionType.END_PLAYER_TURN:
            return self._one_step(Action(ActionType.END_PLAYER_TURN))

        return True  # Game needs user input

    def remove_clocks(self) -> None:
        """
        Remove all clocks.
        """
        self.state.clocks.clear()

    def remove_secondary_clocks(self) -> None:
        """
        Remove all secondary clocks and resume the primary clock - if any.
        """
        self.state.clocks = [clock for clock in self.state.clocks if clock.is_primary]
        for clock in self.state.clocks:
            if not clock.is_running():
                clock.resume()

    def get_clock(self, team: Team) -> Optional[Clock]:
        """
        Returns the clock belonging to the given team.
        """
        for clock in self.state.clocks:
            if clock.team == team:
                return clock
        return None

    def get_agent_clock(self, agent: Agent) -> Optional[Clock]:
        """
        Returns the clock belonging to the given agent's team.
        """
        for clock in self.state.clocks:
            if clock.team == self.get_agent_team(agent):
                return clock
        return None

    def has_clock(self, team: Team) -> bool:
        """
        Returns true if the given team has a clock.
        """
        for clock in self.state.clocks:
            if clock.team == team:
                return True
        return False

    def has_agent_clock(self, agent: Agent) -> bool:
        """
        Returns true if the given agent's team has a clock.
        """
        for clock in self.state.clocks:
            if clock.team == self.get_agent_team(agent):
                return True
        return False

    def pause_clocks(self) -> None:
        """
        Pauses all clocks.
        """
        for clock in self.state.clocks:
            if clock.is_running():
                clock.pause()

    def add_secondary_clock(self, team: Team) -> None:
        """
        Adds a secondary clock for quick decisions.
        """
        self.pause_clocks()
        assert team is not None and type(team) == Team
        clock = Clock(team, self.config.time_limits.secondary)
        self.state.clocks.append(clock)

    def add_primary_clock(self, team: Team) -> None:
        """
        Adds a primary clock that will be paused if secondary clocks are added.
        """
        self.state.clocks.clear()
        assert team is not None and type(team) == Team
        clock = Clock(team, self.config.time_limits.turn, is_primary=True)
        self.state.clocks.append(clock)

    def get_seconds_left(self, team: Team = None) -> Optional[int]:
        '''
        Returns the number of seconds left on the clock for the given team and None if the given team has no clock.
        '''
        if team is None:
            if self.actor is None:
                return None
            t = self.get_agent_team(self.actor)
        else:
            t = team
        for clock in self.state.clocks:
            if clock.team == t:
                return clock.get_seconds_left()
        return None

    # redefined below
    #def is_started(self):
    #    """
    #    Returns true if the game is started else false.
    #    """
    #    return (not self.state.game_over) and len(self.state.stack.items) > 0

    def get_team_agent(self, team: Team) -> Optional[Agent]:
        """
        :param team:
        :return: The agent who's controlling the specified team.
        """
        if team is None:
            return None
        if team == self.state.home_team:
            return self.home_agent
        return self.away_agent

    def get_agent_team(self, agent: Agent) -> Optional[Team]:
        """
        :param agent: The agent controlling the team
        :return: The team controlled by the specified agent.
        """
        if agent is None:
            return None
        if agent == self.home_agent:
            return self.state.home_team
        return self.state.away_team

    def set_seed(self, seed: int) -> None:
        '''
        Sets the random seed of the game.
        '''
        self.seed = seed
        self.rnd = np.random.RandomState(self.seed)

    def set_available_actions(self) -> None:
        """
        Calls the current procedure's available_actions() method and sets the game's available actions to the returned
        list.
        """
        self.state.available_actions = self.state.stack.peek().available_actions()

    def report(self, outcome) -> None:
        """
        Adds the outcome to the game's reports.
        """
        self.state.reports.append(outcome)

    def is_started(self) -> bool:
        return self.start_time is not None

    def is_team_side(self, position: Square, team: Team) -> bool:
        """
        :param position:
        :param team:
        :return: Returns True if pos is on team's side of the arena.
        """
        if team == self.state.home_team:
            return self.arena.board[position.y][position.x] in TwoPlayerArena.home_tiles
        return self.arena.board[position.y][position.x] in TwoPlayerArena.away_tiles

    def get_team_side(self, team: Team) -> List[Square]:
        """
        :param team:
        :return: a list of squares on team's side of the arena.
        """
        tiles = []
        for y in range(len(self.arena.board)):
            for x in range(len(self.arena.board[y])):
                if self.arena.board[y][x] in \
                        (TwoPlayerArena.home_tiles if team == self.state.home_team else TwoPlayerArena.away_tiles):
                    tiles.append(self.get_square(x, y))
        return tiles

    def is_scrimmage(self, position: Square) -> bool:
        """
        :param position:
        :return: Returns True if pos is on the scrimmage line.
        """
        return self.arena.board[position.y][position.x] in TwoPlayerArena.scrimmage_tiles

    def is_wing(self, position: Square, right) -> bool:
        """
        :param position:
        :param right: Whether to check on the right side of the arena. If False, it will check on the left side.
        :return: True if pos is on the arena's wing and on the specified side.
        """
        if right:
            return self.arena.board[position.y][position.x] in TwoPlayerArena.wing_right_tiles
        return self.arena.board[position.y][position.x] in TwoPlayerArena.wing_left_tiles

    def remove_balls(self) -> None:
        """
        Removes all balls from the arena.
        """
        self.state.pitch.balls.clear()

    def is_last_turn(self) -> bool:
        """
        :return: True if this turn is the last turn of the game.
        """
        return self.get_next_team().state.turn == self.config.rounds and self.state.half == 2

    def is_last_round(self) -> bool:
        """
        :return: True if this round is the las round of the game.
        """
        return self.state.round == self.config.rounds

    def get_next_team(self) -> Team:
        """
        :return: The team who's turn it is next.
        """
        idx = self.state.turn_order.index(self.state.current_team)
        if idx + 1 == len(self.state.turn_order):
            return self.state.turn_order[0]
        return self.state.turn_order[idx + 1]

    def add_or_skip_turn(self, turns: None) -> None:
        """
        Adds or removes a number of turns from the current half. This method will raise an assertion error if the turn
        counter goes to a negative number.
        :param turns: The number of turns to add (if positive) or remove (if negative). 
        """
        for team in self.state.teams:
            team.state.turn += turns
            assert team.state.turn >= 0

    def get_player(self, player_id: str) -> Optional[Player]:
        """
        :param player_id: 
        :return: Returns the player with player_id
        """
        return self.state.player_by_id[player_id]

    def get_player_at(self, position: Square) -> Optional[Player]:
        """
        :param position: 
        :return: Returns the player at pos else None.
        """
        return self.state.pitch.board[position.y][position.x]

    def set_turn_order_from(self, first_team: Team) -> None:
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

    def set_turn_order_after(self, last_team: Team) -> None:
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

    def get_turn_order(self) -> List[Team]:
        """
        :return: The list of teams sorted by turn order.
        """
        return self.state.turn_order

    def is_home_team(self, team: Team) -> bool:
        """
        :return: True if team is the home team.
        """
        return team == self.state.home_team

    def get_opp_team(self, team: Team) -> Team:
        """
        :param team: 
        :return: The opponent team of team.
        """
        return self.state.home_team if self.state.away_team == team else self.state.away_team

    def get_dugout(self, team: Team) -> Dugout:
        return self.state.dugouts[team.team_id]

    def get_reserves(self, team: Team) -> List[Player]:
        """
        :param team: 
        :return: The reserves in the dugout of this team.
        """
        return self.get_dugout(team).reserves

    def get_knocked_out(self, team: Team) -> List[Player]:
        """
        :param team:
        :return: The knocked out players in the dugout of this team.
        """
        return self.get_dugout(team).kod

    def get_casualties(self, team: Team) -> List[Player]:
        """
        :param team:
        :return: The badly hurt, injured, and dead players in th dugout of this team.
        """
        return self.get_dugout(team).casualties

    def get_dungeon(self, team: Team) -> List[Player]:
        """
        :param team:
        :return: The ejected players of this team, who's locked to a cell in the dungeon.
        """
        return self.get_dugout(team).dungeon

    def current_turn(self) -> Optional[Turn]:
        """
        :return: The top-most Turn procedure in the stack.
        """
        for i in reversed(range(self.state.stack.size())):
            proc = self.state.stack.items[i]
            if isinstance(proc, Turn):
                return proc
        return None

    def can_use_reroll(self, team: Team) -> bool:
        """
        :param team:
        :return: True if the team can use reroll right now (i.e. this turn).
        """
        if not team.state.reroll_used and team.state.rerolls > 0 and self.state.current_team == team:
            current_turn = self.current_turn()
            if current_turn is not None and isinstance(current_turn, Turn):
                return not current_turn.quick_snap
        return False

    def get_kicking_team(self, half: Optional[int] = None):
        """
        :param half: Set this to None if you want the team who's kicking this drive.
        :return: The team who's kicking in the specified half. If half is None, the team who's kicking this drive.
        """
        if half is None:
            return self.state.kicking_this_drive
        return self.state.kicking_first_half if half == 1 else self.state.receiving_first_half

    def get_receiving_team(self, half: Optional[int] = None) -> Team:
        """
        :param half: Set this to None if you want the team who's receiving this drive.
        :return: The team who's receiving in the specified half. If half is None, the team who's receiving this drive.
        """
        if half is None:
            return self.state.receiving_this_drive
        return self.state.receiving_first_half if half == 1 else self.state.kicking_first_half

    def has_ball(self, player: Player) -> bool:
        """
        :param player:
        :return: True if player has the ball.
        """
        ball = self.get_ball_at(player.position)
        return ball is not None and ball.is_carried

    def get_ball(self) -> Optional[Ball]:
        """
        :return: A ball on the pitch or None.
        """
        for ball in self.state.pitch.balls:
            return ball

    def is_touchdown(self, player: Player) -> bool:
        """
        :param player:
        :return: True if player is in the opponent's endzone with the ball.
        """
        return self.arena.is_in_opp_endzone(player.position, player.team == self.state.home_team) and \
               self.has_ball(player)

    def is_blitz_available(self) -> bool:
        """
        :return: True if the current team can make a blitz this turn.
        """
        turn = self.current_turn()
        if turn is not None:
            return turn.blitz_available

    def use_blitz_action(self) -> None:
        """
        Uses this turn's blitz action.
        """
        turn = self.current_turn()
        if turn is not None:
            turn.blitz_available = False

    def unuse_blitz_action(self) -> None:
        """
        Unuses this turn's blitz action.
        """
        turn = self.current_turn()
        if turn is not None:
            turn.blitz_available = True

    def is_pass_available(self) -> bool:
        """
        :return: True if the current team can make a pass this turn.
        """
        turn = self.current_turn()
        if turn is not None:
            return turn.pass_available

    def use_pass_action(self) -> None:
        """
        Use this turn's pass action.
        """
        turn = self.current_turn()
        if turn is not None:
            turn.pass_available = False

    def unuse_pass_action(self) -> None:
        """
        Use this turn's pass action.
        """
        turn = self.current_turn()
        if turn is not None:
            turn.pass_available = True

    def is_handoff_available(self) -> bool:
        """
        :return: True if the current team can make a handoff this turn.
        """
        turn = self.current_turn()
        if turn is not None:
            return turn.handoff_available

    def use_handoff_action(self) -> None:
        """
        Uses this turn's handoff action.
        """
        turn = self.current_turn()
        if turn is not None:
            turn.handoff_available = False

    def unuse_handoff_action(self) -> None:
        """
        Uses this turn's handoff action.
        """
        turn = self.current_turn()
        if turn is not None:
            turn.handoff_available = True

    def is_foul_available(self) -> bool:
        """
        :return: True if the current team can make a foul this turn.
        """
        turn = self.current_turn()
        if turn is not None:
            return turn.foul_available

    def use_foul_action(self) -> None:
        """
        Uses this turn's foul action.
        """
        turn = self.current_turn()
        if turn is not None:
            turn.foul_available = False

    def unuse_foul_action(self) -> None:
        """
        Uses this turn's foul action.
        """
        turn = self.current_turn()
        if turn is not None:
            turn.foul_available = True

    def is_blitz(self) -> bool:
        """
        :return: True if the current turn is a Blitz!
        """
        turn = self.current_turn()
        if turn is not None:
            return turn.blitz

    def is_quick_snap(self) -> bool:
        """
        :return: True if the current turn is a Quick Snap!
        """
        turn = self.current_turn()
        if turn is not None:
            return turn.quick_snap

    def get_players_on_pitch(self, team: Team = None, used=None, up=None) -> List[Player]:
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
                if player is not None and (team is None or player.team == team) and (
                        used is None or used == player.state.used) and (up is None or up == player.state.up):
                    players.append(player)
        return players

    def pitch_to_reserves(self, player: Player) -> None:
        """
        Moves player from the pitch to the reserves section in the dugout.
        :param player:
        """
        self.remove(player)
        self.get_reserves(player.team).append(player)
        player.state.used = False
        player.state.up = True

    def reserves_to_pitch(self, player: Player, position: Square) -> None:
        """
        Moves player from the reserves section in the dugout to the pitch.
        :param player:
        :param position: position on pitch to put player
        """
        self.get_reserves(player.team).remove(player)
        player_at = self.get_player_at(position)
        if player_at is not None:
            self.pitch_to_reserves(player_at)
        self.put(player, position)
        player.state.up = True

    def pitch_to_kod(self, player: Player) -> None:
        """
        Moves player from the pitch to the KO section in the dugout.
        :param player:
        """
        self.remove(player)
        self.get_knocked_out(player.team).append(player)
        player.state.knocked_out = True
        player.state.up = True

    def kod_to_reserves(self, player: Player) -> None:
        """
        Moves player from the KO section in the dugout to the pitch. This also resets the players knocked_out state.
        :param player:
        """
        self.get_knocked_out(player.team).remove(player)
        self.get_reserves(player.team).append(player)
        player.state.knocked_out = False
        player.state.up = True

    def pitch_to_casualties(self, player: Player) -> None:
        """
        Moves player from the pitch to the CAS section in the dugout and applies the casualty and effect to the player.
        :param player:
        """
        self.remove(player)
        player.state.up = True
        player.state.stunned = False
        self.get_casualties(player.team).append(player)

    def pitch_to_dungeon(self, player: Player) -> None:
        """
        Moves player from the pitch to the dungeon and ejects the player from the game.
        :param player:
        """
        self.remove(player)
        self.get_dungeon(player.team).append(player)
        player.state.ejected = True
        player.state.up = True

    def lift(self, player: Player) -> None:
        """
        Lifts player from the board. Call put_down(player) to set player down again.
        """
        assert not player.state.in_air
        player.state.in_air = True
        self.state.pitch.board[player.position.y][player.position.x] = None

    def put_down(self, player: Player) -> None:
        """
        Puts a player down on the board on the square it was hovering.
        """
        assert player.state.in_air
        player.state.in_air = False
        self.put(player, player.position)

    def put(self, piece: Union[Catchable, Player], position: Square) -> None:
        """
        Put a piece on or above a square.
        :param piece: Ball or player
        :param position: square to put the player on or above
        """
        piece.position = position
        if type(piece) is Player:
            if not piece.state.in_air:
                self.state.pitch.board[position.y][position.x] = piece

            log_entry = MovementStep(self.state.pitch.board if not piece.state.in_air else None, piece, position,
                                     put=True)
            self.trajectory.log_state_change(log_entry)
        elif type(piece) is Ball:
            piece: Ball
            self.state.pitch.balls.append(piece)
        elif type(piece) is Bomb:
            self.state.pitch.bomb = piece
        else:
            raise Exception("Unknown piece type")

    def remove(self, piece: Union[Catchable, Player]) -> None:
        """
        Remove a piece from the board.
        :param piece:
        """
        assert piece.position is not None
        if type(piece) is Player:
            if not piece.state.in_air:
                self.state.pitch.board[piece.position.y][piece.position.x] = None

            log_entry = MovementStep(self.state.pitch.board if not piece.state.in_air else None, piece, piece.position,
                                     put=False)
            self.trajectory.log_state_change(log_entry)

            piece.position = None
        elif type(piece) is Ball:
            piece: Ball
            self.state.pitch.balls.remove(piece)
        elif type(piece) is Bomb:
            self.state.pitch.bomb = None

    def move(self, piece: Union[Catchable, Player], position: Square) -> None:
        """
        Move a piece already on the board. If the piece is a ball carrier, the ball is moved as well.
        :param piece:
        :param position:
        :return:
        """
        if type(piece) is Player:
            assert self.get_player_at(position) is None or (type(piece) == Player and piece.state.in_air)
            for ball in self.state.pitch.balls:
                if ball.position == piece.position and ball.is_carried:
                    ball.move_to(position)
            self.remove(piece)
            self.put(piece, position)
        elif piece.is_catchable():
            piece.move_to(position)

    def shove(self, piece: Union[Catchable, Player], x: int, y: int) -> None:
        """
        Shove a push x number of step in the horizontal direction and y number of steps in the vertical direction.
        :param piece
        :param x
        :param y
        """
        self.move(piece, self.get_square(piece.position.x + x, piece.position.y + y))

    def swap(self, piece_a: Union[Catchable, Player], piece_b: Union[Catchable, Player]) -> None:
        """
        Swap two pieces on the board.
        :param piece_a:
        :param piece_b:
        :return:
        """
        assert piece_a.position is not None
        assert piece_b.position is not None
        pos_a = piece_a.position
        pos_b = piece_b.position
        piece_a.position = pos_b
        piece_b.position = pos_a
        if type(piece_b) is Player:
            ball = self.get_ball_at(pos_b)
            if ball is not None:
                self.move(ball, pos_a)
            self.state.pitch.board[pos_a.y][pos_a.x] = piece_b
        elif type(piece_b) is Catchable:
            piece_b.move_to(pos_a)
        if type(piece_a) is Player:
            ball = self.get_ball_at(pos_a)
            if ball is not None:
                self.move(ball, pos_b)
            self.state.pitch.board[pos_b.y][pos_b.x] = piece_a
        elif type(piece_b) is Catchable:
            piece_a.move_to(pos_b)

    def get_catch_modifiers(self, catcher: Player, accurate: bool = False, interception: bool = False,
                            handoff: bool = False) -> int:
        """
        :param catcher:
        :param accurate: whether it is an accurate pass.
        :param interception: whether it is an interception catch.
        :param handoff: whether it is a handoff catch.
        :return: the modifier to be added to the pass roll.
        """
        modifiers = 1 if accurate or handoff else 0
        if catcher.has_skill(Skill.DIVING_CATCH) and accurate:
            modifiers += 1
        modifiers = -2 if interception else modifiers
        if interception and catcher.has_skill(Skill.VERY_LONG_LEGS):
            modifiers += 1
        # opposing tackle zones
        if not catcher.has_skill(Skill.NERVES_OF_STEEL):
            modifiers -= self.num_tackle_zones_in(catcher)
        if self.state.weather == WeatherType.POURING_RAIN:
            modifiers -= 1
        if catcher.has_skill(Skill.EXTRA_ARMS):
            modifiers += 1
        # Disturbing presence
        for opp_player in self.get_opp_team(catcher.team).players:
            if opp_player.has_skill(Skill.DISTURBING_PRESENCE) and opp_player.position and opp_player.position.distance(catcher.position) <= 3:
                modifiers -= 1
        return modifiers

    def get_pass_modifiers(self, passer: Player, pass_distance: PassDistance, ttm: bool = False) -> int:
        """
        :param passer:
        :param pass_distance: the PassDistance to the target.
        :param ttm: Throwing a team-mate?
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

        if ttm:
            modifiers -= 1

        if passer.has_skill(Skill.STRONG_ARM):
            if pass_distance == PassDistance.SHORT_PASS or pass_distance == PassDistance.LONG_PASS or \
                    pass_distance == PassDistance.LONG_BOMB:
                modifiers += 1

        if passer.has_skill(Skill.STUNTY):
            modifiers -= 1

        # Disturbing presence
        for opp_player in self.get_opp_team(passer.team).players:
            if opp_player.has_skill(Skill.DISTURBING_PRESENCE) and opp_player.position and opp_player.position.distance(passer.position) <= 3:
                modifiers -= 1

        return modifiers

    def get_leap_modifiers(self, player: Player) -> int:
        """
        :param player:
        :return: the modifier to be added to the leap roll.
        """
        return 1 if player.has_skill(Skill.VERY_LONG_LEGS) else 0

    def get_dodge_modifiers(self, player: Player, position: Square, include_diving_tackle: bool = False) -> int:
        """
        :param player:
        :param position: The position the player is dodging to
        :param include_diving_tackle:
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
        modifiers -= len(prehensile_tailers)  # subtract 1 for each prehensile tail detractor

        if not ignore_opp_mods:
            modifiers -= tackle_zones_to

        if include_diving_tackle:
            diving_tacklers = self.get_adjacent_opponents(player, skill=Skill.DIVING_TACKLE)
            if len(diving_tacklers) > 0:
                modifiers -= 2

        return modifiers

    def get_pickup_modifiers(self, player: Player, position: Square) -> int:
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

    def num_tackle_zones_in(self, player: Player) -> int:
        """
        :param player:
        :return: Number of opponent tackle zones the player is in.
        """
        return self.num_tackle_zones_at(player, player.position)

    def num_tackle_zones_at(self, player: Player, position: Square) -> int:
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

    def get_catcher(self, position: Square) -> Optional[Player]:
        """
        :param position: A square on the board
        :return: A player if the ball can be catched by one at the given square, otherwise None.
        """
        catcher = self.get_player_at(position)
        if catcher is not None:
            return catcher
        diving_catchers = self.get_adjacent_players(position, team=None, down=False, skill=Skill.DIVING_CATCH)
        if len(diving_catchers) == 1:
            return diving_catchers[0]
        else:
            return None

    def is_setup_legal(self, team: Team) -> bool:
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

    def is_setup_legal_count(self, team: Team, tile=None, max_players=11, min_players=3) -> bool:
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
                if not self.is_team_side(self.get_square(x, y), team):
                    continue
                if tile is None or self.arena.board[y][x] == tile:
                    piece = self.state.pitch.board[y][x]
                    if isinstance(piece, Player) and piece.team == team:
                        cnt += 1
        if cnt > max_players or cnt < min_players_checked:
            return False
        return True

    def num_casualties(self, team: Team = None) -> int:
        """
        :param team: If None, return the sum of both teams casualties.
        :return: The number of casualties suffered by team.
        """
        if team is not None:
            return len(self.get_casualties(team))
        else:
            return len(self.get_casualties(self.state.home_team)) + len(self.get_casualties(self.state.away_team))

    def get_winning_team(self) -> Optional[Team]:
        """
        :return: The team with most touchdowns, otherwise None.
        """
        if self.state.home_team.state.score > self.state.away_team.state.score:
            return self.state.home_team
        elif self.state.home_team.state.score < self.state.away_team.state.score:
            return self.state.away_team
        return None

    def is_setup_legal_scrimmage(self, team: Team, min_players=3) -> bool:
        """
        :param team:
        :param min_players:
        :return: True if team is setup legally on scrimmage.
        """
        if team == self.state.home_team:
            return self.is_setup_legal_count(team, tile=Tile.HOME_SCRIMMAGE, min_players=min_players)
        return self.is_setup_legal_count(team, tile=Tile.AWAY_SCRIMMAGE, min_players=min_players)

    def is_setup_legal_wings(self, team: Team, min_players=0, max_players=2) -> bool:
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

    def get_procedure_names(self) -> List[str]:
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

    def get_player_action_type(self) -> Optional[PlayerActionType]:
        """
        :return: the player PlayerActionType if there is any on the stack.
        """
        return self.state.player_action_type

    def remove_recursive_refs(self) -> None:
        """
        Removes recursive references. Must be called before serializing.
        """
        for team in self.state.teams:
            for player in team.players:
                player.team = None

    def add_recursive_refs(self) -> None:
        """
        Adds recursive references. Can be called after serializing.
        """
        for team in self.state.teams:
            for player in team.players:
                player.team = team

    #def get_termination_time(self):
    #    """
    #    The time at which the current turn must be terminated - or the opponent's action choice (like selecting block die).
    #    """
    #    if self.state.termination_opp is not None:
    #        return self.state.termination_opp
    #    return self.state.termination_turn

    def get_team_by_id(self, team_id) -> Optional[Team]:
        """
        :param team_id:
        :return: returns the team with the id or None
        """
        if self.state.home_team.team_id == team_id:
            return self.state.home_team
        if self.state.away_team.team_id == team_id:
            return self.state.away_team
        return None

    def get_winner(self) -> Optional[Agent]:
        """
        returns the winning agent of the game. None if it's a draw.
        If the game timed out the current player loses.
        A disqualified player will lose.
        If the game is over, the team with most TDs win.
        """
        # If game timed out the current player lost
        '''
        if self.home_agent == self.actor:
            return self.away_agent
        elif self.away_agent == self.actor:
            return self.home_agent
        '''
        # If the game is over the player with most TDs wins
        if self.state.game_over:
            return self.get_team_agent(self.get_winning_team())

        return None

    def get_other_agent(self, agent: Agent) -> Optional[Agent]:
        """
        Returns the other agent in the game.
        """
        if agent is None:
            return None
        if agent == self.home_agent:
            return self.away_agent
        return self.home_agent

    def get_other_active_player_id(self) -> Optional[str]:
        """
        Returns the player id of the other player involved in current procedures - if any.
        """
        for proc in self.state.stack.items:
            if isinstance(proc, Block):
                if proc.defender is not None:
                    return proc.defender.player_id
            if isinstance(proc, PassAttempt):
                if proc.catcher is not None:
                    return proc.catcher.player_id
            if isinstance(proc, Handoff):
                if proc.catcher is not None:
                    return proc.catcher.player_id
            if isinstance(proc, Push):
                if proc.player is not None:
                    return proc.player.player_id
            if isinstance(proc, PassAction):
                if proc.picked_up_teammate is not None:
                    return proc.picked_up_teammate.player_id
        return None

    def replace_home_agent(self, agent: Agent) -> None:
        """
        Replaces the home agent safely.
        :param agent:
        """
        self.home_agent = agent

    def replace_away_agent(self, agent: Agent) -> None:
        """
        Replaces the away agent safely.
        :param agent:
        """
        self.away_agent = agent

    def has_report_of_type(self, outcome_type, last=None) -> bool:
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

    def get_balls_at(self, position: Square, in_air=False) -> List[Ball]:
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

    def get_ball_at(self, position: Square, in_air=False) -> Optional[Ball]:
        """
        Assumes there is only one ball on the square.
        :param position:
        :param in_air:
        :return: Ball or None
        """
        balls_at = self.get_balls_at(position, in_air)
        return balls_at[0] if balls_at else None

    def get_bomb(self) -> Optional[Bomb]:
        """
        Returns a bomb or None.
        :return: Bomb or None
        """
        return self.state.pitch.bomb

    def remove_bomb(self) -> None:
        """
        Removes the bombe from the pitch.
        """
        self.state.pitch.bomb = None

    def put_bomb(self, bomb) -> None:
        """
        Adds a bomb to the pitch.
        """
        assert self.state.pitch.bomb is None
        self.state.pitch.bomb = bomb

    def get_ball_positions(self) -> List[Square]:
        """
        :return: The position of the ball. If no balls are in the arena None is returned. If multiple balls are in the
        arena, the position of the first ball is return.
        """
        return [ball.position for ball in self.state.pitch.balls]

    def get_ball_position(self) -> Optional[Square]:
        """
        Assumes there is only one ball on the square
        :return: Ball or None
        """
        for ball in self.state.pitch.balls:
            return ball.position
        return None

    def get_ball_carrier(self) -> Optional[Player]:
        """
        :return: the ball carrier if any - otherwise None.
        """
        ball_position: Square = self.get_ball_position()
        if ball_position is None:
            return None
        else:
            return self.get_player_at(ball_position)

    def is_out_of_bounds(self, position: Square) -> bool:
        """
        :param position:
        :return: True if pos is out of bounds.
        """
        return position.x < 1 or position.x >= self.state.pitch.width - 1 or \
               position.y < 1 or position.y >= self.state.pitch.height - 1

    def get_push_squares(self, from_position: Square, to_position: Square) -> List[Square]:
        """
        :param from_position: The position of the attacker.
        :param to_position: The position of the defender.
        :return: Possible square to push the player standing on pos_to on to.
        """
        attacker = self.get_player_at(from_position)
        defender = self.get_player_at(to_position)
        if defender.has_skill(Skill.SIDE_STEP) and not attacker.has_skill(Skill.GRAB):
            return self.get_adjacent_squares(to_position, out=True, occupied=False)
        squares_to = self.get_adjacent_squares(to_position, out=True)
        squares_empty = []
        squares_out = []
        squares = []
        for square in squares_to:
            include = False
            if from_position.x == to_position.x or from_position.y == to_position.y:
                if from_position.distance(square, manhattan=False) >= 2:
                    include = True
            else:
                if from_position.distance(square, manhattan=True) >= 3:
                    include = True
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

    def get_square(self, x, y) -> Square:
        """
        Returns an existing square object for the given position to avoid a new instantiation. If the square object
        is out of bounds it may be instantiated.
        :param x:
        :param y:
        :return: A square with the position (x,y)
        """
        if not 0 >= y < len(self.square_shortcut):
            return Square(x, y)
        if not 0 >= x < len(self.square_shortcut[y]):
            return Square(x, y)
        return self.square_shortcut[y][x]

    def get_adjacent_squares(self, position: Square, diagonal=True, out=False, occupied=True, distance=1) \
            -> List[Square]:
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
        r = range(-distance, distance + 1)
        for yy in r:
            for xx in r:
                if yy == 0 and xx == 0:
                    continue
                sq = self.get_square(position.x + xx, position.y + yy)
                if not out and self.is_out_of_bounds(sq):
                    continue
                if not occupied and self.get_player_at(sq) is not None:
                    continue
                if diagonal:
                    squares.append(sq)
                elif xx == 0 or yy == 0:
                    squares.append(sq)
        return squares

    def get_adjacent_opponents(self, player: Player, diagonal=True, down=True, standing=True, stunned=True, skill=None) -> List[Player]:
        """
        Returns a list of adjacent opponents to the player it its current position.
        :param player:
        :param diagonal: Whether to include diagonally adjacent players.
        :param down: Whether to include down players.
        :param standing: Whether to include standing players.
        :param stunned: Whether to include stunned players.
        :param skill: Only include players with this skill.
        :return:
        """
        return self.get_adjacent_players(player.position, self.get_opp_team(player.team), diagonal, down, standing,
                                         stunned, skill=skill)

    def get_adjacent_teammates(self, player: Player, diagonal=True, down=True, standing=True, stunned=True, skill=None) -> List[Player]:
        """
        Returns a list of adjacent teammates to the player it its current position.
        :param player:
        :param diagonal: Whether to include diagonally adjacent players.
        :param down: Whether to include down players.
        :param standing: Whether to include standing players.
        :param stunned: Whether to include stunned players.
        :param skill: Only include players with this skill.
        :return:
        """
        return self.get_adjacent_players(player.position, player.team, diagonal, down, standing, stunned, skill=skill)

    def get_adjacent_players(self, position: Square, team: Team=None, diagonal=True, down=True, standing=True, stunned=True,
                             skill=None) -> List[Player]:
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

    def get_assisting_players(self, player: Player, opp_player: Player, foul=False) -> List[Player]:
        """
        :param player:
        :param opp_player:
        :param foul: Indicates whether it is a foul. The Guard skill is ignored if fould=True.
        :return: a list of assisting players in a block between player and opp_player.
        """
        assists = []
        for yy in range(-1, 2, 1):
            for xx in range(-1, 2, 1):
                if yy == 0 and xx == 0:
                    continue
                p = self.get_square(opp_player.position.x + xx, opp_player.position.y + yy)
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

    def can_assist(self, player: Player, foul: bool = False) -> bool:
        """
        :param player: The player which potentially can assist
        :param foul:
        :return:
        """
        if not player.can_assist():
            return False
        if (not foul and player.has_skill(Skill.GUARD)) or \
                self.num_tackle_zones_in(player) <= 1:
            return True
        return False

    def get_assisting_players_at(self, player: Player, opp_player: Player, foul: bool=False) -> List[Player]:
        """
        :param player:
        :param opp_player:
        :param foul: Indicates whether it is a foul. The Guard skill is ignored if fould=True.
        :return: a list of assisting players in a block between player and opp_player.
        """
        assists = []
        for yy in range(-1, 2, 1):
            for xx in range(-1, 2, 1):
                if yy == 0 and xx == 0:
                    continue
                p = self.get_square(opp_player.position.x + xx, opp_player.position.y + yy)
                if not self.is_out_of_bounds(p) and player.position != p:
                    player_at = self.get_player_at(p)
                    if player_at is not None:
                        if player_at.team == player.team and self.can_assist(player_at, foul):
                            assists.append(player_at)
        return assists

    def get_block_strengths(self, attacker: Player, defender: Player, blitz: bool = False) -> Tuple[int, int]:
        """
        :param attacker:
        :param defender:
        :return: a tuple containing the attacker and defenders strengths during a block including assists.
        """
        assert attacker.position.distance(defender.position) == 1
        attacker_strength = attacker.get_st()
        defender_strength = defender.get_st()
        if blitz and attacker.has_skill(Skill.HORNS):
            attacker_strength += 1
        attacker_strength += len(self.get_assisting_players(attacker, defender))
        defender_strength += len(self.get_assisting_players(defender, attacker))
        return attacker_strength, defender_strength

    def num_block_dice(self, attacker: Player, defender: Player, blitz: bool = False, dauntless_success: bool = False) -> int:
        """
        :param attacker: 
        :param defender: 
        :param blitz: if it is a blitz
        :param dauntless_success: If a dauntless rolls was successful.
        :return: The number of block dice used in a block between the attacker and defender.
        """
        return self.num_block_dice_at(attacker, defender, attacker.position, blitz, dauntless_success)

    def get_block_probs(self, attacker: Player, defender: Player) -> Tuple[float, float, float, float]:
        """
        :param attacker:
        :param defender:
        :return: a tuple containing the knock-down probabilities of the attacker and defender.
        """
        dice = self.num_block_dice(attacker, defender)
        push_squares = self.get_push_squares(attacker.position, defender.position)
        crowd_push = self.arena.board[push_squares[0].y][push_squares[0].x] == Tile.CROWD and not defender.has_skill(Skill.STAND_FIRM)
        p_self = 1.0 / 6.0 if attacker.has_skill(Skill.BLOCK) else 2.0 / 6.0
        p_opp = 2.0 / 6.0 if attacker.has_skill(Skill.BLOCK) else 2.0 / 6.0
        if crowd_push:
            p_opp += 2.0 / 6.0
        if not crowd_push:
            p_opp -= (1.0 / 6.0 if defender.has_skill(Skill.DODGE) and not attacker.has_skill(Skill.TACKLE) else 0.0)
        if dice == 2:
            p_self -= (1.0 - p_self) * p_self
            p_opp += (1.0 - p_opp) * p_opp
        if dice == 3:
            p_self -= (1.0 - p_self) * p_self
            p_opp += (1.0 - p_opp) * p_opp
        if dice == -2:
            p_self += (1.0 - p_self) * p_self
            p_opp -= (1.0 - p_opp) * p_opp
        if dice == -3:
            p_self += (1.0 - p_self) * p_self
            p_opp -= (1.0 - p_opp) * p_opp
        p_fumble_opp = 0.0
        p_fumble_self = 0.0
        if self.get_ball_carrier() == defender:
            p_fumble_opp = p_opp
            if not crowd_push and attacker.has_skill(Skill.STRIP_BALL) and not defender.has_skill(Skill.SURE_HANDS):
                p_fumble_opp += 2.0 / 6.0
        elif self.get_ball_carrier() == attacker:
            p_fumble_self = p_self
        return p_self, p_opp, p_fumble_self, p_fumble_opp

    def get_blitz_probs(self, attacker: Player, attack_position: Square, defender: Player) -> Tuple[float, float, float, float]:
        """
        :param attacker:
        :param attack_position:
        :param defender:
        :return: a tuple containing the knock-down probabilities of the attacker and defender given that attacker
        blitzes from attack_position.
        """
        orig_position = self.get_square(attacker.position.x, attacker.position.y)
        if attacker.position != attack_position:
            self.move(attacker, attack_position)
        p_self, p_opp, p_fumble_self, p_fumble_opp = self.get_block_probs(attacker, defender)
        if attacker.position != orig_position:
            self.move(attacker, orig_position)
        return p_self, p_opp, p_fumble_self, p_fumble_opp

    def get_dodge_prob(self, player: Player, position: Square, allow_dodge_reroll: bool=True, allow_team_reroll: bool=False) -> float:
        """
        :param player:
        :param position:
        :param allow_dodge_reroll:
        :param allow_team_reroll:
        :return: the probability of a successful dodge for player to position.
        """
        if self.num_tackle_zones_in(player) == 0:
            return 1.0
        ag_roll = Rules.agility_table[player.get_ag()] - self.get_dodge_modifiers(player, position)
        ag_roll = max(2, min(6, ag_roll))
        successful_outcomes = 6 - (ag_roll - 1)
        p = successful_outcomes / 6.0
        if allow_dodge_reroll and player.has_skill(Skill.DODGE) and not self.get_adjacent_opponents(player, down=False, skill=Skill.TACKLE):
            p += (1.0-p)*p
        elif allow_team_reroll and self.can_use_reroll(player.team):
            p += (1.0 - p) * p
        return p

    def get_catch_prob(self, player: Player, accurate: bool=False, interception: bool=False, handoff: bool=False, allow_catch_reroll: bool=True,
                       allow_team_reroll: bool=False) -> float:
        """
        :param player:
        :param accurate: whether it is an accurate pass
        :param interception: whether it is an interception attempt
        :param handoff: whether it is a handoff
        :param allow_catch_reroll:
        :param allow_team_reroll:
        :return: the probability of a successful catch for player.
        """
        ag_roll = Rules.agility_table[player.get_ag()] - self.get_catch_modifiers(player, accurate=accurate, interception=interception, handoff=handoff)
        ag_roll = max(2, min(6, ag_roll))
        successful_outcomes = 6 - (ag_roll - 1)
        p = successful_outcomes / 6.0
        if allow_catch_reroll and player.has_skill(Skill.CATCH):
            p += (1.0 - p) * p
        elif allow_team_reroll and self.can_use_reroll(player.team):
            p += (1.0 - p) * p
        return p

    def get_dodge_prob_from(self, player: Player, from_position: Square, to_position: Square,
                            allow_dodge_reroll: bool=False, allow_team_reroll: bool=False) -> float:
        """
        :param player:
        :param from_position:
        :param to_position
        :param allow_dodge_reroll:
        :param allow_team_reroll:
        :return: the probability of a successful dodge for player from from_position to to_position.
        """
        orig_position = self.get_square(player.position.x, player.position.y)
        self.move(player, from_position)
        p = self.get_dodge_prob(player, to_position, allow_dodge_reroll, allow_team_reroll)
        self.move(player, orig_position)
        return p

    def get_pickup_prob(self, player: Player, position: Square, allow_pickup_reroll: bool=True,
                        allow_team_reroll: bool=False) -> float:
        """
        :param player:
        :param position: the position of the ball
        :param allow_pickup_reroll:
        :param allow_team_reroll:
        :return: the probability of a successful catch for player.
        """
        ag_roll = Rules.agility_table[player.get_ag()] - self.get_pickup_modifiers(player, position=position)
        ag_roll = max(2, min(6, ag_roll))
        successful_outcomes = 6 - (ag_roll - 1)
        p = successful_outcomes / 6.0
        if allow_pickup_reroll and player.has_skill(Skill.SURE_HANDS):
            p += (1.0 - p) * p
        elif allow_team_reroll and self.can_use_reroll(player.team):
            p += (1.0 - p) * p
        return p

    def get_pass_prob(self, player: Player, piece: Piece, position: Square,
                      allow_pass_reroll: bool = True, allow_team_reroll: bool = False) -> float:
        """
        :param player: passer
        :param piece: piece to pass
        :param position: the position of the ball
        :param allow_pass_reroll:
        :param allow_team_reroll:
        :return: the probability of a successful catch for player.
        """
        distance = self.get_pass_distance(from_position=player.position, to_position=position)
        ttm = type(piece) != Ball
        if ttm:
            assert distance in {PassDistance.SHORT_PASS, PassDistance.QUICK_PASS}, "Throw team mate distance is too far"

        modifiers = self.get_pass_modifiers(player, pass_distance=distance, ttm=ttm)
        ag_roll = Rules.agility_table[player.get_ag()] - modifiers
        ag_roll = max(2, min(6, ag_roll))
        successful_outcomes = 6 - (ag_roll - 1)
        p = successful_outcomes / 6.0
        if allow_pass_reroll and player.has_skill(Skill.Pass):
            p += (1.0 - p) * p
        elif allow_team_reroll and self.can_use_reroll(player.team):
            p += (1.0 - p) * p
        return p

    def num_block_dice_at(self, attacker, defender, position: Square, blitz: bool=False, dauntless_success: bool=False):
        """
        :param attacker:
        :param defender:
        :param position: attackers position
        :param blitz: if it is a blitz
        :param dauntless_success: If a dauntless rolls was successful.
        :return: The number of block dice used in a block between the attacker and defender if the attacker block at
                 the given position.
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
        elif st_for * 2 < st_against:
            return -3
        elif st_for < st_against:
            return -2

    def num_assists_at(self, attacker: Player, defender: Player, position: Square, foul: bool = False) \
            -> Tuple[int, int]:
        '''
        Return net assists for a block of player on opp_player when player has moved to position first.  Required for
        calculating assists after moving in a Blitz action.
        :return: int - Net # of assists
        '''

        # Note that because blitzing/fouling player may have moved,
        # calculating assists for is slightly different to against.
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
                    # Need to make sure we take into account the blocking/blitzing player may be in a different square
                    # than currently represented on the board.
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
        return n_assists_for, n_assist_against

    def get_pass_distances(self, passer: Player, piece: Piece, dump_off: bool = False) -> Tuple[List[Square], List[PassDistance]]:
        """
        :return: two lists (squares, distances) indicating the PassDistance to each square that the passer can pass to.
        """
        return self.get_pass_distances_at(passer, piece, passer.position, dump_off=dump_off)

    def get_pass_distances_at(self, passer: Player, piece: Piece, position: Square, dump_off: bool = False) \
            -> Tuple[List[Square], List[PassDistance]]:
        """
        :param passer:
        :param piece:
        :param position:
        :param dump_off:
        :return: two lists (squares, distances) indicating the PassDistance to each square that the passer can pass to
                 if at the given position.
        """
        squares = []
        distances = []
        if dump_off:
            distances_allowed = [PassDistance.QUICK_PASS]
        elif type(piece) == Player or self.state.weather == WeatherType.BLIZZARD:
            distances_allowed = [PassDistance.QUICK_PASS, PassDistance.SHORT_PASS]
        else:
            distances_allowed = [PassDistance.QUICK_PASS,
                                 PassDistance.SHORT_PASS,
                                 PassDistance.LONG_PASS,
                                 PassDistance.LONG_BOMB,
                                 PassDistance.HAIL_MARY] if Skill.HAIL_MARY_PASS in passer.get_skills() \
                else [PassDistance.QUICK_PASS, PassDistance.SHORT_PASS, PassDistance.LONG_PASS, PassDistance.LONG_BOMB]
        for y in range(len(self.state.pitch.board)):
            for x in range(len(self.state.pitch.board[y])):
                to_position = self.get_square(x, y)
                if self.is_out_of_bounds(to_position) or position == to_position:
                    continue
                distance = self.get_pass_distance(position, to_position)
                if distance in distances_allowed:
                    squares.append(to_position)
                    distances.append(distance)
        return squares, distances

    def get_pass_distance(self, from_position: Square, to_position: Square) -> PassDistance:
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

    def get_distance_to_endzone(self, player: Player) -> int:
        """
        :param player:
        :return: direct distance to the nearest opponent endzone tile.
        """
        assert player.position is not None
        x = self.get_opp_endzone_x(player.team)
        return abs(x - player.position.x)

    def get_opp_endzone_x(self, team: Team) -> int:
        """
        :param team:
        :return: the x-coordinate of the opponents endzone
        """
        if team == self.state.home_team:
            return 1
        else:
            return self.arena.width - 2

    def get_interceptors(self, position_from: Square, position_to: Square, team: Team) -> List[Player]:
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
        :param team: team that can attempt interception
        """

        # 1) Find line x from a to b
        x = get_line((position_from.x, position_from.y), (position_to.x, position_to.y))

        # 3) Find squares s where x intersects
        s = []
        for i in x:
            s.append(self.get_square(i[0], i[1]))

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

    def get_available_actions(self) -> List[ActionChoice]:
        """
        :return: a list of available action choices in the current state.
        """
        return self.state.available_actions

    def clear_board(self) -> None:
        """
        Moves all players from the board to their respective reserves box.
        """
        for player in self.get_players_on_pitch(self.state.home_team):
            self.pitch_to_reserves(player)
        for player in self.get_players_on_pitch(self.state.away_team):
            self.pitch_to_reserves(player)

    def get_active_player(self) -> Optional[Player]:
        """
        :return: the current player to make a move if any, else None.
        """
        return self.state.active_player

    def get_procedure(self) -> Procedure:
        """
        :return: The current procedure on the top of the stack.
        """
        return self.state.stack.peek()

    def get_weather(self) -> WeatherType:
        """
        :return: The current weather.
        """
        return self.state.weather

    def apply_casualty(self, player: Player, inflictor: Player, casualty, effect: CasualtyEffect, roll: DiceRoll) \
            -> None:
        """
        Applies a casualty to a player and moves it to the dugout.
        :param player: the player to apply the casualty to.
        :param inflictor: the player who inflicted the casuality - can be None.
        :param casualty: the Casualty to apply.
        :param effect: The CasualtyEffect to apply.
        :param roll: The casualty roll that caused the casualty.
        """
        # Move to casualty box
        if player.position is not None:
            self.pitch_to_casualties(player)
        # Report effect and MNG
        if effect == CasualtyEffect.NONE:
            self.report(Outcome(OutcomeType.BADLY_HURT, player=player, opp_player=inflictor, team=player.team,
                                rolls=[roll]))
        elif effect in Rules.miss_next_game:
            if effect not in player.state.injuries_gained and CasualtyEffect.MNG not in player.state.injuries_gained:
                player.state.injuries_gained.append(CasualtyEffect.MNG)
            self.report(Outcome(OutcomeType.MISS_NEXT_GAME, player=player, opp_player=inflictor, team=player.team,
                                rolls=[roll], n=effect.name))
        elif effect == CasualtyEffect.DEAD:
            self.report(Outcome(OutcomeType.DEAD, player=player, opp_player=inflictor, team=player.team,
                                rolls=[roll]))
        # Add injuries
        if effect is not CasualtyEffect.MNG and effect is not CasualtyEffect.NONE:
            player.state.injuries_gained.append(effect)

    def get_current_turn_proc(self) -> Optional[Procedure]:
        """
        :return: the Turn procedure that is highest on the stack.
        """
        for i in range(len(self.state.stack.items)):
            idx = len(self.state.stack.items) - 1 - i
            if type(self.state.stack.items[idx]) == Turn:
                return self.state.stack.items[idx]
        return None

    def get_tile(self, position: Square) -> Tile:
        """
        :param position: a Square on the board.
        :return: the tile type at the given position.
        """
        return self.arena.board[position.y][position.x]

    def get_adjacent_blood_lust_victims(self, player: Player)  -> List[Square]:
        """
        :param player: a player on the board.
        :return: return positions of adjecent players that can be bitten
                 because of a failed blood lust roll. 
        """
        return [p.position for p in self.get_adjacent_teammates(player) if not p.has_skill(Skill.BLOOD_LUST)]

    def get_hypno_targets(self, player: Player) -> List[Square]:
        """
        :param player: player on the board. 
        :return: available targets for given player to hypnotize if player has Hypnotic Gaze skill 
        """

        if not player.has_skill(Skill.HYPNOTIC_GAZE):
            return []

        return [o.position for o in self.get_adjacent_opponents(player, down=False) if o.has_tackle_zone()]

    def get_hypno_modifier(self, player: Player) -> int:
        """
        :param player: player on the board with hypnotic gaze skill. 
        :return:  modifier for player to hypnotize target. 
        """
        return 1 - self.num_tackle_zones_in(player)

    def get_landing_modifiers(self, player: Player) -> int:
        """
        :param player: Player attempting to land.
        """
        return self.num_tackle_zones_in(player)

    def get_handoff_actions(self, player: Player) -> List[ActionChoice]:
        """
        :param player: Hand-offing player
        :return: Available hand-off actions for the player.
        """
        actions = []
        hand_off_positions = []
        rolls = []
        for player_to in self.get_adjacent_teammates(player):
            if player_to.can_catch():
                hand_off_positions.append(player_to.position)
                modifiers = self.get_catch_modifiers(player_to, handoff=True)
                target = Rules.agility_table[player.get_ag()]
                rolls.append([min(6, max(2, target - modifiers))])

        if len(hand_off_positions) > 0:
            actions.append(ActionChoice(ActionType.HANDOFF, team=player.team,
                                        positions=hand_off_positions, rolls=rolls))
        return actions

    def get_stand_up_actions(self, player: Player) -> List[ActionChoice]:
        rolls = []
        if not player.state.up:
            moves = 0 if player.has_skill(Skill.JUMP_UP) else 3
            if player.get_ma() < moves:
                stand_up_roll = max(2, min(6, 4 - self.get_stand_up_modifier(player)))
                rolls.append([stand_up_roll])
            else:
                rolls.append([])
            return [ActionChoice(ActionType.STAND_UP, team=player.team, rolls=rolls)]
        return []

    def get_adjacent_move_actions(self, player: Player) -> List[ActionChoice]:
        quick_snap = self.is_quick_snap()
        actions = []
        move_positions = []
        d6_rolls = []
        move_needed = 1 if not player.state.up else 1
        gfi = player.state.moves + move_needed > player.get_ma()
        sprints = 3 if player.has_skill(Skill.SPRINT) else 2
        gfi_roll = 3 if self.state.weather == WeatherType.BLIZZARD else 2
        if (not quick_snap
            and not player.state.taken_root
            and player.state.moves + move_needed <= player.get_ma() + sprints) \
                or (quick_snap and player.state.moves == 0):
            # Regular movement
            for square in self.get_adjacent_squares(player.position, occupied=False):
                ball_at = self.get_ball_at(square)
                move_positions.append(square)
                rolls = []
                if not quick_snap:
                    if gfi:
                        rolls.append(gfi_roll)
                    if self.num_tackle_zones_in(player) > 0:
                        modifiers = self.get_dodge_modifiers(player, square)
                        target = Rules.agility_table[player.get_ag()]
                        rolls.append(min(6, max(2, target - modifiers)))
                    if ball_at is not None and ball_at.on_ground:
                        target = Rules.agility_table[player.get_ag()]
                        modifiers = self.get_pickup_modifiers(player, square)
                        rolls.append(min(6, max(2, target - modifiers)))
                d6_rolls.append(rolls)
            if len(move_positions) > 0:
                actions.append(ActionChoice(ActionType.MOVE, team=player.team,
                                            positions=move_positions, rolls=d6_rolls))

        return actions

    def get_leap_actions(self, player: Player) -> List[ActionChoice]:
        actions = []
        if player.can_use_skill(Skill.LEAP) and not self.is_quick_snap():
            sprints = 3 if player.has_skill(Skill.SPRINT) else 2
            gfi_roll = 3 if self.state.weather == WeatherType.BLIZZARD else 2
            leap_rolls = []
            leap_positions = []
            modifiers = 0 if player.has_skill(Skill.VERY_LONG_LEGS) else 0
            target = Rules.agility_table[player.get_ag()]
            leap_roll = min(6, max(2, target - modifiers))
            for square in self.get_adjacent_squares(player.position, occupied=False, distance=2):
                distance = player.position.distance(square)
                if player.state.moves + distance <= player.get_ma() + sprints:
                    rolls = []
                    leap_positions.append(square)
                    gfis = max(0, (player.state.moves + distance) - player.get_ma())
                    for gfi in range(gfis):
                        rolls.append(gfi_roll)
                    rolls.append(leap_roll)
                    ball_at = self.get_ball_at(square)
                    if ball_at is not None and ball_at.on_ground:
                        modifiers = self.get_pickup_modifiers(player, square)
                        rolls.append(min(6, max(2, target - modifiers)))
                    leap_rolls.append(rolls)
            if len(leap_positions) > 0:
                actions.append(ActionChoice(ActionType.LEAP, team=player.team,
                                            positions=leap_positions, rolls=leap_rolls))
        return actions

    def get_foul_actions(self, player: Player) -> List[ActionChoice]:
        """
        :param player: Fouling player
        :return: Available foul actions for the player.
        """
        actions = []
        foul_positions = []
        foul_rolls = []
        for player_to in self.get_adjacent_opponents(player, standing=False, down=True):
            foul_positions.append(player_to.position)
            assists_from = self.get_assisting_players(player, player_to, foul=True)
            assists_to = self.get_assisting_players(player_to, player, foul=True)
            target = min(12, max(2, player_to.get_av() + 1 - len(assists_from) + len(assists_to)))
            foul_rolls.append([target])

        if len(foul_positions) > 0:
            actions.append(ActionChoice(ActionType.FOUL, team=player.team,
                                        positions=foul_positions, rolls=foul_rolls))
        return actions

    def get_pickup_teammate_actions(self, player: Player) -> List[ActionChoice]:
        actions = []
        teammates = self.get_adjacent_teammates(player, down=False, skill=Skill.RIGHT_STUFF)
        if teammates:
            positions = [teammate.position for teammate in teammates]
            rolls = [2] if player.has_skill(Skill.ALWAYS_HUNGRY) else []
            d6_rolls = [rolls for _ in teammates]
            actions.append(ActionChoice(ActionType.PICKUP_TEAM_MATE, team=player.team,
                                        positions=positions, rolls=d6_rolls))
        return actions

    def get_block_actions(self, player: Player, blitz = False) -> List[ActionChoice]:

        if player.state.has_blocked:
            return []

        move_needed = 1 if blitz else 0
        jump_up_rolls = []
        if not player.state.up:
            if not player.has_skill(Skill.JUMP_UP):
                if blitz:
                    move_needed += 3
                else:
                    return []
            else:
                jump_up_rolls.append(min(6, Rules.agility_table[player.get_ag()] + 2))

        actions = []

        # Check movement left if blitz,
        gfi = False
        if blitz:
            gfi_allowed = 3 if player.has_skill(Skill.SPRINT) else 2
            if player.state.moves + move_needed > player.get_ma() + gfi_allowed:
                return actions
            gfi = player.state.moves + move_needed > player.get_ma()

        # Find adjacent enemies to block
        block_positions = []
        block_dice = []
        stab_rolls = []
        for player_to in self.get_adjacent_opponents(player, down=False):
            block_positions.append(player_to.position)
            dice = self.num_block_dice(attacker=player, defender=player_to,
                                       blitz=blitz,
                                       dauntless_success=False)
            block_dice.append(dice)
            if player.has_skill(Skill.STAB):
                roll = player_to.get_av() + 1
                if player.has_skill(Skill.STAKES) and player_to.team.race in ['Khemri', 'Necromantic',
                                                                              'Undead', 'Vampire']:
                    roll += 1
                stab_rolls.append(roll)
        if len(block_positions) > 0:
            rolls = [(jump_up_rolls + [2] if gfi else []) for _ in block_positions]
            actions.append(ActionChoice(ActionType.BLOCK, team=player.team,
                                        positions=block_positions,
                                        block_dice=block_dice,
                                        rolls=rolls))
            if player.has_skill(Skill.STAB):
                rolls = [roll + [stab_rolls[i]] for i, roll in enumerate(rolls)]
                actions.append(ActionChoice(ActionType.STAB, team=player.team, positions=block_positions, rolls=rolls))

        return actions

    def get_pass_actions(self, player: Player, piece, dump_off = False) -> List[ActionChoice]:
        actions = []
        if piece:
            ttm = type(piece) == Player
            bomb = type(piece) == Bomb
            positions, distances = self.get_pass_distances(player, piece, dump_off=dump_off)
            d6_rolls = []
            cache = {}
            for i in range(len(distances)):
                distance = distances[i]
                position = positions[i]
                if distance not in cache:
                    modifiers = self.get_pass_modifiers(player, distance, ttm=ttm)
                    target = Rules.agility_table[player.get_ag()]
                    cache[distance] = min(6, max(2, target - modifiers))
                rolls = [cache[distance]]
                player_to = self.get_player_at(position)
                if player_to is not None and type(piece) != Player and (player_to.team == player.team or type(piece) == Bomb) and player_to.can_catch():
                    catch_target = Rules.agility_table[player_to.get_ag()]
                    catch_modifiers = self.get_catch_modifiers(player_to, accurate=True)
                    rolls.append(min(6, max(2, catch_target - catch_modifiers)))
                d6_rolls.append(rolls)
            if len(positions) > 0:
                if bomb:
                    action_type = ActionType.THROW_BOMB
                elif ttm:
                    action_type = ActionType.THROW_TEAM_MATE
                else:
                    action_type = ActionType.PASS
                actions.append(ActionChoice(action_type, team=player.team,
                                            positions=positions, rolls=d6_rolls))
        if dump_off:
            actions.append(ActionChoice(ActionType.DONT_USE_SKILL, team=player.team, skill=Skill.DUMP_OFF))
        return actions

    def get_hypnotic_gaze_actions(self, player: Player) -> List[ActionChoice]:
        actions = []
        if player.has_skill(Skill.HYPNOTIC_GAZE) and player.state.up:

            hypno_positions = self.get_hypno_targets(player)

            if len(hypno_positions) > 0:
                modifier = self.get_hypno_modifier(player)
                target = Rules.agility_table[player.get_ag()]
                d6_roll = min(6, max(2, target - modifier))
                rolls = [[d6_roll]] * len(hypno_positions)

                actions.append(ActionChoice(ActionType.HYPNOTIC_GAZE, team=player.team,
                                            skill=Skill.HYPNOTIC_GAZE, positions=hypno_positions,
                                            rolls=rolls))
        return actions

    def purge_stack_until(self, proc_class, inclusive = False) -> None:
        assert proc_class in [proc.__class__ for proc in self.state.stack.items]
        while not isinstance(self.state.stack.peek(), proc_class):
            self.state.stack.pop()
        if inclusive:
            self.state.stack.pop()
        assert not self.state.stack.is_empty()

    def get_proc(self, proc_type) -> Optional[Procedure]:
        for proc in self.state.stack.items:
            if type(proc) == proc_type:
                return proc
        return None

    def get_stand_up_modifier(self, player: Player) -> int:
        """
        :param player: player on the board with MA < 3. 
        :return:  modifier for player to stand up. 
        """
        if player.has_skill(Skill.TIMMMBER):
            return len([p for p in self.get_adjacent_teammates(player, down=False) if self.num_tackle_zones_in(p) == 0])
        return 0

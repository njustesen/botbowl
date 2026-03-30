"""
Human input handling — translates mouse/keyboard events into game Actions.
Implements a multi-step state machine for action selection.
"""
from __future__ import annotations
import pygame
from dataclasses import dataclass, field
from typing import Optional

from botbowl.core.model import Action, Square
from botbowl.core.table import ActionType
from botbowl.gui.rendering.board import px_to_sq
from botbowl.gui.rendering.buttons import (
    POSITIONAL_ACTIONS, START_ACTIONS, build_player_action_dots
)


@dataclass
class UIState:
    """Persistent UI interaction state between frames."""
    selected_player: object = None           # Player or None (selected for action)
    selected_action_type: object = None      # ActionType or None
    selected_action_choice: object = None    # ActionChoice or None
    highlighted_squares: list = field(default_factory=list)
    hover_path: object = None               # Path or None (MOVE hover)
    hover_pass_rolls: object = None         # list[int] or None (PASS hover roll targets)
    hover_square: object = None             # Square or None
    hover_player: object = None             # Player under mouse (for info display)
    pinned_home_player: object = None       # Last clicked home player (sticky info)
    pinned_away_player: object = None       # Last clicked away player (sticky info)
    player_dots: list = field(default_factory=list)
    special_toggle: object = None           # 'pass' when pass-targets toggle is ON
    pass_mode_pass_ac: object = None        # PASS ActionChoice when MOVE+PASS combined

    def reset_selection(self):
        self.selected_player = None
        self.selected_action_type = None
        self.selected_action_choice = None
        self.highlighted_squares = []
        self.hover_path = None
        self.hover_pass_rolls = None
        self.player_dots = []
        # special_toggle and pass_mode_pass_ac intentionally NOT reset here —
        # they persist across MOVE steps during a pass sequence and are cleared
        # by _rebuild_buttons() when pass mode ends.
        # hover_player, hover_square, pinned_*_player persist across selections


class InputHandler:
    """Translates pygame events into Actions given the current game state."""

    def __init__(self, tile_size: int, pitch_offset: tuple):
        self.tile_size = tile_size
        self.pitch_offset = pitch_offset

    def handle(self, event: pygame.event.Event, game,
               ui_state: UIState,
               action_buttons: list,
               player_dots: list) -> Optional[Action]:
        """
        Process a single event. Returns an Action if one is ready, else None.
        Also updates ui_state in-place.
        """
        if event.type == pygame.MOUSEMOTION:
            self._handle_hover(event.pos, game, ui_state)
            for btn in action_buttons + player_dots:
                btn.update_hover(event.pos)
            return None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                return self._handle_left_click(event.pos, game, ui_state,
                                               action_buttons, player_dots)
            if event.button == 3:
                ui_state.reset_selection()

        if event.type == pygame.KEYDOWN:
            self._handle_key(event, ui_state)

        return None

    def _handle_left_click(self, pos: tuple, game, ui_state: UIState,
                            action_buttons: list,
                            player_dots: list) -> Optional[Action]:
        # 1. Check action bar buttons
        for btn in action_buttons:
            if btn.is_clicked(pos):
                ui_state.reset_selection()
                ac = btn.action
                # For actions with no player/position selection needed
                return Action(ac.action_type,
                              position=None, player=None)

        # 2. Check player action dots (START_* actions)
        for btn in player_dots:
            if btn.is_clicked(pos):
                ac = btn.action
                # If the START action has players to select, set highlight mode
                # and wait for position/player click
                if ac.action_type in START_ACTIONS:
                    ui_state.selected_action_type = ac.action_type
                    ui_state.selected_action_choice = ac
                    # For player-start actions we need to select a player first
                    # but since we clicked on a player dot from selected player,
                    # the player is already selected — use it if it's in ac.players
                    if (ui_state.selected_player and
                            ui_state.selected_player in (ac.players or [])):
                        # Submit START_* action immediately with the selected player
                        player = ui_state.selected_player
                        ui_state.reset_selection()
                        return Action(ac.action_type, player=player)
                    # Otherwise wait for player selection
                    ui_state.highlighted_squares = [sq for sq in (ac.positions or []) if sq is not None]
                return None

        # 3. Check board click
        sq = px_to_sq(pos, self.tile_size, self.pitch_offset,
                      game.arena.width, game.arena.height)
        if sq is None:
            ui_state.reset_selection()
            return None

        return self._handle_board_click(sq, game, ui_state)

    def _handle_board_click(self, sq: Square, game,
                             ui_state: UIState) -> Optional[Action]:
        at = ui_state.selected_action_type
        ac = ui_state.selected_action_choice

        # A. If we have a selected action type and this square is a valid target
        if at is not None and ac is not None:
            if sq in (ac.positions or []):
                # PLACE_PLAYER requires a bench player to be selected first
                if at == ActionType.PLACE_PLAYER and ui_state.selected_player is None:
                    return None
                action = Action(at, position=sq, player=ui_state.selected_player)
                ui_state.reset_selection()
                return action
            else:
                # Clicked outside highlights — maybe selecting a different player
                ui_state.reset_selection()
                # Fall through to player selection

        # B. Try to select player on this square (pitch or bench column)
        player = game.state.pitch.board[sq.y][sq.x] if (
            0 <= sq.x < game.state.pitch.width and
            0 <= sq.y < game.state.pitch.height
        ) else None
        if player is None:
            player = self._get_bench_player(sq, game)

        if player is not None:
            # Check if this player can start an action
            start_actions = [a for a in game.state.available_actions
                             if a.action_type in START_ACTIONS
                             and player in (a.players or [])]
            if start_actions:
                ui_state.selected_player = player
                if player.team == game.state.home_team:
                    ui_state.pinned_home_player = player
                else:
                    ui_state.pinned_away_player = player
                ui_state.player_dots = build_player_action_dots(
                    player, game.state.available_actions,
                    self.tile_size, self.pitch_offset
                )
                ui_state.highlighted_squares = []
                return None

            # Player-only actions (e.g. SELECT_PLAYER for touchback) — no position needed
            player_only = [a for a in game.state.available_actions
                           if a.action_type not in START_ACTIONS
                           and player in (a.players or [])
                           and not a.positions]
            if player_only:
                ac = player_only[0]
                ui_state.selected_player = player
                if player.team == game.state.home_team:
                    ui_state.pinned_home_player = player
                else:
                    ui_state.pinned_away_player = player
                action = Action(ac.action_type, player=player)
                ui_state.reset_selection()
                return action

            # If player has a single direct action (like STAND_UP with position)
            direct = [a for a in game.state.available_actions
                      if a.action_type not in START_ACTIONS
                      and player in (a.players or [])
                      and sq in (a.positions or [])]
            if direct:
                ac = direct[0]
                ui_state.selected_player = player
                if player.team == game.state.home_team:
                    ui_state.pinned_home_player = player
                else:
                    ui_state.pinned_away_player = player
                action = Action(ac.action_type, position=sq, player=player)
                ui_state.reset_selection()
                return action

            # No action for this player — just update info panels (sticky display only)
            if player.team == game.state.home_team:
                ui_state.pinned_home_player = player
            else:
                ui_state.pinned_away_player = player

        # C. Check if any positional action covers this square and submit the first match.
        # Works for single actions (MOVE, PUSH…) and multi-positional (blitz MOVE+BLOCK).
        # selected_player is only set here if it was explicitly chosen (section B no longer
        # corrupts it by setting it to enemies that fall through without a matching action).
        for available_ac in game.state.available_actions:
            if available_ac.action_type in POSITIONAL_ACTIONS:
                if sq in (available_ac.positions or []):
                    # PLACE_PLAYER requires a specific bench player to be selected first
                    if (available_ac.action_type == ActionType.PLACE_PLAYER
                            and ui_state.selected_player is None):
                        continue
                    action = Action(available_ac.action_type,
                                    position=sq, player=ui_state.selected_player)
                    ui_state.reset_selection()
                    return action

        # D. Deselect if nothing matched
        ui_state.reset_selection()
        return None

    def _handle_hover(self, pos: tuple, game, ui_state: UIState):
        sq = px_to_sq(pos, self.tile_size, self.pitch_offset,
                      game.arena.width, game.arena.height)
        ui_state.hover_square = sq
        ui_state.hover_path = None
        ui_state.hover_pass_rolls = None

        # Track player under cursor for info display (pitch or bench)
        if sq is not None and (0 <= sq.x < game.state.pitch.width and
                                0 <= sq.y < game.state.pitch.height):
            ui_state.hover_player = (game.state.pitch.board[sq.y][sq.x]
                                     or self._get_bench_player(sq, game))
        else:
            ui_state.hover_player = None

        if sq is not None and ui_state.selected_action_choice is not None:
            ac = ui_state.selected_action_choice

            if ui_state.selected_action_type == ActionType.MOVE:
                # Find path to hovered square
                for path in (ac.paths or []):
                    if path.steps and path.steps[-1] == sq:
                        ui_state.hover_path = path
                        break

                # Combined pass mode with toggle ON: show pass prob near cursor
                if (ui_state.pass_mode_pass_ac is not None and
                        ui_state.special_toggle == 'pass'):
                    pass_positions = ui_state.pass_mode_pass_ac.positions or []
                    if sq in pass_positions:
                        idx = pass_positions.index(sq)
                        rolls = ui_state.pass_mode_pass_ac.rolls or []
                        if idx < len(rolls):
                            ui_state.hover_pass_rolls = rolls[idx]

            elif ui_state.selected_action_type == ActionType.PASS:
                # Show roll targets for hovered pass destination
                positions = ac.positions or []
                if sq in positions:
                    idx = positions.index(sq)
                    rolls = ac.rolls or []
                    if idx < len(rolls):
                        ui_state.hover_pass_rolls = rolls[idx]

        elif sq is not None and ui_state.selected_action_choice is None:
            # Multi-positional mode (blitz/foul): show path preview from any action with paths
            _PATH_TYPES = (ActionType.MOVE, ActionType.BLOCK, ActionType.FOUL,
                           ActionType.STAB, ActionType.HANDOFF)
            for ac in game.state.available_actions:
                if ac.action_type in _PATH_TYPES:
                    for path in (ac.paths or []):
                        if path.steps and path.steps[-1] == sq:
                            ui_state.hover_path = path
                            break
                if ui_state.hover_path:
                    break

    def _get_bench_player(self, sq, game):
        """Return the bench player drawn at this tile (side crowd column), or None."""
        if sq is None:
            return None
        arena_h = game.arena.height
        if sq.y < 1 or sq.y > arena_h - 2:
            return None
        idx = sq.y - 1
        if sq.x == 0:
            off_pitch = [p for p in game.state.away_team.players if p.position is None]
            return off_pitch[idx] if idx < len(off_pitch) else None
        if sq.x == game.arena.width - 1:
            off_pitch = [p for p in game.state.home_team.players if p.position is None]
            return off_pitch[idx] if idx < len(off_pitch) else None
        return None

    def _handle_key(self, event: pygame.event.Event, ui_state: UIState):
        if event.key == pygame.K_ESCAPE:
            ui_state.reset_selection()

    def select_action_type(self, action_type: ActionType,
                            game, ui_state: UIState):
        """Called when the user selects a positional action type to highlight targets."""
        for ac in game.state.available_actions:
            if ac.action_type == action_type:
                ui_state.selected_action_type = action_type
                ui_state.selected_action_choice = ac
                ui_state.highlighted_squares = [sq for sq in (ac.positions or []) if sq is not None]
                return

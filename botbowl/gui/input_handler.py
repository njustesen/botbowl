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
    grid_mode: str = 'none'                 # 'none', 'full', 'dots'
    special_toggle: object = None           # ActionType or None

    def reset_selection(self):
        self.selected_player = None
        self.selected_action_type = None
        self.selected_action_choice = None
        self.highlighted_squares = []
        self.hover_path = None
        self.hover_pass_rolls = None
        self.player_dots = []
        self.special_toggle = None
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
                player = ui_state.selected_player
                # If this action requires a player but none is selected, skip
                if player is None and ac.players:
                    ui_state.reset_selection()
                    return None
                action = Action(at, position=sq, player=player)
                ui_state.reset_selection()
                return action
            else:
                # Clicked outside highlights — maybe selecting a different player
                ui_state.reset_selection()
                # Fall through to player selection

        # B. Try to select player on this square
        player = game.state.pitch.board[sq.y][sq.x] if (
            0 <= sq.x < game.state.pitch.width and
            0 <= sq.y < game.state.pitch.height
        ) else None

        if player is not None:
            # Always update selected_player when clicking a player (for action + sticky info)
            ui_state.selected_player = player
            # Pin to the appropriate team panel
            if player.team == game.state.home_team:
                ui_state.pinned_home_player = player
            else:
                ui_state.pinned_away_player = player

            # Check if this player can start an action
            start_actions = [a for a in game.state.available_actions
                             if a.action_type in START_ACTIONS
                             and player in (a.players or [])]
            if start_actions:
                ui_state.player_dots = build_player_action_dots(
                    player, game.state.available_actions,
                    self.tile_size, self.pitch_offset
                )
                ui_state.highlighted_squares = []
                return None

            # If player has a single direct action (like STAND_UP with position)
            direct = [a for a in game.state.available_actions
                      if a.action_type not in START_ACTIONS
                      and player in (a.players or [])
                      and sq in (a.positions or [])]
            if direct:
                ac = direct[0]
                action = Action(ac.action_type, position=sq, player=player)
                ui_state.reset_selection()
                return action

        # C. Check if any positional action covers this square and submit the first match.
        # Works for single actions (MOVE, PUSH…) and multi-positional (blitz MOVE+BLOCK).
        for available_ac in game.state.available_actions:
            if available_ac.action_type in POSITIONAL_ACTIONS:
                if sq in (available_ac.positions or []):
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

        # Track player under cursor for info display
        if sq is not None and (0 <= sq.x < game.state.pitch.width and
                                0 <= sq.y < game.state.pitch.height):
            ui_state.hover_player = game.state.pitch.board[sq.y][sq.x]
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

            elif ui_state.selected_action_type == ActionType.PASS:
                # Show roll targets for hovered pass destination
                positions = ac.positions or []
                if sq in positions:
                    idx = positions.index(sq)
                    rolls = ac.rolls or []
                    if idx < len(rolls):
                        ui_state.hover_pass_rolls = rolls[idx]

    def _handle_key(self, event: pygame.event.Event, ui_state: UIState):
        if event.key == pygame.K_ESCAPE:
            ui_state.reset_selection()
        elif event.key == pygame.K_g:
            modes = ['none', 'full', 'dots']
            idx = modes.index(ui_state.grid_mode)
            ui_state.grid_mode = modes[(idx + 1) % len(modes)]

    def select_action_type(self, action_type: ActionType,
                            game, ui_state: UIState):
        """Called when the user selects a positional action type to highlight targets."""
        for ac in game.state.available_actions:
            if ac.action_type == action_type:
                ui_state.selected_action_type = action_type
                ui_state.selected_action_choice = ac
                ui_state.highlighted_squares = [sq for sq in (ac.positions or []) if sq is not None]
                return

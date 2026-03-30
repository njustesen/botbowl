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
from botbowl.core.pathfinding import Pathfinder
from botbowl.gui.rendering.board import px_to_sq
from botbowl.gui.rendering.buttons import (
    POSITIONAL_ACTIONS, START_ACTIONS, BAR_ICON_ACTIONS, build_player_action_dots
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
    projected_player_dots: list = field(default_factory=list)  # Projected dots for player-switching
    projected_paths: list = field(default_factory=list)        # Pathfinder paths for projected player
    special_toggle: object = None           # 'pass' when pass-targets toggle is ON
    pass_mode_pass_ac: object = None        # PASS ActionChoice when MOVE+PASS combined
    # Pending chain state (survives reset_selection; resolved by game_screen after game.step)
    pending_player_select: object = None    # Player to activate after END_PLAYER_TURN
    pending_start_action_type: object = None  # START_* to submit after pending_player_select
    pending_position: object = None         # Square to auto-submit positional action
    pending_positional_type: object = None  # MOVE/BLOCK/FOUL ActionType for auto-submit
    pending_end_turn: bool = False          # After END_PLAYER_TURN, also submit END_TURN

    def reset_selection(self):
        self.selected_player = None
        self.selected_action_type = None
        self.selected_action_choice = None
        self.highlighted_squares = []
        self.hover_path = None
        self.hover_pass_rolls = None
        self.player_dots = []
        self.projected_player_dots = []
        self.projected_paths = []
        # special_toggle and pass_mode_pass_ac intentionally NOT reset here —
        # they persist across MOVE steps during a pass sequence and are cleared
        # by _rebuild_buttons() when pass mode ends.
        # pending_* fields also NOT reset here — they survive across game.step() boundaries.
        # hover_player, hover_square, pinned_*_player persist across selections


def _find_turn_proc(game):
    """Return the Turn procedure in the stack (may be below MoveAction), or None."""
    for proc in reversed(game.state.stack.items):
        if hasattr(proc, 'blitz_available') and hasattr(proc, 'team'):
            return proc
    return None


def _infer_start_action(selected_player, sq, game, available_actions):
    """
    Given a selected player and clicked board square, infer which START_* action to use
    and what follow-up positional action type to auto-submit.

    Returns (start_action_type, positional_action_type) or None if no inference possible.
    positional_action_type is None for START_BLITZ (user navigates manually).
    """
    at_sq = (game.state.pitch.board[sq.y][sq.x]
             if (0 <= sq.x < game.state.pitch.width and
                 0 <= sq.y < game.state.pitch.height)
             else None)

    def has_start(at):
        return any(a.action_type == at and selected_player in (a.players or [])
                   for a in available_actions)

    if at_sq is not None and at_sq.team != selected_player.team:
        # Enemy player at clicked square
        dist = selected_player.position.distance(sq)
        if dist == 1:
            if not at_sq.state.up or at_sq.state.stunned:
                # Adjacent prone/stunned enemy — prefer FOUL, fall back to BLOCK
                if has_start(ActionType.START_FOUL):
                    return (ActionType.START_FOUL, ActionType.FOUL)
                if has_start(ActionType.START_BLOCK):
                    return (ActionType.START_BLOCK, ActionType.BLOCK)
            else:
                # Adjacent standing enemy — BLOCK
                if has_start(ActionType.START_BLOCK):
                    return (ActionType.START_BLOCK, ActionType.BLOCK)
        else:
            # Non-adjacent enemy — BLITZ (user moves manually after)
            if has_start(ActionType.START_BLITZ):
                return (ActionType.START_BLITZ, None)
    elif at_sq is not None and at_sq.team == selected_player.team:
        # Friendly player at clicked square — HANDOFF (adjacent) or PASS
        dist = selected_player.position.distance(sq)
        if dist == 1 and has_start(ActionType.START_HANDOFF):
            return (ActionType.START_HANDOFF, ActionType.HANDOFF)
        if has_start(ActionType.START_PASS):
            return (ActionType.START_PASS, ActionType.PASS)
    else:
        # Empty square — MOVE
        if has_start(ActionType.START_MOVE):
            return (ActionType.START_MOVE, ActionType.MOVE)
    return None


def _get_projected_paths(game, player, turn_proc):
    """Compute projected paths for a player using Pathfinder (read-only, no game state change).

    Returns a list of Path objects. Used to display projected highlights and overlays
    (probabilities, block dice) when another player is currently active.
    """
    if player is None or player.position is None:
        return []
    try:
        blitz_av = getattr(turn_proc, 'blitz_available', False)
        foul_av = getattr(turn_proc, 'foul_available', False)
        directly = getattr(game.config, 'pathfinding_directly_to_adjacent', False)
        pf = Pathfinder(game, player,
                        directly_to_adjacent=directly,
                        can_block=blitz_av,
                        can_foul=foul_av)
        return pf.get_paths()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []


def _projected_squares_from_paths(paths):
    """Extract unique endpoint squares from a list of Pathfinder paths."""
    sqs = set()
    for path in paths:
        if path.steps:
            sqs.add(path.steps[-1])
    return list(sqs)


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
            for btn in action_buttons + player_dots + ui_state.projected_player_dots:
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
                ac = btn.action
                if ac.action_type in BAR_ICON_ACTIONS:
                    # Icon buttons for START_* actions — need a player to submit with
                    if ui_state.projected_player_dots:
                        # Player-switching mode: end current player's turn, then activate
                        ui_state.pending_player_select = ui_state.selected_player
                        ui_state.pending_start_action_type = ac.action_type
                        ui_state.reset_selection()
                        return Action(ActionType.END_PLAYER_TURN)
                    elif (ui_state.selected_player and
                            ui_state.selected_player in (ac.players or [])):
                        player = ui_state.selected_player
                        ui_state.reset_selection()
                        return Action(ac.action_type, player=player)
                    elif len(ac.players or []) == 1:
                        ui_state.reset_selection()
                        return Action(ac.action_type, player=ac.players[0])
                    else:
                        # Highlight eligible players for manual selection
                        ui_state.selected_action_type = ac.action_type
                        ui_state.selected_action_choice = ac
                        ui_state.highlighted_squares = [
                            p.position for p in (ac.players or []) if p.position is not None
                        ]
                        return None
                else:
                    if ac.action_type == ActionType.END_PLAYER_TURN:
                        ui_state.pending_end_turn = True
                    ui_state.reset_selection()
                    return Action(ac.action_type, position=None, player=None)

        # 2. Check projected player dots (player-switching: clicking a dot ends current
        #    player's turn and starts the new player's action)
        for btn in ui_state.projected_player_dots:
            if btn.is_clicked(pos):
                ui_state.pending_player_select = ui_state.selected_player
                ui_state.pending_start_action_type = btn.action.action_type
                ui_state.reset_selection()
                return Action(ActionType.END_PLAYER_TURN)

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
            # PASS/HANDOFF: if clicking a teammate and one of these actions targets this
            # square, submit it immediately — don't select or switch to the player.
            _PASS_HANDOFF = (ActionType.PASS, ActionType.HANDOFF)
            pass_handoff_ac = next(
                (a for a in game.state.available_actions
                 if a.action_type in _PASS_HANDOFF and sq in (a.positions or [])),
                None
            )
            if pass_handoff_ac is not None:
                if player.team == game.state.home_team:
                    ui_state.pinned_home_player = player
                else:
                    ui_state.pinned_away_player = player
                action = Action(pass_handoff_ac.action_type, position=sq,
                                player=ui_state.selected_player)
                ui_state.reset_selection()
                return action

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

            # Player switching shortcut: show projected dots for unused teammate when another
            # player is mid-turn (END_PLAYER_TURN is available).
            end_turn_ac = next((a for a in game.state.available_actions
                                if a.action_type == ActionType.END_PLAYER_TURN), None)
            if (end_turn_ac is not None
                    and player != game.state.active_player
                    and not player.state.used):
                turn_proc = _find_turn_proc(game)
                if turn_proc is not None and player.team == turn_proc.team:
                    proj_actions = turn_proc.available_actions()
                    player_proj = [a for a in proj_actions
                                   if a.action_type in START_ACTIONS
                                   and player in (a.players or [])]
                    if player_proj:
                        ui_state.reset_selection()
                        ui_state.selected_player = player
                        ui_state.projected_player_dots = build_player_action_dots(
                            player, proj_actions, self.tile_size, self.pitch_offset)
                        ui_state.projected_paths = _get_projected_paths(game, player, turn_proc)
                        ui_state.highlighted_squares = _projected_squares_from_paths(
                            ui_state.projected_paths)
                        if player.team == game.state.home_team:
                            ui_state.pinned_home_player = player
                        else:
                            ui_state.pinned_away_player = player
                        return None

            # No action for this player — just update info panels (sticky display only)
            if player.team == game.state.home_team:
                ui_state.pinned_home_player = player
            else:
                ui_state.pinned_away_player = player

        # B2. Smart action inference: player selected but no action type yet — infer from sq.
        # Only applies when the clicked square has no friendly player (avoid mis-inferring
        # MOVE when the user clicks a teammate).
        # Case 1: normal Turn phase (no END_PLAYER_TURN available).
        # Case 2: END_PLAYER_TURN available + projected dots visible (player-switching).
        if (ui_state.selected_player is not None
                and ui_state.selected_action_type is None):
            end_turn_avail = any(a.action_type == ActionType.END_PLAYER_TURN
                                 for a in game.state.available_actions)
            if not end_turn_avail:
                # Normal Turn phase: infer action from current available_actions
                result = _infer_start_action(
                    ui_state.selected_player, sq, game, game.state.available_actions)
                if result:
                    start_type, pos_type = result
                    pending_player = ui_state.selected_player
                    ui_state.pending_position = sq if pos_type else None
                    ui_state.pending_positional_type = pos_type
                    ui_state.reset_selection()
                    return Action(start_type, player=pending_player)
            elif ui_state.projected_player_dots:
                # Player-switching: infer action from projected Turn actions
                turn_proc = _find_turn_proc(game)
                if turn_proc is not None:
                    proj_actions = turn_proc.available_actions()
                    result = _infer_start_action(
                        ui_state.selected_player, sq, game, proj_actions)
                    if result:
                        start_type, pos_type = result
                        ui_state.pending_player_select = ui_state.selected_player
                        ui_state.pending_start_action_type = start_type
                        ui_state.pending_position = sq if pos_type else None
                        ui_state.pending_positional_type = pos_type
                        ui_state.reset_selection()
                        return Action(ActionType.END_PLAYER_TURN)

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
            if ui_state.projected_paths:
                # Projected mode: show path preview from projected player's paths
                for path in ui_state.projected_paths:
                    if path.steps and path.steps[-1] == sq:
                        ui_state.hover_path = path
                        break
            else:
                # Multi-positional mode (blitz/foul): show path preview from game actions
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

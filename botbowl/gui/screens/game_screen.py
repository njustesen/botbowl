"""
Main game screen — handles rendering and input for all game phases.
Supports AI vs AI (spectating), human vs AI, and human vs human (hotseat).
"""
from __future__ import annotations
import pygame
from typing import Optional

from botbowl.core.model import Action
from botbowl.core.table import ActionType, Rules, PassDistance, OutcomeType
from botbowl.gui.fonts import get_font, get_body_font
from botbowl.gui.input_handler import (InputHandler, UIState, _find_turn_proc,
                                        _get_projected_paths, _projected_squares_from_paths)
from botbowl.gui.rendering.board import BoardRenderer, sq_to_px, highlight_color_for_action
from botbowl.gui.rendering.players import PlayerRenderer
from botbowl.gui.rendering.hud import HUDRenderer, ActionBarRenderer, PlayerInfoRenderer
from botbowl.gui.rendering.log_panel import LogPanelRenderer
from botbowl.gui.rendering.buttons import (
    build_action_buttons, build_player_action_dots, build_board_block_dice,
    POSITIONAL_ACTIONS, START_ACTIONS, FORMATION_ACTIONS
)
from botbowl.gui.rendering.ui_primitives import (
    Button, Modal, TextInput, LabeledToggle, KickoffEventBody,
    COLOR_BTN_NEUTRAL, COLOR_BTN_DEFAULT
)
from botbowl.gui.save_load import save_game, save_exists, list_saves
from botbowl.gui.assets import prettify
from botbowl.gui.sprites import get_action_icon

TILE_SIZE = 30

_BLOCK_CURSOR_TYPES = {ActionType.BLOCK, ActionType.STAB, ActionType.HYPNOTIC_GAZE}
_FOUL_CURSOR_TYPES = {ActionType.FOUL}
_HANDOFF_CURSOR_TYPES = {ActionType.HANDOFF}
_COMBAT_TYPES = _BLOCK_CURSOR_TYPES | _FOUL_CURSOR_TYPES

_PASS_ABBR = {
    PassDistance.QUICK_PASS: 'QP',
    PassDistance.SHORT_PASS: 'SP',
    PassDistance.LONG_PASS:  'LP',
    PassDistance.LONG_BOMB:  'LB',
    PassDistance.HAIL_MARY:  'HM',
}


def _pass_type_label(from_sq, to_sq) -> str:
    """Return abbreviated pass-type label using botbowl's pass_matrix."""
    dy = abs(from_sq.y - to_sq.y)
    dx = abs(from_sq.x - to_sq.x)
    if dy >= len(Rules.pass_matrix) or dx >= len(Rules.pass_matrix[0]):
        return 'HM'
    val = Rules.pass_matrix[dy][dx]
    try:
        return _PASS_ABBR.get(PassDistance(val), '??')
    except ValueError:
        return 'HM'
HUD_H = 72          # Scoreboard height
INFO_H = 155        # Bottom panel height
INFO_PLAYER_W = 200 # Width of each player info panel (log takes the rest)

# Maps OutcomeType → (display title, description shown in modal)
_KICKOFF_INFO: dict = {
    OutcomeType.KICKOFF_GET_THE_REF:        (
        'Get the Ref!',
        'The fans drag the referee into the stands after one too many bad calls. '
        'His trembling replacement has no intention of making enemies. '
        'Both teams gain +1 Bribe for this game.'),
    OutcomeType.KICKOFF_RIOT:               (
        'Riot!',
        'A brawl erupts on the pitch! Roll D6: on 1-3 both teams lose a turn, on 4-6 both gain one.'),
    OutcomeType.KICKOFF_PERFECT_DEFENSE:    (
        'Perfect Defence!',
        'The kicking team may reorganise their defensive formation.'),
    OutcomeType.KICKOFF_HIGH_KICK:          (
        'High Kick!',
        ''),   # overwritten dynamically in _show_kickoff_modal
    OutcomeType.KICKOFF_CHEERING_FANS:      (
        'Cheering Fans!',
        'Both teams roll D3 + Cheerleaders + Fame. '
        'The higher total earns a bonus Re-Roll this half.'),
    OutcomeType.KICKOFF_CHANGING_WHEATHER:  (
        'Changing Weather!',
        ''),   # overwritten dynamically in _show_kickoff_modal
    OutcomeType.KICKOFF_BRILLIANT_COACHING: (
        'Brilliant Coaching!',
        'Inspired tactics! Both teams roll D3 + Coaches + Fame. '
        'The higher total earns a bonus Re-Roll this half.'),
    OutcomeType.KICKOFF_QUICK_SNAP:         (
        'Quick Snap!',
        'The offence surges forward before the whistle! Every receiving player may move one '
        'square for free, ignoring tackle zones.'),
    OutcomeType.KICKOFF_BLITZ:              (
        'Blitz!',
        'The defence charges in before the kick-off! The kicking team gets a free bonus turn. '
        'Players in tackle zones may not act; any turnover ends the turn immediately.'),
    OutcomeType.KICKOFF_THROW_A_ROCK:       (
        'Throw a Rock!',
        'An enraged fan hurls a rock! Both teams roll D3 + Fame. The lower score has a random '
        'player struck — no armour roll needed. Ties mean both teams suffer.'),
    OutcomeType.KICKOFF_PITCH_INVASION:     (
        'Pitch Invasion!',
        'Fans storm the pitch! For each opposing player, roll D6 + enemy Fame. '
        'On 6+ the player is Stunned (Ball & Chain carriers are KO\'d). A natural 1 always fails.'),
}

# Weather name, dynamic description, and rules effect text
_WEATHER_INFO: dict = {
    OutcomeType.WEATHER_SWELTERING_HEAT: (
        'Sweltering Heat',
        'The scorching heat is unbearable! Players coming off the pitch risk heatstroke.',
        'At the end of each drive, players in reserves roll D6 — on a 1 they\'re KO\'d.'),
    OutcomeType.WEATHER_VERY_SUNNY:      (
        'Very Sunny',
        'The sun is blazing! Glare makes it harder to track the ball.',
        '-1 to all Pass rolls.'),
    OutcomeType.WEATHER_NICE:            (
        'Nice',
        'The weather clears up — perfect playing conditions.',
        'No weather effects.'),
    OutcomeType.WEATHER_POURING_RAIN:    (
        'Pouring Rain',
        'It\'s pouring with rain! The ball is soaked and slippery.',
        '-1 to Catch and Pick-up rolls.'),
    OutcomeType.WEATHER_BLIZZARD:        (
        'Blizzard',
        'A blizzard sweeps across the pitch! Visibility is near zero.',
        'Only Quick and Short Passes allowed. Go For It rolls are harder.'),
}


def _high_kick_description(game) -> str:
    """Return a description for High Kick that reflects whether it can actually be used."""
    try:
        receiving_team = game.get_receiving_team()
        balls = game.state.pitch.balls
        if balls:
            ball = balls[0]
            if (ball.position is not None
                    and game.is_team_side(ball.position, receiving_team)
                    and game.get_player_at(ball.position) is None):
                return ('The ball soars sky-high! '
                        'Move one unmarked receiver to the landing square for free.')
        return ('The ball soars sky-high! '
                'But the landing square is already occupied or out of reach — High Kick has no effect.')
    except Exception:
        return ('The ball soars sky-high! '
                'Move one unmarked receiver to the landing square for free.')


def _weather_dynamic_description(sub_outcomes: list, game) -> str:
    """Build a dynamic description for the Changing Weather kickoff event."""
    for o in sub_outcomes:
        info = _WEATHER_INFO.get(o.outcome_type)
        if info:
            wname, wdesc, _ = info
            if o.outcome_type == OutcomeType.WEATHER_NICE and getattr(game.state, 'gentle_gust', False):
                return 'The weather stays Nice — a gentle gust scatters the ball one extra square.'
            return wdesc
    return 'The weather has changed!'

# Procedure class name to wait for before showing the modal (None = show immediately).
# We wait for the sub-procedure to finish so we can collect its outcome rolls.
_KICKOFF_WAIT_PROC: dict = {
    OutcomeType.KICKOFF_GET_THE_REF:        'GetTheRef',
    OutcomeType.KICKOFF_RIOT:               'Riot',
    OutcomeType.KICKOFF_HIGH_KICK:          None,   # HighKick awaits human input — show now
    OutcomeType.KICKOFF_CHEERING_FANS:      'CheeringFans',
    OutcomeType.KICKOFF_CHANGING_WHEATHER:  'WeatherTable',
    OutcomeType.KICKOFF_BRILLIANT_COACHING: 'BrilliantCoaching',
    OutcomeType.KICKOFF_QUICK_SNAP:         None,   # no sub-rolls; free turn follows
    OutcomeType.KICKOFF_BLITZ:              None,   # no sub-rolls; free turn follows
    OutcomeType.KICKOFF_THROW_A_ROCK:       'ThrowARock',
    OutcomeType.KICKOFF_PITCH_INVASION:     'PitchInvasionRoll',
    OutcomeType.KICKOFF_PERFECT_DEFENSE:    None,   # Setup (human action) follows
}


def _team_label(team, game) -> str:
    return team.name if team is not None else '?'


def _team_color(team, game) -> tuple:
    """Return the home/away text color for a team (bright, readable on dark background)."""
    if team is None:
        return (245, 242, 235)
    if team == game.state.home_team:
        return (180, 220, 255)   # light sky blue
    return (255, 200, 140)       # light orange


def _format_kickoff_sub_rows(ot: OutcomeType, sub_outcomes: list, game) -> list:
    """Return colour-prefixed display strings for kickoff event sub-roll outcomes.

    Prefix '+' → green (gain), '-' → red (loss), ' ' → neutral.
    """
    rows = []

    if ot == OutcomeType.KICKOFF_GET_THE_REF:
        pass  # effect is stated in the description; no sub-row needed

    elif ot == OutcomeType.KICKOFF_RIOT:
        for o in sub_outcomes:
            if o.outcome_type == OutcomeType.TURN_ADDED:
                if o.rolls:
                    rows.append(f' [{o.rolls[0].get_sum()}]  \u2192  Both teams gain a turn.')
                else:
                    rows.append(f' No roll \u2014 it\'s turn 8, so both teams gain a turn.')
            elif o.outcome_type == OutcomeType.TURN_SKIPPED:
                if o.rolls:
                    rows.append(f' [{o.rolls[0].get_sum()}]  \u2192  Both teams lose a turn.')
                else:
                    rows.append(f' No roll \u2014 it\'s turn 1, so both teams lose a turn.')

    elif ot in (OutcomeType.KICKOFF_CHEERING_FANS, OutcomeType.KICKOFF_BRILLIANT_COACHING):
        roll_ot = (OutcomeType.CHEERING_FANS_ROLL
                   if ot == OutcomeType.KICKOFF_CHEERING_FANS
                   else OutcomeType.BRILLIANT_COACHING_ROLL)
        _neutral = (245, 242, 235)
        roll_outcomes = [o for o in sub_outcomes
                         if o.outcome_type == roll_ot and o.team is not None]
        reroll_teams = [o.team for o in sub_outcomes
                        if o.outcome_type == OutcomeType.EXTRA_REROLL and o.team is not None]
        for o in roll_outcomes:
            d = o.rolls[0].get_sum() if o.rolls else '?'
            mod = o.rolls[0].modifiers if o.rolls else 0
            total = o.rolls[0].get_result() if o.rolls else '?'
            tc = _team_color(o.team, game)
            rows.append([
                (_team_label(o.team, game), tc),
                (f': [{d}] + {mod} = {total}', _neutral),
            ])
        if len(reroll_teams) >= 2:
            rows.append(f' Both teams gain +1 Re-Roll this half.')
        elif len(reroll_teams) == 1:
            tc = _team_color(reroll_teams[0], game)
            rows.append([
                (_team_label(reroll_teams[0], game), tc),
                (' gains +1 Re-Roll this half.', _neutral),
            ])

    elif ot == OutcomeType.KICKOFF_CHANGING_WHEATHER:
        for o in sub_outcomes:
            info = _WEATHER_INFO.get(o.outcome_type)
            if info is not None:
                wname, _, weffect = info
                if o.rolls:
                    vals = o.rolls[0].get_values()
                    d_str = f'[{vals[0]}][{vals[1]}]  \u2192 '
                else:
                    d_str = '\u2192 '
                rows.append(f' {d_str}{wname}')
                if o.outcome_type != OutcomeType.WEATHER_NICE:
                    rows.append(f'   {weffect}')

    elif ot == OutcomeType.KICKOFF_THROW_A_ROCK:
        _neutral = (245, 242, 235)
        for o in sub_outcomes:
            if o.outcome_type == OutcomeType.THROW_A_ROCK_ROLL and o.team is not None:
                d = o.rolls[0].get_sum() if o.rolls else '?'
                mod = o.rolls[0].modifiers if o.rolls else 0
                total = o.rolls[0].get_result() if o.rolls else '?'
                tc = _team_color(o.team, game)
                rows.append([
                    (_team_label(o.team, game), tc),
                    (f': [{d}] + {mod} = {total}', _neutral),
                ])
        for o in sub_outcomes:
            if o.outcome_type == OutcomeType.HIT_BY_ROCK and o.player is not None:
                t = o.player.team if o.player.team else None
                tc = _team_color(t, game)
                lbl = _team_label(t, game) if t is not None else '?'
                rows.append([
                    (f'Player #{o.player.nr} (', _neutral),
                    (lbl, tc),
                    (') hit by a rock!', _neutral),
                ])

    elif ot == OutcomeType.KICKOFF_PITCH_INVASION:
        _neutral = (245, 242, 235)
        for team in (game.state.home_team, game.state.away_team):
            team_outcomes = [o for o in sub_outcomes
                             if o.outcome_type in (OutcomeType.STUNNED, OutcomeType.KNOCKED_OUT,
                                                   OutcomeType.PLAYER_READY)
                             and (o.team == team
                                  or (o.player and getattr(o.player, 'team', None) == team))]
            if not team_outcomes:
                continue
            stunned = sum(1 for o in team_outcomes if o.outcome_type == OutcomeType.STUNNED)
            koed = sum(1 for o in team_outcomes if o.outcome_type == OutcomeType.KNOCKED_OUT)
            parts = []
            if stunned:
                parts.append(f'{stunned} stunned')
            if koed:
                parts.append(f'{koed} KO\'d')
            effect = ', '.join(parts) if parts else 'unaffected'
            tc = _team_color(team, game)
            rows.append([
                (_team_label(team, game), tc),
                (f': {effect}', _neutral),
            ])

    return rows


class GameScreen:
    """Main game playing/spectating screen."""

    def __init__(self, app, game, home_agent, away_agent,
                 spectating: bool = False, ai_delay_ms: int = 50):
        self.app = app
        self.game = game
        self.home_agent = home_agent
        self.away_agent = away_agent
        self.spectating = spectating
        self.ai_delay_ms = ai_delay_ms

        arena = game.arena
        self.tile_size = TILE_SIZE

        # Pitch pixel dimensions
        self.pitch_px_w = arena.width * TILE_SIZE
        self.pitch_px_h = arena.height * TILE_SIZE

        # Total window size: pitch width only, log moves to bottom center
        total_w = self.pitch_px_w
        total_h = HUD_H + self.pitch_px_h + INFO_H

        # Resize window if needed
        self._resize_display(total_w, total_h)
        self.width = total_w
        self.height = total_h

        # Pitch origin sits directly below the HUD
        self.pitch_offset = (0, HUD_H)

        # Sub-rects
        self.hud_rect = pygame.Rect(0, 0, total_w, HUD_H)
        self.pitch_rect = pygame.Rect(0, HUD_H, self.pitch_px_w, self.pitch_px_h)
        # Context row overlays the top crowd row (row 0)
        self.action_bar_rect = pygame.Rect(0, HUD_H, total_w, TILE_SIZE)
        # Buttons overlay the bottom crowd row (last row)
        self._btns_rect = pygame.Rect(
            0, HUD_H + self.pitch_px_h - TILE_SIZE, total_w, TILE_SIZE
        )
        # Bottom panel: away info | log (remainder) | home info
        info_y = HUD_H + self.pitch_px_h
        log_w = total_w - INFO_PLAYER_W * 2
        self.away_info_rect = pygame.Rect(0, info_y, INFO_PLAYER_W, INFO_H)
        self.log_rect = pygame.Rect(INFO_PLAYER_W, info_y, log_w, INFO_H)
        self.home_info_rect = pygame.Rect(INFO_PLAYER_W + log_w, info_y,
                                          INFO_PLAYER_W, INFO_H)

        # Renderers
        self.board_renderer = BoardRenderer(TILE_SIZE, self.pitch_offset)
        self.player_renderer = PlayerRenderer(TILE_SIZE, self.pitch_offset)
        self.hud_renderer = HUDRenderer(self.hud_rect)
        self.action_bar_renderer = ActionBarRenderer(self.action_bar_rect)
        self.away_info_renderer = PlayerInfoRenderer(self.away_info_rect)
        self.home_info_renderer = PlayerInfoRenderer(self.home_info_rect)
        self.log_renderer = LogPanelRenderer(self.log_rect)
        self.input_handler = InputHandler(TILE_SIZE, self.pitch_offset)

        # State
        self.ui_state = UIState()
        self.action_buttons: list[Button] = []
        self._modal: Optional[Modal] = None
        self._modal_type: str = ''   # 'save', 'quit', or 'kickoff'
        self._save_input: Optional[TextInput] = None
        self._save_error: str = ''
        self._game_over_displayed = False

        # Probability/dice overlay data rebuilt each frame in _rebuild_buttons
        self._highlight_probs: dict = {}     # Square → float (MOVE / PASS)
        self._highlight_rolls: dict = {}     # Square → List[int] (MOVE / PASS)
        self._block_dice_pairs: list = []    # [(Square, int)] for BLOCK (legacy, kept for compat)
        self._board_block_dice: list = []    # Board-positioned die-selection buttons (post-roll)
        self._pass_squares: list = []          # all PASS target squares (toggle-ON view)
        self._pass_receiver_squares: list = []  # PASS targets with a friendly receiver (overlay)
        self._pass_probs: dict = {}            # Square → float for pass overlay / labels
        self._pass_receiver_rolls: dict = {}   # Square → List[int] for combined receiver overlay
        self._pass_toggle: Optional[LabeledToggle] = None
        self._show_probs: bool = False         # Hidden setting: True to show % labels
        self._show_numbers: bool = False       # Hidden setting: True to show jersey numbers on bench

        # Kick-off event modal detection state
        self._kickoff_report_base: int = 0          # scan reports from this index for new kickoff outcomes
        self._kickoff_modal_pending: Optional[dict] = None   # {kickoff_outcome, sub_start, wait_proc}

        # Cursor state ('arrow' | 'pass' | 'block' | 'foul' | 'handoff')
        self._cursor_state: str = 'arrow'
        self._pass_cursor: Optional[pygame.cursors.Cursor] = None
        self._block_cursor: Optional[pygame.cursors.Cursor] = None
        self._foul_cursor: Optional[pygame.cursors.Cursor] = None
        self._handoff_cursor: Optional[pygame.cursors.Cursor] = None

        # Initial button build
        self._rebuild_buttons()

    def _resize_display(self, w: int, h: int):
        current = pygame.display.get_surface()
        if current is None or current.get_size() != (w, h):
            pygame.display.set_mode((w, h))

    def _drain_intermediate(self, limit=50):
        """Advance the game through intermediate states (fast_mode=False) until actions appear."""
        for _ in range(limit):
            if self.game.state.available_actions or self.game.state.game_over:
                break
            self.game.step(None)
            self._scan_kickoff_events()

    def _resolve_pending(self):
        """Resolve pending chained actions (player-switching + smart click auto-chain)."""
        # Step 0: if End Turn was clicked (END_PLAYER_TURN + END_TURN chain)
        if self.ui_state.pending_end_turn:
            self.ui_state.pending_end_turn = False
            self._drain_intermediate()
            end_turn_ac = next(
                (a for a in self.game.state.available_actions
                 if a.action_type == ActionType.END_TURN),
                None
            )
            if end_turn_ac is not None:
                self.game.step(Action(ActionType.END_TURN))
                self._scan_kickoff_events()
                self._drain_intermediate()
            return

        # Step 1: activate pending player after END_PLAYER_TURN
        if self.ui_state.pending_player_select:
            # Drain intermediate procedures — fast_mode=False needs step(None) calls
            self._drain_intermediate()

            player = self.ui_state.pending_player_select
            start_type = self.ui_state.pending_start_action_type
            self.ui_state.pending_player_select = None
            self.ui_state.pending_start_action_type = None
            for ac in self.game.state.available_actions:
                if ac.action_type == start_type and player in (ac.players or []):
                    self.game.step(Action(start_type, player=player))
                    self._scan_kickoff_events()
                    self._drain_intermediate()
                    self.ui_state.selected_player = player
                    break
            else:
                # Fallback: pre-select player if they're available for any START action
                for ac in self.game.state.available_actions:
                    if ac.action_type in START_ACTIONS and player in (ac.players or []):
                        self.ui_state.selected_player = player
                        break

        # Step 2: auto-submit positional action (after START_* becomes active)
        if self.ui_state.pending_position is not None:
            pos_type = self.ui_state.pending_positional_type
            sq = self.ui_state.pending_position
            self.ui_state.pending_position = None
            self.ui_state.pending_positional_type = None
            if pos_type is not None:
                for ac in self.game.state.available_actions:
                    if ac.action_type == pos_type and sq in (ac.positions or []):
                        self.game.step(Action(pos_type, position=sq,
                                              player=self.ui_state.selected_player))
                        self._scan_kickoff_events()
                        self._drain_intermediate()
                        break
                # If sq not reachable, silently skip — normal highlights are shown

    def _rebuild_buttons(self):
        # Auto-trigger the first available formation action (hidden from UI)
        formation_ac = next(
            (ac for ac in self.game.state.available_actions
             if ac.action_type in FORMATION_ACTIONS),
            None
        )
        if formation_ac is not None:
            from botbowl.core.model import Action
            self.game.step(Action(formation_ac.action_type))
            self._scan_kickoff_events()

        self._board_block_dice = build_board_block_dice(
            self.game.state.available_actions, self.game,
            self.tile_size, self.pitch_offset
        )
        self.ui_state.player_dots = []

        # Compute projected actions for player-switching; use them as icon source
        icon_action_src = self.game.state.available_actions
        if self.ui_state.selected_player:
            end_turn_ac = next((a for a in self.game.state.available_actions
                                if a.action_type == ActionType.END_PLAYER_TURN), None)
            if (end_turn_ac is not None and
                    self.ui_state.selected_player != self.game.state.active_player):
                turn_proc = _find_turn_proc(self.game)
                if (turn_proc is not None and
                        self.ui_state.selected_player.team == turn_proc.team):
                    proj_actions = turn_proc.available_actions()
                    icon_action_src = proj_actions  # icons come from projected state
                    self.ui_state.projected_player_dots = build_player_action_dots(
                        self.ui_state.selected_player, proj_actions,
                        self.tile_size, self.pitch_offset)
                    self.ui_state.projected_paths = _get_projected_paths(
                        self.game, self.ui_state.selected_player, turn_proc)
                    self.ui_state.highlighted_squares = _projected_squares_from_paths(
                        self.ui_state.projected_paths)
                else:
                    self.ui_state.projected_player_dots = []
                    self.ui_state.projected_paths = []
                    self.ui_state.highlighted_squares = []
            else:
                self.ui_state.projected_player_dots = []
                self.ui_state.projected_paths = []
                # Start-of-turn: show movement preview for selected player before
                # START_MOVE is submitted (START_MOVE available but MOVE not yet).
                sp = self.ui_state.selected_player
                if sp is not None:
                    has_start_move = any(
                        ac.action_type == ActionType.START_MOVE and sp in (ac.players or [])
                        for ac in self.game.state.available_actions
                    )
                    if has_start_move:
                        turn_proc = _find_turn_proc(self.game)
                        if turn_proc is not None:
                            self.ui_state.projected_paths = _get_projected_paths(
                                self.game, sp, turn_proc)
                            self.ui_state.highlighted_squares = _projected_squares_from_paths(
                                self.ui_state.projected_paths)

        self.action_buttons = build_action_buttons(
            self.game.state.available_actions, self.game,
            self._btns_rect,
            selected_player=self.ui_state.selected_player,
            player_action_src=icon_action_src
        )

        # Auto-highlight positional action squares (skip when in projected/switch mode).
        if not self.ui_state.selected_action_type and not self.ui_state.projected_paths:
            positional = [ac for ac in self.game.state.available_actions
                          if ac.action_type in POSITIONAL_ACTIONS]
            move_acs = [ac for ac in positional if ac.action_type == ActionType.MOVE]
            pass_acs = [ac for ac in positional if ac.action_type == ActionType.PASS]

            if move_acs and pass_acs:
                # Combined pass mode: passer can move first, then throw.
                move_ac = move_acs[0]
                pass_ac = pass_acs[0]
                self.ui_state.pass_mode_pass_ac = pass_ac

                # Build pass squares and probability map (used for overlay and hover)
                self._pass_squares = [sq for sq in (pass_ac.positions or []) if sq is not None]
                self._pass_receiver_squares = []   # only squares with a friendly receiver (2+ rolls)
                self._pass_probs = {}
                self._pass_receiver_rolls = {}
                for i, sq in enumerate(pass_ac.positions or []):
                    if sq is None:
                        continue
                    rolls = (pass_ac.rolls or [])
                    if i < len(rolls):
                        p = 1.0
                        for r in rolls[i]:
                            p *= (7 - r) / 6
                        self._pass_probs[sq] = p
                        if len(rolls[i]) >= 2:
                            self._pass_receiver_squares.append(sq)
                            self._pass_receiver_rolls[sq] = list(rolls[i])

                if self.ui_state.special_toggle == 'pass':
                    # Toggle ON: show only pass targets
                    self.ui_state.selected_action_type = ActionType.PASS
                    self.ui_state.selected_action_choice = pass_ac
                    self.ui_state.highlighted_squares = self._pass_squares[:]
                else:
                    # Toggle OFF (default): show movement paths with pass overlay
                    self.ui_state.selected_action_type = ActionType.MOVE
                    self.ui_state.selected_action_choice = move_ac
                    self.ui_state.highlighted_squares = [
                        path.steps[-1] for path in (move_ac.paths or []) if path.steps
                    ]

                # Build "Pass Mode:" labeled toggle anchored to right of action bar
                toggle_h = 20
                tx = self._btns_rect.right - 38 - 80   # approx: toggle_w=38 + label ~80px
                ty = self._btns_rect.centery - toggle_h // 2 - 1
                self._pass_toggle = LabeledToggle(
                    x=tx, y=ty,
                    label='Pass Mode:',
                    state=(self.ui_state.special_toggle == 'pass'),
                    font_size=13,
                    toggle_w=38, toggle_h=toggle_h,
                )

            else:
                # Not combined pass mode — clear pass mode state
                self.ui_state.pass_mode_pass_ac = None
                self.ui_state.special_toggle = None
                self._pass_squares = []
                self._pass_receiver_squares = []
                self._pass_probs = {}
                self._pass_receiver_rolls = {}
                self._pass_toggle = None

                if len(positional) == 1:
                    ac = positional[0]
                    self.ui_state.selected_action_type = ac.action_type
                    self.ui_state.selected_action_choice = ac
                    if ac.positions:
                        self.ui_state.highlighted_squares = [sq for sq in ac.positions if sq is not None]
                    else:
                        # Player-only action (e.g. SELECT_PLAYER for touchback)
                        self.ui_state.highlighted_squares = [
                            p.position for p in (ac.players or []) if p.position is not None
                        ]
                elif len(positional) > 1:
                    all_sqs: set = set()
                    for pac in positional:
                        for sq in (pac.positions or []):
                            if sq is not None:
                                all_sqs.add(sq)
                    self.ui_state.highlighted_squares = list(all_sqs)
                    # selected_action_type/choice stay None; input_handler section C picks action

        # Build probability map, roll map, and block dice overlays for current selection
        self._highlight_probs = {}
        self._highlight_rolls = {}
        self._block_dice_pairs = []
        ac = self.ui_state.selected_action_choice
        at = self.ui_state.selected_action_type
        if ac is not None:
            if at == ActionType.MOVE:
                for path in (ac.paths or []):
                    if path.steps:
                        self._highlight_probs[path.steps[-1]] = path.prob
                    if path.steps and path.rolls:
                        self._highlight_rolls[path.steps[-1]] = list(path.rolls[-1])
            elif at == ActionType.PASS:
                for i, sq in enumerate(ac.positions or []):
                    if sq is None:
                        continue
                    rolls = (ac.rolls or [])
                    if i < len(rolls):
                        p = 1.0
                        for r in rolls[i]:
                            p *= (7 - r) / 6
                        self._highlight_probs[sq] = p
                        if rolls[i]:
                            self._highlight_rolls[sq] = list(rolls[i])
            elif at == ActionType.HANDOFF:
                for path in (ac.paths or []):
                    if path.steps:
                        self._highlight_probs[path.steps[-1]] = path.prob
                    if path.steps and path.rolls:
                        self._highlight_rolls[path.steps[-1]] = list(path.rolls[-1])
            elif at == ActionType.BLOCK:
                positions = ac.positions or []
                dice = ac.block_dice or []
                for i, sq in enumerate(positions):
                    if sq is not None and i < len(dice):
                        self._block_dice_pairs.append((sq, dice[i]))
        elif self.ui_state.projected_paths:
            # Projected mode: build overlays from Pathfinder paths of the projected player
            for path in self.ui_state.projected_paths:
                if path.steps:
                    sq = path.steps[-1]
                    self._highlight_probs[sq] = path.prob
                    if path.rolls:
                        self._highlight_rolls[sq] = list(path.rolls[-1])
                    if path.block_dice is not None:
                        self._block_dice_pairs.append((sq, path.block_dice))
        else:
            # Multi-positional mode: build block dice + MOVE probs from component actions
            positional = [a for a in self.game.state.available_actions
                          if a.action_type in POSITIONAL_ACTIONS]
            block_ac = next((a for a in positional if a.action_type == ActionType.BLOCK), None)
            if block_ac:
                for i, sq in enumerate(block_ac.positions or []):
                    if sq is not None and i < len(block_ac.block_dice or []):
                        self._block_dice_pairs.append((sq, block_ac.block_dice[i]))
            for pac in positional:
                if pac.action_type in (ActionType.MOVE, ActionType.BLOCK,
                                       ActionType.FOUL, ActionType.STAB, ActionType.HANDOFF):
                    for path in (pac.paths or []):
                        if path.steps:
                            self._highlight_probs[path.steps[-1]] = path.prob
                        if path.steps and path.rolls:
                            self._highlight_rolls[path.steps[-1]] = list(path.rolls[-1])

    def is_human_turn(self) -> bool:
        if self.spectating:
            return False
        actor = self.game.actor
        return actor is not None and actor.human

    def _scan_kickoff_events(self):
        """Scan game reports for new kickoff outcomes and show a modal when the event resolves."""
        reports = self.game.state.reports
        new = reports[self._kickoff_report_base:]

        # Detect a new kickoff outcome in reports we haven't seen yet
        if not self._kickoff_modal_pending:
            for i, outcome in enumerate(new):
                if outcome.outcome_type in _KICKOFF_WAIT_PROC:
                    self._kickoff_modal_pending = {
                        'kickoff_outcome': outcome,
                        'sub_start': self._kickoff_report_base + i + 1,
                        'wait_proc': _KICKOFF_WAIT_PROC[outcome.outcome_type],
                    }
                    break

        self._kickoff_report_base = len(reports)

        # Try to show the modal if one is pending and no other modal is open
        if self._kickoff_modal_pending and not self._modal:
            wait_proc = self._kickoff_modal_pending['wait_proc']
            if wait_proc is None:
                self._show_kickoff_modal()
            else:
                proc = self.game.get_procedure()
                proc_class = type(proc).__name__ if proc else ''
                if proc_class != wait_proc:
                    self._show_kickoff_modal()

    def _show_kickoff_modal(self):
        """Build and open the kickoff event modal with dice visualisation and sub-roll results."""
        data = self._kickoff_modal_pending
        self._kickoff_modal_pending = None

        kickoff_outcome = data['kickoff_outcome']
        sub_outcomes = self.game.state.reports[data['sub_start']:]

        ot = kickoff_outcome.outcome_type
        name, description = _KICKOFF_INFO[ot]
        if ot == OutcomeType.KICKOFF_CHANGING_WHEATHER:
            description = _weather_dynamic_description(sub_outcomes, self.game)
        elif ot == OutcomeType.KICKOFF_HIGH_KICK:
            description = _high_kick_description(self.game)
        die_values = kickoff_outcome.rolls[0].get_values() if kickoff_outcome.rolls else []

        sub_rows = _format_kickoff_sub_rows(ot, sub_outcomes, self.game)

        modal_w = 520 if sub_rows else 500
        modal_h = 420 + len(sub_rows) * 28
        modal_h = min(modal_h, self.height - 40)  # never overflow screen
        mx = (self.width - modal_w) // 2
        my = (self.height - modal_h) // 2
        content_top = my + int(modal_h * 0.24)
        body_y = content_top + 32
        btn_ratio = 0.76
        button_y = my + int(modal_h * btn_ratio)
        margin = 50
        body_rect = pygame.Rect(mx + margin, body_y, modal_w - margin * 2, button_y - 10 - body_y)

        body = KickoffEventBody(body_rect, description, die_values, sub_rows)
        self._modal = Modal(
            (self.width, self.height), name, body,
            ok_label='OK', size=(modal_w, modal_h),
            ok_only=True, btn_y_ratio=btn_ratio
        )
        self._modal_type = 'kickoff'

    def update(self):
        if self.game.state.game_over:
            return

        # Pause game advancement while a kickoff event modal is being shown
        if self._modal and self._modal_type == 'kickoff':
            return

        # Advance intermediate procedures that require no action (fast_mode=False).
        # game.step() breaks on intermediate steps when fast_mode=False, so we must
        # call step(None) repeatedly until available_actions is non-empty or game is over.
        if not self.game.state.available_actions:
            self.game.step(None)
            self._scan_kickoff_events()
            self._rebuild_buttons()
            return

        # If it's an AI turn (not waiting for human), let the agent act
        if not self.is_human_turn():
            actor = self.game.actor
            if actor is not None and not actor.human:
                try:
                    action = actor.act(self.game)
                    self.game.step(action)
                except Exception:
                    # If agent fails, try a fallback
                    if self.game.state.available_actions:
                        ac = self.game.state.available_actions[0]
                        fallback = Action(ac.action_type)
                        self.game.step(fallback)
                self._scan_kickoff_events()
                self._rebuild_buttons()
                if self.ai_delay_ms > 0:
                    pygame.time.wait(self.ai_delay_ms)

    def handle_event(self, event: pygame.event.Event):
        # Handle modal first
        if self._modal:
            result = self._modal.handle_event(event)
            if result == 'ok':
                if self._modal_type == 'quit':
                    self._modal = None
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                    self.app.pop_to_root()
                    return
                elif self._modal_type == 'kickoff':
                    self._modal = None
                else:
                    self._handle_save_ok()
            elif result == 'cancel':
                self._modal = None
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s and not self.spectating:
                self._open_save_modal()
            elif event.key == pygame.K_ESCAPE:
                self._open_quit_modal()
                return

        # Log panel scroll
        if event.type == pygame.MOUSEWHEEL:
            if self.log_rect.collidepoint(pygame.mouse.get_pos()):
                self.log_renderer.handle_scroll(-event.y)
                return

        # Hover always updates regardless of turn — drives the info panels
        if event.type == pygame.MOUSEMOTION:
            bench_player = self._get_bench_player_at(event.pos)
            if bench_player is not None:
                self.ui_state.hover_player = bench_player
            else:
                self.input_handler.handle(
                    event, self.game, self.ui_state,
                    self.action_buttons, self.ui_state.player_dots
                )
            for btn in self._board_block_dice:
                btn.update_hover(event.pos)
            return

        if self.spectating or self.game.state.game_over:
            return

        # Human input (clicks / keys)
        if self.is_human_turn():
            # Check pass mode toggle (non-game-action, just switches UI mode)
            if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and
                    self._pass_toggle and self._pass_toggle.is_clicked(event.pos)):
                self.ui_state.special_toggle = (
                    None if self.ui_state.special_toggle == 'pass' else 'pass'
                )
                self.ui_state.selected_action_type = None
                self.ui_state.selected_action_choice = None
                self._rebuild_buttons()
                return

            # Check board block dice clicks (post-roll die selection on the board)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for btn in self._board_block_dice:
                    if btn.is_clicked(event.pos):
                        ac = btn.action
                        self.game.step(Action(ac.action_type))
                        self._scan_kickoff_events()
                        self.ui_state.reset_selection()
                        self._rebuild_buttons()
                        return

            # Check bench clicks first (for PLACE_PLAYER and info display)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                bench_player = self._get_bench_player_at(event.pos)
                if bench_player is not None:
                    self.ui_state.selected_player = bench_player
                    if bench_player.team == self.game.state.home_team:
                        self.ui_state.pinned_home_player = bench_player
                    else:
                        self.ui_state.pinned_away_player = bench_player
                    self.ui_state.selected_action_type = None
                    self.ui_state.selected_action_choice = None
                    self.ui_state.highlighted_squares = []
                    self.ui_state.player_dots = []
                    self._rebuild_buttons()
                    return
            action = self.input_handler.handle(
                event, self.game, self.ui_state,
                self.action_buttons, self.ui_state.player_dots
            )
            if action is not None:
                self.game.step(action)
                self._scan_kickoff_events()
                self.ui_state.reset_selection()
                self._resolve_pending()
                self._rebuild_buttons()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Player switched (projected mode) or selection changed — rebuild overlays
                self._rebuild_buttons()

    def draw(self, surface: pygame.Surface):
        surface.fill((10, 10, 15))

        # Board
        weather = self.game.state.weather.name if self.game.state.weather else 'NICE'
        self.board_renderer.draw_board(surface, self.game, weather)

        # Highlights (probability-colored for MOVE and PASS)
        if self.ui_state.highlighted_squares:
            at = self.ui_state.selected_action_type
            if at:
                color = highlight_color_for_action(at)
            elif self.ui_state.projected_paths:
                color = highlight_color_for_action(ActionType.MOVE)  # projected: MOVE-style tint
            else:
                color = (0, 200, 0, 100)
            probs = self._highlight_probs if self._highlight_probs else None
            self.board_renderer.draw_highlights(surface,
                                                self.ui_state.highlighted_squares,
                                                color, probs=probs)

        # Combined pass mode: show receiver squares as blue overlay on top of movement tiles
        if (self._pass_receiver_squares and
                self.ui_state.pass_mode_pass_ac is not None and
                self.ui_state.special_toggle != 'pass'):
            receiver_probs = {sq: self._pass_probs[sq]
                              for sq in self._pass_receiver_squares if sq in self._pass_probs}
            self.board_renderer.draw_highlights(surface, self._pass_receiver_squares,
                                                (60, 60, 220, 80),
                                                probs=receiver_probs if receiver_probs else None)


        # Path hover (MOVE) — drawn before players so line is under player sprite
        if self.ui_state.hover_path:
            sp = self.ui_state.selected_player or self.game.get_active_player()
            player_sq = sp.position if sp else None
            self.board_renderer.draw_path(surface, self.ui_state.hover_path,
                                          player_square=player_sq)

        # Bench players on side crowd columns
        self.player_renderer.draw_bench_on_board(
            surface, self.game, self.game.state.away_team, is_home=False,
            selected_player=self.ui_state.pinned_away_player,
            show_numbers=self._show_numbers
        )
        self.player_renderer.draw_bench_on_board(
            surface, self.game, self.game.state.home_team, is_home=True,
            selected_player=self.ui_state.pinned_home_player,
            show_numbers=self._show_numbers
        )

        # Players and ball
        self.player_renderer.draw_players(
            surface, self.game,
            selected_player=self.ui_state.selected_player,
            ui_state=self.ui_state
        )
        self.player_renderer.draw_ball(surface, self.game)

        # Board block dice (post-roll die selection) drawn after players, above attacker
        for btn in self._board_block_dice:
            btn.draw(surface)

        # Hover highlight on valid target squares
        if (self.ui_state.hover_square is not None and
                self.ui_state.hover_square in self.ui_state.highlighted_squares):
            self.board_renderer.draw_hover_highlight(surface, self.ui_state.hover_square)

        # Block dice count badges on blitz/block target squares
        if self._block_dice_pairs:
            self.board_renderer.draw_block_dice_overlays(surface, self._block_dice_pairs)

        # Dice roll labels (on by default)
        at = self.ui_state.selected_action_type
        if self._highlight_rolls:
            self.board_renderer.draw_roll_labels(surface, self._highlight_rolls)
        if (self._pass_receiver_rolls and
                self.ui_state.pass_mode_pass_ac is not None and
                self.ui_state.special_toggle != 'pass'):
            self.board_renderer.draw_roll_labels(surface, self._pass_receiver_rolls)

        # Probability % text (hidden setting, off by default)
        if self._show_probs:
            if self.ui_state.selected_action_type == ActionType.PASS and self._highlight_probs:
                self.board_renderer.draw_prob_labels(surface, self._highlight_probs)
            elif (self._pass_receiver_squares and
                    self.ui_state.pass_mode_pass_ac is not None and
                    self.ui_state.special_toggle != 'pass'):
                receiver_probs = {sq: self._pass_probs[sq]
                                  for sq in self._pass_receiver_squares if sq in self._pass_probs}
                self.board_renderer.draw_prob_labels(surface, receiver_probs)

        # Pass arrow: rotated rectangle from passer → hovered pass target.
        # Shown in toggle-ON / standalone PASS mode (hover_pass_rolls is set) AND
        # in toggle-OFF mode when hovering a receiver square.
        _arrow_rolls = None
        _arrow_prob  = 1.0
        _arrow_sq    = self.ui_state.hover_square

        if self.ui_state.hover_pass_rolls and _arrow_sq:
            # Toggle ON or standalone PASS mode
            _arrow_rolls = self.ui_state.hover_pass_rolls
            for r in _arrow_rolls:
                _arrow_prob *= (7 - r) / 6
        elif (_arrow_sq is not None and
              _arrow_sq in self._pass_receiver_squares and
              self.ui_state.pass_mode_pass_ac is not None and
              self.ui_state.special_toggle != 'pass'):
            # Toggle OFF: hovering a friendly receiver square
            _arrow_prob = self._pass_probs.get(_arrow_sq, 1.0)
            pass_positions = self.ui_state.pass_mode_pass_ac.positions or []
            if _arrow_sq in pass_positions:
                idx = pass_positions.index(_arrow_sq)
                all_rolls = self.ui_state.pass_mode_pass_ac.rolls or []
                if idx < len(all_rolls):
                    _arrow_rolls = all_rolls[idx]
        else:
            _arrow_sq = None  # nothing to draw

        if _arrow_sq is not None:
            passer = self.ui_state.selected_player or self.game.get_active_player()
            if passer and passer.position:
                ts = self.tile_size
                ox, oy = self.pitch_offset
                p1 = sq_to_px(passer.position, ts, (ox, oy))
                p1 = (p1[0] + ts // 2, p1[1] + ts // 2)
                p2 = sq_to_px(_arrow_sq, ts, (ox, oy))
                p2 = (p2[0] + ts // 2, p2[1] + ts // 2)
                plabel = _pass_type_label(passer.position, _arrow_sq)
                self.board_renderer.draw_pass_arrow(surface, p1, p2, _arrow_prob,
                                                    pass_label=plabel,
                                                    rolls=_arrow_rolls)

        # projected_player_dots are kept for smart-inference logic but not rendered

        # HUD
        self.hud_renderer.draw(surface, self.game)

        # Context row (top crowd row) + buttons (bottom crowd row)
        self.action_bar_renderer.draw(surface, self.action_buttons, self.game,
                                      btns_rect=self._btns_rect)

        # Pass mode labeled toggle (rendered on top of action bar)
        if self._pass_toggle:
            self._pass_toggle.draw(surface)

        # Split player info: away on left, home on right.
        # Hover always takes precedence for that team's panel; falls back to pinned player.
        hover = self.ui_state.hover_player
        home_team = self.game.state.home_team
        away_display = (hover if (hover is not None and hover.team != home_team)
                        else self.ui_state.pinned_away_player)
        home_display = (hover if (hover is not None and hover.team == home_team)
                        else self.ui_state.pinned_home_player)
        self.away_info_renderer.draw(surface, away_display, self.game)
        self.home_info_renderer.draw(surface, home_display, self.game)

        # Event log
        self.log_renderer.draw(surface, self.game)

        # Spectating label (in the buttons row, not context row)
        if self.spectating:
            font = get_body_font(12)
            lbl = font.render('SPECTATING', True, (150, 150, 150))
            surface.blit(lbl, (self._btns_rect.x + 4,
                                self._btns_rect.y + (self._btns_rect.height - lbl.get_height()) // 2))

        # Game over overlay
        if self.game.state.game_over:
            self._draw_game_over(surface)

        # Path/pass probability label near cursor (hidden setting, off by default)
        if self._show_probs:
            if self.ui_state.hover_path:
                self.board_renderer.draw_path_prob(
                    surface, self.ui_state.hover_path, pygame.mouse.get_pos())
            if self.ui_state.hover_pass_rolls:
                self.board_renderer.draw_pass_prob(
                    surface, self.ui_state.hover_pass_rolls, pygame.mouse.get_pos())

        # Cursor: ball when hovering a pass target, crosshair when hovering a combat target
        hover_sq = self.ui_state.hover_square
        want_ball = bool(self.ui_state.hover_pass_rolls) or (
            self.ui_state.hover_square in self._pass_receiver_squares and
            self.ui_state.pass_mode_pass_ac is not None and
            self.ui_state.special_toggle != 'pass'
        )
        want_block = want_foul = want_handoff = False
        if not want_ball and hover_sq is not None and hover_sq in self.ui_state.highlighted_squares:
            at = self.ui_state.selected_action_type
            if at in _BLOCK_CURSOR_TYPES:
                want_block = True
            elif at in _FOUL_CURSOR_TYPES:
                want_foul = True
            elif at in _HANDOFF_CURSOR_TYPES:
                want_handoff = True
            elif at is None and self.ui_state.projected_paths:
                # Projected mode: show block cursor over blitz target squares
                for path in self.ui_state.projected_paths:
                    if path.steps and path.steps[-1] == hover_sq and path.block_dice is not None:
                        want_block = True
                        break
            elif at is None:
                for ac in self.game.state.available_actions:
                    if ac.action_type in _BLOCK_CURSOR_TYPES and hover_sq in (ac.positions or []):
                        want_block = True
                        break
                    if ac.action_type in _FOUL_CURSOR_TYPES and hover_sq in (ac.positions or []):
                        want_foul = True
                        break
                    if ac.action_type in _HANDOFF_CURSOR_TYPES and hover_sq in (ac.positions or []):
                        want_handoff = True
                        break
        new_cursor = ('pass' if want_ball else
                      'block' if want_block else
                      'foul' if want_foul else
                      'handoff' if want_handoff else 'arrow')
        if new_cursor != self._cursor_state:
            self._cursor_state = new_cursor
            if new_cursor == 'pass':
                if self._pass_cursor is None:
                    self._pass_cursor = pygame.cursors.Cursor(
                        (8, 8), get_action_icon('START_PASS', (16, 16)))
                pygame.mouse.set_cursor(self._pass_cursor)
            elif new_cursor == 'block':
                if self._block_cursor is None:
                    self._block_cursor = pygame.cursors.Cursor(
                        (8, 8), get_action_icon('START_BLOCK', (16, 16)))
                pygame.mouse.set_cursor(self._block_cursor)
            elif new_cursor == 'foul':
                if self._foul_cursor is None:
                    self._foul_cursor = pygame.cursors.Cursor(
                        (8, 8), get_action_icon('START_FOUL', (16, 16)))
                pygame.mouse.set_cursor(self._foul_cursor)
            elif new_cursor == 'handoff':
                if self._handoff_cursor is None:
                    self._handoff_cursor = pygame.cursors.Cursor(
                        (8, 8), get_action_icon('START_HANDOFF', (16, 16)))
                pygame.mouse.set_cursor(self._handoff_cursor)
            else:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        # Modal
        if self._modal:
            self._modal.draw(surface)

    def _draw_game_over(self, surface: pygame.Surface):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        surface.blit(overlay, (0, 0))

        home = self.game.state.home_team
        away = self.game.state.away_team
        hs, as_ = home.state.score, away.state.score
        if hs > as_:
            msg = f'{home.name} wins!'
        elif as_ > hs:
            msg = f'{away.name} wins!'
        else:
            msg = 'Draw!'

        score_str = f'{away.name} {as_} - {hs} {home.name}'
        msg_surf = get_font(40, bold=True).render(msg, True, (255, 220, 100))
        score_surf = get_font(24).render(score_str, True, (200, 200, 200))
        esc_surf = get_body_font(16).render('Press ESC to return', True, (160, 160, 160))

        cx, cy = self.width // 2, self.height // 2
        surface.blit(msg_surf, (cx - msg_surf.get_width() // 2, cy - 50))
        surface.blit(score_surf, (cx - score_surf.get_width() // 2, cy))
        surface.blit(esc_surf, (cx - esc_surf.get_width() // 2, cy + 40))

    def _get_bench_player_at(self, pos: tuple):
        """Return the bench player at screen pos (crowd rows on the board), or None."""
        ts = self.tile_size
        ox, oy = self.pitch_offset
        arena = self.game.arena

        for team, is_home in [
            (self.game.state.away_team, False),
            (self.game.state.home_team, True),
        ]:
            off_pitch = [p for p in team.players if p.position is None]
            if is_home:
                y_tile = arena.height - 1
                x_tile = arena.width - 2
                dx = -1
            else:
                y_tile = 0
                x_tile = 1
                dx = 1
            for player in off_pitch:
                if is_home and x_tile < 1:
                    break
                if not is_home and x_tile > arena.width - 2:
                    break
                px = ox + x_tile * ts
                py = oy + y_tile * ts
                if pygame.Rect(px, py, ts, ts).collidepoint(pos):
                    return player
                x_tile += dx
        return None

    def _open_quit_modal(self):
        screen_size = (self.width, self.height)
        self._modal = Modal(screen_size, 'Exit Game?', None,
                            ok_label='Exit', cancel_label='Cancel',
                            description='Any unsaved data will be lost.',
                            ok_variant='red')
        self._modal_type = 'quit'

    def _open_save_modal(self):
        screen_size = (self.width, self.height)
        modal_w, modal_h = 360, 304
        modal_x = (screen_size[0] - modal_w) // 2
        modal_y = (screen_size[1] - modal_h) // 2
        content_top = modal_y + int(modal_h * 0.30)
        self._save_input = TextInput(
            pygame.Rect(modal_x + int(modal_w * 0.10), content_top + 48,
                        int(modal_w * 0.80), 34),
            placeholder='Enter save name...',
            max_len=40
        )
        self._modal = Modal(screen_size, 'Save Game', self._save_input,
                            ok_label='Save', cancel_label='Cancel')
        self._modal_type = 'save'

    def _handle_save_ok(self):
        name = self._save_input.text.strip() if self._save_input else ''
        if len(name) < 3:
            self._modal.error_message = 'Name must be at least 3 characters.'
            return
        try:
            save_game(self.game, name)
            self._modal = None
        except Exception as e:
            self._modal.error_message = f'Error saving: {e}'

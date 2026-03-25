"""
Main game screen — handles rendering and input for all game phases.
Supports AI vs AI (spectating), human vs AI, and human vs human (hotseat).
"""
from __future__ import annotations
import pygame
from typing import Optional

from botbowl.core.model import Action
from botbowl.core.table import ActionType
from botbowl.gui.input_handler import InputHandler, UIState
from botbowl.gui.rendering.board import BoardRenderer, sq_to_px, highlight_color_for_action
from botbowl.gui.rendering.players import PlayerRenderer
from botbowl.gui.rendering.hud import HUDRenderer, ActionBarRenderer, PlayerInfoRenderer
from botbowl.gui.rendering.log_panel import LogPanelRenderer
from botbowl.gui.rendering.buttons import (
    build_action_buttons, build_player_action_dots, POSITIONAL_ACTIONS, START_ACTIONS
)
from botbowl.gui.rendering.ui_primitives import (
    Button, Modal, TextInput,
    COLOR_BTN_NEUTRAL, COLOR_BTN_DEFAULT
)
from botbowl.gui.save_load import save_game, save_exists, list_saves
from botbowl.gui.assets import prettify

TILE_SIZE = 30
HUD_H = 72          # Scoreboard height
ACTION_BAR_H = 44   # Action buttons bar
INFO_H = 155        # Bottom panel height
INFO_PLAYER_W = 200 # Width of each player info panel (log takes the rest)


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
        total_h = HUD_H + self.pitch_px_h + ACTION_BAR_H + INFO_H

        # Resize window if needed
        self._resize_display(total_w, total_h)
        self.width = total_w
        self.height = total_h

        # Pitch origin sits directly below the HUD
        self.pitch_offset = (0, HUD_H)

        # Sub-rects
        self.hud_rect = pygame.Rect(0, 0, total_w, HUD_H)
        self.pitch_rect = pygame.Rect(0, HUD_H, self.pitch_px_w, self.pitch_px_h)
        self.action_bar_rect = pygame.Rect(0, HUD_H + self.pitch_px_h,
                                           total_w, ACTION_BAR_H)
        # Bottom panel: away info | log (remainder) | home info
        info_y = HUD_H + self.pitch_px_h + ACTION_BAR_H
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
        self._save_input: Optional[TextInput] = None
        self._save_error: str = ''
        self._game_over_displayed = False

        # Initial button build
        self._rebuild_buttons()

    def _resize_display(self, w: int, h: int):
        current = pygame.display.get_surface()
        if current is None or current.get_size() != (w, h):
            pygame.display.set_mode((w, h))

    def _rebuild_buttons(self):
        self.action_buttons = build_action_buttons(
            self.game.state.available_actions, self.game,
            self.action_bar_rect
        )
        self.ui_state.player_dots = build_player_action_dots(
            self.ui_state.selected_player,
            self.game.state.available_actions,
            self.tile_size, self.pitch_offset
        )

        # Auto-highlight the single available positional action's squares.
        # Covers MOVE (after START_MOVE), PLACE_BALL, PUSH, FOLLOW_UP, PLACE_PLAYER, etc.
        if not self.ui_state.selected_action_type:
            positional = [ac for ac in self.game.state.available_actions
                          if ac.action_type in POSITIONAL_ACTIONS]
            if len(positional) == 1:
                ac = positional[0]
                self.ui_state.selected_action_type = ac.action_type
                self.ui_state.selected_action_choice = ac
                self.ui_state.highlighted_squares = [sq for sq in (ac.positions or []) if sq is not None]

    def is_human_turn(self) -> bool:
        if self.spectating:
            return False
        actor = self.game.actor
        return actor is not None and actor.human

    def update(self):
        if self.game.state.game_over:
            return

        # Advance intermediate procedures that require no action (fast_mode=False).
        # game.step() breaks on intermediate steps when fast_mode=False, so we must
        # call step(None) repeatedly until available_actions is non-empty or game is over.
        if not self.game.state.available_actions:
            self.game.step(None)
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
                self._rebuild_buttons()
                if self.ai_delay_ms > 0:
                    pygame.time.wait(self.ai_delay_ms)

    def handle_event(self, event: pygame.event.Event):
        # Handle modal first
        if self._modal:
            result = self._modal.handle_event(event)
            if result == 'ok':
                self._handle_save_ok()
            elif result == 'cancel':
                self._modal = None
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s and not self.spectating:
                self._open_save_modal()
            elif event.key == pygame.K_ESCAPE:
                self.app.pop_screen()
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
            return

        if self.spectating or self.game.state.game_over:
            return

        # Human input (clicks / keys)
        if self.is_human_turn():
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
                self.ui_state.reset_selection()
                self._rebuild_buttons()

    def draw(self, surface: pygame.Surface):
        surface.fill((10, 10, 15))

        # Board
        weather = self.game.state.weather.name if self.game.state.weather else 'NICE'
        self.board_renderer.draw_board(surface, self.game, weather,
                                       self.ui_state.grid_mode)

        # Highlights
        if self.ui_state.highlighted_squares:
            at = self.ui_state.selected_action_type
            color = highlight_color_for_action(at) if at else (0, 200, 0, 100)
            self.board_renderer.draw_highlights(surface,
                                                self.ui_state.highlighted_squares,
                                                color)

        # Path hover
        if self.ui_state.hover_path:
            self.board_renderer.draw_path(surface, self.ui_state.hover_path)

        # Bench players on crowd rows (row 0 = away, row height-1 = home)
        self.player_renderer.draw_bench_on_board(
            surface, self.game, self.game.state.away_team, is_home=False,
            selected_player=self.ui_state.pinned_away_player
        )
        self.player_renderer.draw_bench_on_board(
            surface, self.game, self.game.state.home_team, is_home=True,
            selected_player=self.ui_state.pinned_home_player
        )

        # Players and ball
        self.player_renderer.draw_players(
            surface, self.game,
            selected_player=self.ui_state.selected_player,
            ui_state=self.ui_state
        )
        self.player_renderer.draw_ball(surface, self.game)

        # Hover highlight on valid target squares
        if (self.ui_state.hover_square is not None and
                self.ui_state.hover_square in self.ui_state.highlighted_squares):
            self.board_renderer.draw_hover_highlight(surface, self.ui_state.hover_square)

        # Player action panel + dots
        if self.ui_state.player_dots:
            rects = [btn.rect for btn in self.ui_state.player_dots]
            union = rects[0].unionall(rects[1:])
            panel = union.inflate(8, 8)
            pygame.draw.rect(surface, (25, 25, 35), panel, border_radius=8)
            pygame.draw.rect(surface, (70, 70, 100), panel, width=1, border_radius=8)
        for btn in self.ui_state.player_dots:
            btn.draw(surface)

        # HUD
        self.hud_renderer.draw(surface, self.game)

        # Action bar + buttons
        self.action_bar_renderer.draw(surface, self.action_buttons)

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

        # Spectating label
        if self.spectating:
            font = pygame.font.SysFont('Arial', 12)
            lbl = font.render('SPECTATING', True, (150, 150, 150))
            surface.blit(lbl, (self.action_bar_rect.x + 4,
                                self.action_bar_rect.y + (ACTION_BAR_H - lbl.get_height()) // 2))

        # Game over overlay
        if self.game.state.game_over:
            self._draw_game_over(surface)

        # Modal
        if self._modal:
            self._modal.draw(surface)

    def _draw_game_over(self, surface: pygame.Surface):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        surface.blit(overlay, (0, 0))

        font = pygame.font.SysFont('Arial', 40, bold=True)
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
        msg_surf = font.render(msg, True, (255, 220, 100))
        score_surf = pygame.font.SysFont('Arial', 24).render(score_str, True, (200, 200, 200))
        esc_surf = pygame.font.SysFont('Arial', 16).render('Press ESC to return', True, (160, 160, 160))

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

    def _open_save_modal(self):
        screen_size = (self.width, self.height)
        self._save_input = TextInput(
            pygame.Rect(screen_size[0] // 2 - 180,
                        screen_size[1] // 2 - 20,
                        360, 34),
            placeholder='Enter save name...',
            max_len=40
        )
        self._modal = Modal(screen_size, 'Save Game', self._save_input,
                            ok_label='Save', cancel_label='Cancel')

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

"""
Game creation screen — mode selection, team/agent picker, roster preview.
"""
from __future__ import annotations
import copy
import os
import pygame
from typing import Optional

import botbowl
from botbowl.gui.gui import SCREEN_W, SCREEN_H
from botbowl.core.load import load_all_teams, load_rule_set
from botbowl.core.model import Agent, Action
from botbowl.core.table import ActionType
from botbowl.ai.registry import list_bots
from botbowl.gui.rendering.ui_primitives import (
    Button, draw_bordered_rect, get_button_image,
    COLOR_BTN_DEFAULT, COLOR_BTN_HOME, COLOR_BTN_AWAY,
    COLOR_BTN_NEUTRAL, COLOR_TEXT, COLOR_TEXT_DIM, COLOR_PANEL_BG, COLOR_BORDER
)
from botbowl.gui.screens.team_utils import load_team_logo, calc_team_tv, draw_roster_table

PADDING = 24
INNER_PAD = 14
BOTTOM_BAR_H = 64
TITLE_H = 62

TAB_BAR_H = 36
TAB_W = 120
TAB_GAP = 4

CARD_W = SCREEN_W - 2 * PADDING          # 792
CARD_Y = TITLE_H + TAB_BAR_H             # 98
CARD_H = SCREEN_H - CARD_Y - BOTTOM_BAR_H  # 589

DD_GAP     = 8
DD_RACE_W  = (CARD_W - 2 * INNER_PAD - 2 * DD_GAP) // 4        # ~187
DD_TEAM_W  = (CARD_W - 2 * INNER_PAD - 2 * DD_GAP) // 2        # ~374
DD_AGENT_W = (CARD_W - 2 * INNER_PAD - 2 * DD_GAP) // 4        # ~187
DD_H = 30
LOGO_SIZE = 54

_TAB_RECTS = [
    pygame.Rect(PADDING,                    TITLE_H, TAB_W, TAB_BAR_H),
    pygame.Rect(PADDING + TAB_W + TAB_GAP,  TITLE_H, TAB_W, TAB_BAR_H),
]


def _card_layout():
    """Returns a dict of shared y-positions for card content (screen-absolute)."""
    cy = CARD_Y + INNER_PAD          # 76
    logo_block_h = LOGO_SIZE + 14    # 68
    lbl_gap = 3

    # Team and Agent dropdowns sit on the SAME row (side-by-side)
    y_dd_lbl    = cy + logo_block_h              # 144
    y_dd_row    = y_dd_lbl + 16 + lbl_gap        # 163  (both dropdowns at this y)
    y_separator = y_dd_row + DD_H + 10
    y_roster    = y_separator + 8
    return dict(cy=cy, logo_block_h=logo_block_h,
                y_dd_lbl=y_dd_lbl, y_dd_row=y_dd_row,
                y_separator=y_separator, y_roster=y_roster)

_GAME_MODES = [
    {'label': '1v1',   'config': 'gym-1',    'board_size': 1,  'desc': '4 × 3',   'pitch_img': 'nice-4x3.jpg'},
    {'label': '3v3',   'config': 'gym-3',    'board_size': 3,  'desc': '12 × 5',  'pitch_img': 'nice-12x5.jpg'},
    {'label': '5v5',   'config': 'gym-5',    'board_size': 5,  'desc': '16 × 9',  'pitch_img': 'nice-16x9.jpg'},
    {'label': '7v7',   'config': 'gym-7',    'board_size': 7,  'desc': '20 × 9',  'pitch_img': 'nice-20x9.jpg'},
    {'label': '11v11', 'config': 'bot-bowl', 'board_size': 11, 'desc': '26 × 15', 'pitch_img': 'nice-26x15.jpg'},
]

_PITCH_IMG_BASE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'img', 'arenas', 'pitch'))
_PITCH_CACHE: dict = {}


def _load_pitch_native(filename: str) -> Optional[pygame.Surface]:
    """Load pitch image at its native 1:1 pixel size."""
    key = ('pitch_native', filename)
    if key in _PITCH_CACHE:
        return _PITCH_CACHE[key]
    path = os.path.join(_PITCH_IMG_BASE, filename)
    if not os.path.exists(path):
        _PITCH_CACHE[key] = None
        return None
    try:
        img = pygame.image.load(path).convert()
        _PITCH_CACHE[key] = img
        return img
    except Exception:
        _PITCH_CACHE[key] = None
        return None


# Team accent colors
COLOR_HOME_ACCENT = (80, 120, 220)
COLOR_AWAY_ACCENT = (220, 80, 60)
COLOR_HOME_CARD_BG = (20, 20, 24)
COLOR_AWAY_CARD_BG = (20, 20, 24)
COLOR_HOME_TEXT = (200, 210, 255)
COLOR_AWAY_TEXT = (255, 210, 200)

from botbowl.gui.fonts import get_font as _fancy, get_body_font as _font


class Dropdown:
    """Simple dropdown widget with optional per-item colored suffixes.

    Shows at most *max_visible* items; scrolls when the list is longer.
    Mouse-wheel events can be forwarded via handle_scroll().
    """

    MAX_VISIBLE = 8   # max rows shown at once in open list

    def __init__(self, rect: pygame.Rect, options: list[str],
                 selected: int = 0, font_size: int = 13,
                 suffixes: Optional[list] = None):
        self.rect = rect
        self.options = options
        self.selected = selected
        self.font_size = font_size
        self.open = False
        self.suffixes = suffixes  # list of (text, color) or None per item
        self._scroll_offset = 0   # index of first visible item

    def _clamp_scroll(self):
        n = len(self.options)
        max_off = max(0, n - self.MAX_VISIBLE)
        self._scroll_offset = max(0, min(max_off, self._scroll_offset))

    def _draw_item(self, surface, font, label, suffix_entry, x, y, w, h):
        """Draw a single dropdown item (header or list row)."""
        txt_surf = font.render(label[:50], True, COLOR_TEXT)
        surface.blit(txt_surf, (x + 8, y + (h - txt_surf.get_height()) // 2))
        if suffix_entry:
            suf_text, suf_color = suffix_entry
            suf_surf = font.render(suf_text, True, suf_color)
            surface.blit(suf_surf, (x + 8 + txt_surf.get_width(),
                                    y + (h - suf_surf.get_height()) // 2))

    def draw(self, surface: pygame.Surface):
        # Header button
        draw_bordered_rect(surface, (52, 56, 75), (90, 95, 125), self.rect, radius=4, border_w=1)
        font = _font(self.font_size)
        label = self.options[self.selected] if self.options else ''
        suffix = (self.suffixes[self.selected]
                  if self.suffixes and self.selected < len(self.suffixes) else None)
        self._draw_item(surface, font, label, suffix,
                        self.rect.x, self.rect.y, self.rect.width, self.rect.height)
        # Draw a simple downward triangle as the dropdown arrow
        ax = self.rect.right - 14
        ay = self.rect.centery
        pygame.draw.polygon(surface, (160, 160, 160),
                            [(ax - 5, ay - 3), (ax + 5, ay - 3), (ax, ay + 4)])

        if not self.open or not self.options:
            return

        self._clamp_scroll()
        item_h   = self.rect.height
        n        = len(self.options)
        visible  = min(self.MAX_VISIBLE, n)
        list_h   = visible * item_h

        # Flip above button if not enough room below
        screen_h = surface.get_height()
        if self.rect.bottom + list_h > screen_h:
            list_y = self.rect.top - list_h
        else:
            list_y = self.rect.bottom

        # Draw visible items
        for vis_i in range(visible):
            data_i = self._scroll_offset + vis_i
            opt    = self.options[data_i]
            item_rect = pygame.Rect(self.rect.x, list_y + vis_i * item_h,
                                    self.rect.width, item_h)
            bg = (75, 85, 125) if data_i == self.selected else (48, 52, 72)
            pygame.draw.rect(surface, bg, item_rect)
            pygame.draw.rect(surface, (90, 95, 125), item_rect, 1)
            suf = (self.suffixes[data_i]
                   if self.suffixes and data_i < len(self.suffixes) else None)
            self._draw_item(surface, font, opt, suf,
                            item_rect.x, item_rect.y, item_rect.width, item_rect.height)

        # Scroll indicator (thin bar on right edge)
        if n > self.MAX_VISIBLE:
            bar_h   = max(20, list_h * self.MAX_VISIBLE // n)
            bar_y   = list_y + (list_h - bar_h) * self._scroll_offset // max(1, n - self.MAX_VISIBLE)
            bar_x   = self.rect.right - 5
            pygame.draw.rect(surface, (110, 115, 145),
                             pygame.Rect(bar_x, bar_y, 3, bar_h), border_radius=2)

    def handle_click(self, pos: tuple) -> bool:
        """Returns True if selection changed."""
        if self.rect.collidepoint(pos):
            self.open = not self.open
            if self.open:
                # Scroll so selected item is visible
                if self.selected < self._scroll_offset:
                    self._scroll_offset = self.selected
                elif self.selected >= self._scroll_offset + self.MAX_VISIBLE:
                    self._scroll_offset = self.selected - self.MAX_VISIBLE + 1
                self._clamp_scroll()
            return False

        if self.open:
            self._clamp_scroll()
            item_h  = self.rect.height
            n       = len(self.options)
            visible = min(self.MAX_VISIBLE, n)
            list_h  = visible * item_h
            screen_h = pygame.display.get_surface().get_height()
            list_y  = (self.rect.top - list_h
                       if self.rect.bottom + list_h > screen_h
                       else self.rect.bottom)
            for vis_i in range(visible):
                data_i    = self._scroll_offset + vis_i
                item_rect = pygame.Rect(self.rect.x, list_y + vis_i * item_h,
                                        self.rect.width, item_h)
                if item_rect.collidepoint(pos):
                    changed       = (data_i != self.selected)
                    self.selected = data_i
                    self.open     = False
                    return changed
            self.open = False
        return False

    def handle_scroll(self, pos: tuple, dy: int) -> bool:
        """Forward MOUSEWHEEL event when the cursor is over the open list.
        Returns True if scroll was consumed."""
        if not self.open:
            return False
        item_h  = self.rect.height
        list_h  = min(self.MAX_VISIBLE, len(self.options)) * item_h
        screen_h = pygame.display.get_surface().get_height()
        list_y  = (self.rect.top - list_h
                   if self.rect.bottom + list_h > screen_h
                   else self.rect.bottom)
        list_rect = pygame.Rect(self.rect.x, list_y, self.rect.width, list_h)
        if list_rect.collidepoint(pos):
            self._scroll_offset -= dy
            self._clamp_scroll()
            return True
        return False

    def close(self):
        self.open = False

    @property
    def value(self) -> Optional[str]:
        return self.options[self.selected] if self.options else None


_MODE_DD_W = 220
_MODE_DD_H = 38
_MODE_DD_Y = 110   # balanced below title/subtitle


class CreateGameScreen:
    """Two-step game creation: mode selection then team/agent selection."""

    def __init__(self, app, step: int = 0):
        self.app = app
        self._step = 0           # 0 = mode, 1 = team/agent
        self._selected_mode = 4  # default 11v11

        dd_x = (SCREEN_W - _MODE_DD_W) // 2
        self._mode_dd = Dropdown(
            pygame.Rect(dd_x, _MODE_DD_Y, _MODE_DD_W, _MODE_DD_H),
            [m['label'] for m in _GAME_MODES],
            selected=4, font_size=15)

        btn_y = SCREEN_H - 54
        self._mode_back_btn = Button(pygame.Rect(PADDING, btn_y, 120, 42),
                                     label='Back', font_size=15, bg_image=get_button_image())
        self._mode_select_btn = Button(pygame.Rect(SCREEN_W - PADDING - 160, btn_y, 160, 42),
                                       label='Select', font_size=15, bg_image=get_button_image())

        # Step 1 state
        self._teams: list = []
        self._races: list[str] = []
        self._teams_by_race: dict = {}
        self._home_filtered_teams: list = []
        self._away_filtered_teams: list = []
        self._reroll_costs: dict = {}
        self._bot_names: list[str] = ['Human'] + list_bots()
        self._home_race_dd: Optional[Dropdown] = None
        self._home_team_dd: Optional[Dropdown] = None
        self._home_agent_dd: Optional[Dropdown] = None
        self._away_race_dd: Optional[Dropdown] = None
        self._away_team_dd: Optional[Dropdown] = None
        self._away_agent_dd: Optional[Dropdown] = None
        self._start_btn: Optional[Button] = None
        self._back_btn: Optional[Button] = None
        self._active_tab: int = 0   # 0 = Home, 1 = Away
        self._tab_from: int = 0
        self._tab_anim_t: float = 1.0   # 1.0 = settled
        self._tab_anim_start_ms: int = 0

        if step == 1:
            self._step = 1
            self._load_teams_for_mode()

    def _card_rect(self, is_home: bool) -> pygame.Rect:
        x = PADDING if is_home else PADDING + CARD_W + CARD_GAP
        return pygame.Rect(x, CARD_Y, CARD_W, CARD_H)

    def _team_suffixes(self, teams: list) -> list:
        """Build list of (tv_text, color) tuples for team dropdown suffixes."""
        return [(f'  {calc_team_tv(t, self._reroll_costs) // 1000}k', (200, 175, 80))
                for t in teams]

    def _make_team_dd(self, rect: pygame.Rect, teams: list, selected: int = 0) -> Dropdown:
        return Dropdown(rect, [t.name for t in teams],
                        selected=selected, suffixes=self._team_suffixes(teams))

    def _load_teams_for_mode(self):
        mode = _GAME_MODES[self._selected_mode]
        config = botbowl.load_config(mode['config'])
        ruleset = botbowl.load_rule_set(config.ruleset)
        self._teams = load_all_teams(ruleset, board_size=mode['board_size'])
        # Build race → reroll_cost map for TV calculation
        self._reroll_costs = {r.name: r.reroll_cost for r in ruleset.races}

        # Build race → teams mapping, sorted by TTV
        raw: dict = {}
        for team in self._teams:
            tv = calc_team_tv(team, self._reroll_costs)
            raw.setdefault(team.race, []).append((tv, team))
        self._teams_by_race = {
            race: [t for _, t in sorted(entries)]
            for race, entries in raw.items()
        }
        self._races = sorted(self._teams_by_race.keys())

        home_race_idx = 0
        away_race_idx = min(1, len(self._races) - 1)
        self._home_filtered_teams = self._teams_by_race[self._races[home_race_idx]]
        self._away_filtered_teams = self._teams_by_race[self._races[away_race_idx]]

        self._active_tab = 0
        ly = _card_layout()
        row_y = ly['y_dd_row']

        x_race  = PADDING + INNER_PAD
        x_team  = x_race + DD_RACE_W + DD_GAP
        x_agent = x_team + DD_TEAM_W + DD_GAP

        self._home_race_dd = Dropdown(
            pygame.Rect(x_race, row_y, DD_RACE_W, DD_H), self._races,
            selected=home_race_idx)
        self._home_team_dd = self._make_team_dd(
            pygame.Rect(x_team, row_y, DD_TEAM_W, DD_H), self._home_filtered_teams)
        self._home_agent_dd = Dropdown(
            pygame.Rect(x_agent, row_y, DD_AGENT_W, DD_H), self._bot_names)

        self._away_race_dd = Dropdown(
            pygame.Rect(x_race, row_y, DD_RACE_W, DD_H), self._races,
            selected=away_race_idx)
        self._away_team_dd = self._make_team_dd(
            pygame.Rect(x_team, row_y, DD_TEAM_W, DD_H), self._away_filtered_teams)
        self._away_agent_dd = Dropdown(
            pygame.Rect(x_agent, row_y, DD_AGENT_W, DD_H), self._bot_names)

        btn_y = SCREEN_H - 54
        self._start_btn = Button(
            pygame.Rect(SCREEN_W - PADDING - 180, btn_y, 180, 42),
            label='Start Game', font_size=15, bg_image=get_button_image())
        self._back_btn = Button(
            pygame.Rect(PADDING, btn_y, 120, 42),
            label='Back', font_size=15, bg_image=get_button_image())

    def _rebuild_team_dd(self, is_home: bool):
        """Rebuild the team dropdown after a race selection change."""
        race_dd = self._home_race_dd if is_home else self._away_race_dd
        team_dd = self._home_team_dd if is_home else self._away_team_dd
        race = race_dd.value if race_dd else None
        filtered = self._teams_by_race.get(race, []) if race else []
        if is_home:
            self._home_filtered_teams = filtered
        else:
            self._away_filtered_teams = filtered
        if team_dd:
            team_dd.options = [t.name for t in filtered]
            team_dd.suffixes = self._team_suffixes(filtered)
            team_dd.selected = 0

    def _all_dropdowns(self):
        return [dd for dd in [
            self._home_race_dd, self._home_team_dd, self._home_agent_dd,
            self._away_race_dd, self._away_team_dd, self._away_agent_dd,
        ] if dd]

    def _active_dropdowns(self):
        if self._active_tab == 0:
            return [dd for dd in [self._home_race_dd, self._home_team_dd, self._home_agent_dd] if dd]
        else:
            return [dd for dd in [self._away_race_dd, self._away_team_dd, self._away_agent_dd] if dd]

    def update(self):
        pass

    def _switch_tab(self, new_tab: int):
        if new_tab == self._active_tab:
            return
        self._active_tab = new_tab
        for dd in self._all_dropdowns():
            dd.close()

    def handle_event(self, event: pygame.event.Event):
        mouse = pygame.mouse.get_pos()

        if self._step == 0:
            self._mode_back_btn.update_hover(mouse)
            self._mode_select_btn.update_hover(mouse)

            if event.type == pygame.MOUSEWHEEL:
                self._mode_dd.handle_scroll(mouse, event.y)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._mode_back_btn.is_clicked(event.pos):
                    self.app.pop_screen()
                    return
                if self._mode_select_btn.is_clicked(event.pos):
                    self._selected_mode = self._mode_dd.selected
                    self._step = 1
                    self._load_teams_for_mode()
                    return
                changed = self._mode_dd.handle_click(event.pos)
                if changed:
                    self._selected_mode = self._mode_dd.selected
                    return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self._mode_dd.open:
                        self._mode_dd.close()
                    else:
                        self.app.pop_screen()
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if not self._mode_dd.open:
                        self._selected_mode = self._mode_dd.selected
                        self._step = 1
                        self._load_teams_for_mode()

        elif self._step == 1:
            if self._start_btn:
                self._start_btn.update_hover(mouse)
            if self._back_btn:
                self._back_btn.update_hover(mouse)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Tab clicks
                for i, rect in enumerate(_TAB_RECTS):
                    if rect.collidepoint(event.pos):
                        self._switch_tab(i)
                        return
                if self._back_btn and self._back_btn.is_clicked(event.pos):
                    self._step = 0
                    for dd in self._all_dropdowns():
                        dd.close()
                    return
                if self._start_btn and self._start_btn.is_clicked(event.pos):
                    self._start_game()
                    return
                # Handle dropdowns: close others when one opens
                is_home = (self._active_tab == 0)
                race_dd = self._home_race_dd if is_home else self._away_race_dd
                for dd in self._active_dropdowns():
                    changed = dd.handle_click(event.pos)
                    if dd.open:
                        for other in self._active_dropdowns():
                            if other is not dd:
                                other.close()
                    if changed and dd is race_dd:
                        self._rebuild_team_dd(is_home)

            if event.type == pygame.KEYDOWN:
                any_open = any(dd.open for dd in self._all_dropdowns())
                if event.key == pygame.K_ESCAPE:
                    if any_open:
                        for dd in self._all_dropdowns():
                            dd.close()
                    else:
                        self._step = 0
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    self._switch_tab(1 - self._active_tab)
                elif event.key in (pygame.K_UP, pygame.K_DOWN):
                    # Navigate the team dropdown on the active tab
                    dds = self._active_dropdowns()
                    if dds:
                        dd = dds[0]  # team dropdown
                        delta = -1 if event.key == pygame.K_UP else 1
                        dd.selected = (dd.selected + delta) % max(1, len(dd.options))
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if any_open:
                        for dd in self._all_dropdowns():
                            dd.close()
                    else:
                        self._start_game()

    def _start_game(self):
        mode = _GAME_MODES[self._selected_mode]
        config_name = mode['config']
        board_size = mode['board_size']

        try:
            config = botbowl.load_config(config_name)
            config.competition_mode = False
            ruleset = botbowl.load_rule_set(config.ruleset)
            arena = botbowl.load_arena(config.arena)

            home_team = copy.deepcopy(self._home_filtered_teams[self._home_team_dd.selected % max(1, len(self._home_filtered_teams))])
            away_team = copy.deepcopy(self._away_filtered_teams[self._away_team_dd.selected % max(1, len(self._away_filtered_teams))])

            home_agent_name = self._home_agent_dd.value
            away_agent_name = self._away_agent_dd.value

            if home_agent_name == 'Human':
                home_agent = Agent('Human', human=True)
            else:
                home_agent = botbowl.make_bot(home_agent_name)

            if away_agent_name == 'Human':
                away_agent = Agent('Human', human=True)
            else:
                away_agent = botbowl.make_bot(away_agent_name)

            import uuid
            game = botbowl.Game(
                str(uuid.uuid4()),
                home_team, away_team,
                home_agent, away_agent,
                config, arena=arena, ruleset=ruleset
            )
            game.config.fast_mode = False
            game.config.pathfinding_enabled = True
            game.config.pathfinding_with_carry_ball = True
            game.init()

            spectating = (not home_agent.human and not away_agent.human)
            from botbowl.gui.screens.game_screen import GameScreen
            screen = GameScreen(
                self.app, game, home_agent, away_agent,
                spectating=spectating,
                ai_delay_ms=self.app.ai_delay_ms
            )
            self.app.push_screen(screen)
        except Exception as e:
            print(f'Error starting game: {e}')
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------ draw

    def draw(self, surface: pygame.Surface):
        surface.fill((10, 10, 16))

        if self._step == 0:
            self._draw_mode_select(surface)
        else:
            self._draw_team_select(surface)

    def _draw_mode_select(self, surface: pygame.Surface):
        # Title
        title = _fancy(28, bold=True).render('Select Game Mode', True, (210, 215, 240))
        surface.blit(title, ((SCREEN_W - title.get_width()) // 2, 18))

        sub = _font(13).render('Pick the size of the pitch and number of players.', True, (105, 108, 130))
        surface.blit(sub, ((SCREEN_W - sub.get_width()) // 2, 56))

        # --- Pitch image at native 1:1 pixel ratio ---
        mode = _GAME_MODES[self._mode_dd.selected]
        dy = _MODE_DD_Y + _MODE_DD_H + 24
        pitch_surf = _load_pitch_native(mode['pitch_img'])

        if pitch_surf:
            tw, th = pitch_surf.get_size()
            px = (SCREEN_W - tw) // 2
            pygame.draw.rect(surface, (55, 60, 80),
                             pygame.Rect(px - 2, dy - 2, tw + 4, th + 4),
                             border_radius=4)
            surface.blit(pitch_surf, (px, dy))

        # Bottom buttons
        self._mode_back_btn.draw(surface)
        self._mode_select_btn.draw(surface)

        # Dropdown drawn last so it renders on top of the pitch preview
        self._mode_dd.draw(surface)

    def _draw_team_select(self, surface: pygame.Surface):
        # Title bar
        mode = _GAME_MODES[self._selected_mode]
        title = _fancy(20, bold=True).render('Select Teams', True, (210, 215, 240))
        surface.blit(title, (PADDING, 16))
        mode_badge = _font(12).render(f'  {mode["label"]}  ', True, (140, 200, 140))
        bx = PADDING + title.get_width() + 14
        badge_rect = pygame.Rect(bx - 4, 18, mode_badge.get_width() + 8, 22)
        draw_bordered_rect(surface, (28, 50, 28), (60, 100, 60), badge_rect, radius=4, border_w=1)
        surface.blit(mode_badge, (bx, 21))
        # Title separator line
        pygame.draw.line(surface, (38, 42, 62), (0, TITLE_H - 1), (SCREEN_W, TITLE_H - 1))

        # Draw card first so tab bar renders on top
        self._draw_team_card(surface, is_home=(self._active_tab == 0))

        # Tab bar drawn after card so active tab can cover card's top border
        self._draw_tab_bar(surface)

        # Bottom bar
        bar_y = SCREEN_H - BOTTOM_BAR_H
        pygame.draw.line(surface, (40, 42, 58), (0, bar_y), (SCREEN_W, bar_y))
        if self._start_btn:
            self._start_btn.draw(surface)
        if self._back_btn:
            self._back_btn.draw(surface)

        # Draw active tab's dropdowns last so they appear on top of roster content
        for dd in self._active_dropdowns():
            if dd:
                dd.draw(surface)

    def _draw_tab_bar(self, surface: pygame.Surface):
        labels = ['Home', 'Away']
        accents = [COLOR_HOME_ACCENT, COLOR_AWAY_ACCENT]
        bgs_active = [COLOR_HOME_CARD_BG, COLOR_AWAY_CARD_BG]
        for i, (rect, label) in enumerate(zip(_TAB_RECTS, labels)):
            active = (i == self._active_tab)
            bg = bgs_active[i] if active else (28, 30, 48)
            border = accents[i] if active else (55, 58, 80)
            if active:
                # Background: rounded top corners, flat bottom — extends 2px down to cover card's top border
                bg_rect = pygame.Rect(rect.x, rect.y, rect.width, rect.height + 2)
                pygame.draw.rect(surface, bg, bg_rect,
                                 border_top_left_radius=6, border_top_right_radius=6,
                                 border_bottom_left_radius=0, border_bottom_right_radius=0)
                # Three-sided border: top, left, right — no bottom
                lw = 2
                pygame.draw.line(surface, border, (rect.x, rect.y), (rect.right, rect.y), lw)
                pygame.draw.line(surface, border, (rect.x, rect.y), (rect.x, rect.bottom), lw)
                pygame.draw.line(surface, border, (rect.right - 1, rect.y), (rect.right - 1, rect.bottom), lw)
            else:
                draw_bordered_rect(surface, bg, border, rect, radius=6, border_w=1)
            col = accents[i] if active else (140, 140, 160)
            lbl_surf = _font(13, bold=active).render(label, True, col)
            surface.blit(lbl_surf, (rect.x + (rect.width - lbl_surf.get_width()) // 2,
                                     rect.y + (rect.height - lbl_surf.get_height()) // 2))

    def _draw_team_card(self, surface: pygame.Surface, is_home: bool):
        card = pygame.Rect(PADDING, CARD_Y, CARD_W, CARD_H)
        card_bg = COLOR_HOME_CARD_BG if is_home else COLOR_AWAY_CARD_BG
        accent = COLOR_HOME_ACCENT if is_home else COLOR_AWAY_ACCENT
        text_col = COLOR_HOME_TEXT if is_home else COLOR_AWAY_TEXT
        ly = _card_layout()
        cx = card.x + INNER_PAD

        # Card background — top-left is square (tab sits flush there), other corners rounded
        bw = 2
        pygame.draw.rect(surface, accent, card,
                         border_top_left_radius=0, border_top_right_radius=8,
                         border_bottom_left_radius=8, border_bottom_right_radius=8)
        inner = card.inflate(-2 * bw, -2 * bw)
        pygame.draw.rect(surface, card_bg, inner,
                         border_top_left_radius=0, border_top_right_radius=6,
                         border_bottom_left_radius=6, border_bottom_right_radius=6)

        # --- Team header (logo + name + race) ---
        teams = self._home_filtered_teams if is_home else self._away_filtered_teams
        dd = self._home_team_dd if is_home else self._away_team_dd
        team = teams[dd.selected % len(teams)] if teams and dd else None

        cy = ly['cy']
        logo_surf = load_team_logo(team.race) if team else None

        if logo_surf:
            surface.blit(logo_surf, (cx, cy))
        else:
            pygame.draw.circle(surface, accent,
                               (cx + LOGO_SIZE // 2, cy + LOGO_SIZE // 2), LOGO_SIZE // 2)
        tx = cx + LOGO_SIZE + 12

        if team:
            name_surf = _fancy(17, bold=True).render(team.name, True, text_col)
            surface.blit(name_surf, (tx, cy + 6))
            race_surf = _font(11).render(team.race, True, (130, 130, 150))
            surface.blit(race_surf, (tx, cy + 30))

            # TV / Rerolls / Apothecary chips — right-aligned in the header block
            players = team.players if hasattr(team, 'players') else []
            player_tv = sum(getattr(p.role, 'cost', 0) for p in players if p.role)
            reroll_cost = self._reroll_costs.get(team.race, 0)
            rerolls = getattr(team, 'rerolls', 0)
            apothecaries = getattr(team, 'apothecaries', 0)
            tv = player_tv + rerolls * reroll_cost + apothecaries * 50000
            fan_factor   = getattr(team, 'fan_factor', 0)
            ass_coaches  = getattr(team, 'ass_coaches', 0)
            cheerleaders = getattr(team, 'cheerleaders', 0)
            chip_val_col = (185, 185, 205)
            chip_border_col = (65, 65, 78)
            chip_bg_col = tuple(max(0, c - 18) for c in card_bg)
            chips = [
                ('TV',           f'{tv // 1000}k',               chip_val_col, chip_bg_col),
                ('Rerolls',      str(rerolls),                    chip_val_col, chip_bg_col),
                ('Apothecary',   'Yes' if apothecaries else 'No', chip_val_col, chip_bg_col),
                ('Ass. Coaches', str(ass_coaches),                chip_val_col, chip_bg_col),
                ('Cheerleaders', str(cheerleaders),               chip_val_col, chip_bg_col),
                ('Fan Factor',   str(fan_factor),                 chip_val_col, chip_bg_col),
            ]
            font_chip_lbl = _font(10)
            font_chip_val = _font(13, bold=True)
            chip_gap = 10
            # Measure total chip width to right-align
            chip_sizes = []
            for lbl, val, val_col, bg_col in chips:
                lbl_surf = font_chip_lbl.render(lbl, True, (140, 140, 160))
                val_surf = font_chip_val.render(val, True, val_col)
                chip_w = max(lbl_surf.get_width(), val_surf.get_width()) + 16
                chip_h = lbl_surf.get_height() + val_surf.get_height() + 6
                chip_sizes.append((chip_w, chip_h, lbl_surf, val_surf, val_col, bg_col))
            total_chips_w = sum(w for w, *_ in chip_sizes) + chip_gap * (len(chip_sizes) - 1)
            chip_x = card.right - INNER_PAD - total_chips_w
            chip_y = cy + (ly['logo_block_h'] - chip_sizes[0][1]) // 2  # vertically centered in header
            for chip_w, chip_h, lbl_surf, val_surf, val_col, bg_col in chip_sizes:
                chip_rect = pygame.Rect(chip_x, chip_y, chip_w, chip_h)
                draw_bordered_rect(surface, bg_col, chip_border_col, chip_rect, radius=4, border_w=1)
                surface.blit(lbl_surf, (chip_x + (chip_w - lbl_surf.get_width()) // 2, chip_y + 2))
                surface.blit(val_surf, (chip_x + (chip_w - val_surf.get_width()) // 2,
                                        chip_y + lbl_surf.get_height() + 4))
                chip_x += chip_w + chip_gap

        # Three-column labels: Race | Team | Agent
        lbl_col = (140, 145, 165)
        x_race  = cx
        x_team  = cx + DD_RACE_W + DD_GAP
        x_agent = x_team + DD_TEAM_W + DD_GAP
        surface.blit(_font(11).render('Race',  True, lbl_col), (x_race,  ly['y_dd_lbl']))
        surface.blit(_font(11).render('Team',  True, lbl_col), (x_team,  ly['y_dd_lbl']))
        surface.blit(_font(11).render('Agent', True, lbl_col), (x_agent, ly['y_dd_lbl']))

        # Separator
        pygame.draw.line(surface, (48, 52, 72),
                         (cx, ly['y_separator']),
                         (card.right - INNER_PAD, ly['y_separator']))

        # Roster
        if team:
            draw_roster_table(surface, team, cx, ly['y_roster'],
                              card.right - INNER_PAD, card.bottom - 4, is_home)


"""
Game creation screen — mode selection, team/agent picker, roster preview.
"""
from __future__ import annotations
import os
import pygame
from typing import Optional

import botbowl
from botbowl.core.load import load_all_teams, load_rule_set
from botbowl.core.model import Agent, Action
from botbowl.core.table import ActionType
from botbowl.ai.registry import list_bots
from botbowl.gui.rendering.ui_primitives import (
    Button, COLOR_BTN_DEFAULT, COLOR_BTN_HOME, COLOR_BTN_AWAY,
    COLOR_BTN_NEUTRAL, COLOR_TEXT, COLOR_TEXT_DIM, COLOR_PANEL_BG, COLOR_BORDER
)
from botbowl.gui import sprites as spr

CREATE_W = 1080
TEAM_SELECT_H = 700
MODE_SELECT_H = 800           # fixed — never changes on mode click
CREATE_H = TEAM_SELECT_H  # used for CARD_H / team-select layout

PADDING = 24
CARD_GAP = 16
INNER_PAD = 14
BOTTOM_BAR_H = 64
TITLE_H = 62

CARD_W = (CREATE_W - 2 * PADDING - CARD_GAP) // 2   # 508
CARD_Y = TITLE_H
CARD_H = CREATE_H - TITLE_H - BOTTOM_BAR_H            # 574

DD_HALF_GAP = 8
DD_HALF_W = (CARD_W - 2 * INNER_PAD - DD_HALF_GAP) // 2  # 236
DD_W = CARD_W - 2 * INNER_PAD                         # 480 (full-width, kept for reference)
DD_H = 30
LOGO_SIZE = 54


def _card_layout():
    """Returns a dict of shared y-positions for card content (screen-absolute)."""
    cy = CARD_Y + INNER_PAD          # 76
    logo_block_h = LOGO_SIZE + 14    # 68
    lbl_gap = 3

    # Team and Agent dropdowns sit on the SAME row (side-by-side)
    y_dd_lbl    = cy + logo_block_h              # 144
    y_dd_row    = y_dd_lbl + 16 + lbl_gap        # 163  (both dropdowns at this y)
    y_resources = y_dd_row + DD_H + 8            # 201
    y_separator = y_resources + 18 + 4           # 223
    y_roster    = y_separator + 8                # 231
    return dict(cy=cy, logo_block_h=logo_block_h,
                y_dd_lbl=y_dd_lbl, y_dd_row=y_dd_row,
                y_resources=y_resources, y_separator=y_separator,
                y_roster=y_roster)

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
COLOR_HOME_CARD_BG = (24, 30, 56)
COLOR_AWAY_CARD_BG = (48, 22, 22)
COLOR_HOME_TEXT = (140, 175, 255)
COLOR_AWAY_TEXT = (255, 145, 115)

_LOGO_CACHE: dict = {}
_IMG_BASE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'img', 'teamlogos'))


def _font(size: int = 13, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont('Arial', size, bold=bold)


def _load_logo(race: str, size: int = LOGO_SIZE) -> Optional[pygame.Surface]:
    key = (race.lower(), size)
    if key in _LOGO_CACHE:
        return _LOGO_CACHE[key]
    path = os.path.join(_IMG_BASE, race.lower() + '.png')
    if not os.path.exists(path):
        _LOGO_CACHE[key] = None
        return None
    try:
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.smoothscale(img, (size, size))
        _LOGO_CACHE[key] = img
        return img
    except Exception:
        _LOGO_CACHE[key] = None
        return None


class Dropdown:
    """Simple dropdown widget."""

    def __init__(self, rect: pygame.Rect, options: list[str],
                 selected: int = 0, font_size: int = 13):
        self.rect = rect
        self.options = options
        self.selected = selected
        self.font_size = font_size
        self.open = False

    def draw(self, surface: pygame.Surface):
        # Header
        pygame.draw.rect(surface, (52, 56, 75), self.rect, border_radius=4)
        pygame.draw.rect(surface, (90, 95, 125), self.rect, 1, border_radius=4)
        font = _font(self.font_size)
        label = self.options[self.selected] if self.options else ''
        surf = font.render(label[:50], True, COLOR_TEXT)
        surface.blit(surf, (self.rect.x + 8,
                             self.rect.y + (self.rect.height - surf.get_height()) // 2))
        arrow = font.render('▾', True, (160, 160, 160))
        surface.blit(arrow, (self.rect.right - 20,
                              self.rect.y + (self.rect.height - arrow.get_height()) // 2))

        if self.open:
            item_h = self.rect.height
            max_visible = 10
            for i, opt in enumerate(self.options[:max_visible]):
                item_rect = pygame.Rect(self.rect.x, self.rect.bottom + i * item_h,
                                        self.rect.width, item_h)
                bg = (75, 85, 125) if i == self.selected else (48, 52, 72)
                pygame.draw.rect(surface, bg, item_rect)
                pygame.draw.rect(surface, (90, 95, 125), item_rect, 1)
                item_surf = font.render(opt[:50], True, COLOR_TEXT)
                surface.blit(item_surf, (item_rect.x + 8,
                                          item_rect.y + (item_h - item_surf.get_height()) // 2))

    def handle_click(self, pos: tuple) -> bool:
        """Returns True if selection changed."""
        if self.rect.collidepoint(pos):
            self.open = not self.open
            return False
        if self.open:
            item_h = self.rect.height
            for i in range(min(len(self.options), 10)):
                item_rect = pygame.Rect(self.rect.x, self.rect.bottom + i * item_h,
                                        self.rect.width, item_h)
                if item_rect.collidepoint(pos):
                    changed = (i != self.selected)
                    self.selected = i
                    self.open = False
                    return changed
            self.open = False
        return False

    def close(self):
        self.open = False

    @property
    def value(self) -> Optional[str]:
        return self.options[self.selected] if self.options else None


# Pre-compute mode button layout
_BTN_W = 155
_BTN_H = 96
_BTN_GAP = 18
_MODE_TOTAL_W = len(_GAME_MODES) * _BTN_W + (len(_GAME_MODES) - 1) * _BTN_GAP
_MODE_START_X = (CREATE_W - _MODE_TOTAL_W) // 2
_MODE_BTN_Y = 165   # balanced below title/subtitle


class CreateGameScreen:
    """Two-step game creation: mode selection then team/agent selection."""

    def __init__(self, app, step: int = 0):
        self.app = app
        self.width = CREATE_W
        self._step = 0           # 0 = mode, 1 = team/agent
        self._selected_mode = 4  # default 11v11 — must be set before _resize_display
        self.height = TEAM_SELECT_H
        self._resize_display()
        self._mode_rects: list[pygame.Rect] = []
        self._build_mode_rects()

        # Step 0 mode-select bottom buttons (positions updated in _resize_display)
        self._mode_back_btn = Button(pygame.Rect(0, 0, 100, 36),
                                     label='← Back', color=COLOR_BTN_DEFAULT, font_size=13)
        self._mode_select_btn = Button(pygame.Rect(0, 0, 150, 36),
                                       label='Select  →', color=COLOR_BTN_NEUTRAL, font_size=13)
        self._resize_display()  # positions mode buttons correctly now height is known

        # Step 1 state
        self._teams: list = []
        self._team_names: list[str] = []
        self._reroll_costs: dict = {}
        self._bot_names: list[str] = ['Human'] + list_bots()
        self._home_team_dd: Optional[Dropdown] = None
        self._away_team_dd: Optional[Dropdown] = None
        self._home_agent_dd: Optional[Dropdown] = None
        self._away_agent_dd: Optional[Dropdown] = None
        self._start_btn: Optional[Button] = None
        self._back_btn: Optional[Button] = None

        if step == 1:
            self._step = 1
            self._load_teams_for_mode()

    def _mode_select_h(self) -> int:
        return MODE_SELECT_H

    def _resize_display(self):
        h = self._mode_select_h() if self._step == 0 else TEAM_SELECT_H
        self.height = h
        current = pygame.display.get_surface()
        if current is None or current.get_size() != (self.width, h):
            pygame.display.set_mode((self.width, h))
        # Update mode-select button positions to match (possibly new) window height
        if hasattr(self, '_mode_back_btn'):
            btn_y = h - 44
            self._mode_back_btn.rect = pygame.Rect(PADDING, btn_y, 100, 36)
            self._mode_select_btn.rect = pygame.Rect(self.width - PADDING - 150, btn_y, 150, 36)

    def _build_mode_rects(self):
        self._mode_rects = []
        for i in range(len(_GAME_MODES)):
            x = _MODE_START_X + i * (_BTN_W + _BTN_GAP)
            self._mode_rects.append(pygame.Rect(x, _MODE_BTN_Y, _BTN_W, _BTN_H))

    def _card_rect(self, is_home: bool) -> pygame.Rect:
        x = PADDING if is_home else PADDING + CARD_W + CARD_GAP
        return pygame.Rect(x, CARD_Y, CARD_W, CARD_H)

    def _load_teams_for_mode(self):
        mode = _GAME_MODES[self._selected_mode]
        config = botbowl.load_config(mode['config'])
        ruleset = botbowl.load_rule_set(config.ruleset)
        self._teams = load_all_teams(ruleset, board_size=mode['board_size'])
        self._team_names = [t.name for t in self._teams]
        # Build race → reroll_cost map for TV calculation
        self._reroll_costs = {r.name: r.reroll_cost for r in ruleset.races}

        x_home = PADDING + INNER_PAD
        x_away = PADDING + CARD_W + CARD_GAP + INNER_PAD
        ly = _card_layout()

        x_home_agent = x_home + DD_HALF_W + DD_HALF_GAP
        x_away_agent = x_away + DD_HALF_W + DD_HALF_GAP
        row_y = ly['y_dd_row']

        self._home_team_dd = Dropdown(
            pygame.Rect(x_home, row_y, DD_HALF_W, DD_H), self._team_names)
        self._away_team_dd = Dropdown(
            pygame.Rect(x_away, row_y, DD_HALF_W, DD_H), self._team_names,
            selected=min(1, len(self._team_names) - 1))
        self._home_agent_dd = Dropdown(
            pygame.Rect(x_home_agent, row_y, DD_HALF_W, DD_H), self._bot_names)
        self._away_agent_dd = Dropdown(
            pygame.Rect(x_away_agent, row_y, DD_HALF_W, DD_H), self._bot_names)

        btn_y = TEAM_SELECT_H - BOTTOM_BAR_H + 12
        self._start_btn = Button(
            pygame.Rect(self.width - PADDING - 180, btn_y, 180, 40),
            label='Start Game  →', color=COLOR_BTN_NEUTRAL, font_size=14)
        self._back_btn = Button(
            pygame.Rect(PADDING, btn_y, 110, 40),
            label='← Back', color=COLOR_BTN_DEFAULT, font_size=13)

    def _all_dropdowns(self):
        return [dd for dd in [self._home_team_dd, self._away_team_dd,
                               self._home_agent_dd, self._away_agent_dd] if dd]

    def update(self):
        pass

    def handle_event(self, event: pygame.event.Event):
        mouse = pygame.mouse.get_pos()

        if self._step == 0:
            self._mode_back_btn.update_hover(mouse)
            self._mode_select_btn.update_hover(mouse)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._mode_back_btn.is_clicked(event.pos):
                    self.app.pop_screen()
                    return
                if self._mode_select_btn.is_clicked(event.pos):
                    self._step = 1
                    self._load_teams_for_mode()
                    return
                for i, rect in enumerate(self._mode_rects):
                    if rect.collidepoint(event.pos):
                        self._selected_mode = i
                        return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.app.pop_screen()

        elif self._step == 1:
            if self._start_btn:
                self._start_btn.update_hover(mouse)
            if self._back_btn:
                self._back_btn.update_hover(mouse)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._back_btn and self._back_btn.is_clicked(event.pos):
                    self._step = 0
                    for dd in self._all_dropdowns():
                        dd.close()
                    return
                if self._start_btn and self._start_btn.is_clicked(event.pos):
                    self._start_game()
                    return
                # Handle dropdowns: close others when one opens
                for dd in self._all_dropdowns():
                    changed = dd.handle_click(event.pos)
                    if dd.open:
                        for other in self._all_dropdowns():
                            if other is not dd:
                                other.close()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                # Close open dropdowns first, then go back
                any_open = any(dd.open for dd in self._all_dropdowns())
                if any_open:
                    for dd in self._all_dropdowns():
                        dd.close()
                else:
                    self._step = 0

    def _start_game(self):
        mode = _GAME_MODES[self._selected_mode]
        config_name = mode['config']
        board_size = mode['board_size']

        try:
            config = botbowl.load_config(config_name)
            config.competition_mode = False
            ruleset = botbowl.load_rule_set(config.ruleset)
            arena = botbowl.load_arena(config.arena)

            home_name = self._home_team_dd.value
            away_name = self._away_team_dd.value
            home_team = next(t for t in self._teams if t.name == home_name)
            away_team = next(t for t in self._teams if t.name == away_name)

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
        self._resize_display()
        surface = pygame.display.get_surface()
        surface.fill((10, 10, 16))

        if self._step == 0:
            self._draw_mode_select(surface)
        else:
            self._draw_team_select(surface)

    def _draw_mode_select(self, surface: pygame.Surface):
        # Title
        title = _font(28, bold=True).render('Select Game Mode', True, (210, 215, 240))
        surface.blit(title, ((self.width - title.get_width()) // 2, 54))

        sub = _font(13).render('Pick the size of the pitch and number of players.', True, (105, 108, 130))
        surface.blit(sub, ((self.width - sub.get_width()) // 2, 96))

        for i, (mode, rect) in enumerate(zip(_GAME_MODES, self._mode_rects)):
            selected = (i == self._selected_mode)

            # Card background — much brighter to stand out from page bg
            bg_col = (52, 72, 110) if selected else (38, 40, 60)
            border_col = (90, 180, 90) if selected else (75, 78, 105)
            border_w = 2 if selected else 1

            pygame.draw.rect(surface, bg_col, rect, border_radius=10)
            pygame.draw.rect(surface, border_col, rect, border_w, border_radius=10)

            # Selected: top green accent bar
            if selected:
                acc = pygame.Rect(rect.x + 10, rect.y + 2, rect.width - 20, 3)
                pygame.draw.rect(surface, (90, 200, 90), acc, border_radius=2)

            # Label
            lbl_col = (165, 235, 165) if selected else (210, 212, 232)
            label = _font(24, bold=True).render(mode['label'], True, lbl_col)
            lx = rect.x + (rect.width - label.get_width()) // 2
            surface.blit(label, (lx, rect.y + 20))

            # Description
            desc_col = (125, 190, 125) if selected else (118, 118, 148)
            desc = _font(11).render(mode['desc'], True, desc_col)
            dx_lbl = rect.x + (rect.width - desc.get_width()) // 2
            surface.blit(desc, (dx_lbl, rect.y + 60))

        # --- Pitch image at native 1:1 pixel ratio ---
        mode = _GAME_MODES[self._selected_mode]
        dy = _MODE_BTN_Y + _BTN_H + 24
        pitch_surf = _load_pitch_native(mode['pitch_img'])

        if pitch_surf:
            tw, th = pitch_surf.get_size()
            px = (self.width - tw) // 2
            pygame.draw.rect(surface, (55, 60, 80),
                             pygame.Rect(px - 2, dy - 2, tw + 4, th + 4),
                             border_radius=4)
            surface.blit(pitch_surf, (px, dy))

        # Bottom buttons
        self._mode_back_btn.draw(surface)
        self._mode_select_btn.draw(surface)

    def _draw_team_select(self, surface: pygame.Surface):
        # Title bar
        mode = _GAME_MODES[self._selected_mode]
        title = _font(20, bold=True).render('Select Teams & Agents', True, (210, 215, 240))
        surface.blit(title, (PADDING, 16))
        mode_badge = _font(12).render(f'  {mode["label"]}  ', True, (140, 200, 140))
        bx = PADDING + title.get_width() + 14
        badge_rect = pygame.Rect(bx - 4, 18, mode_badge.get_width() + 8, 22)
        pygame.draw.rect(surface, (28, 50, 28), badge_rect, border_radius=4)
        pygame.draw.rect(surface, (60, 100, 60), badge_rect, 1, border_radius=4)
        surface.blit(mode_badge, (bx, 21))
        # Title separator line
        pygame.draw.line(surface, (38, 42, 62), (0, TITLE_H - 1), (self.width, TITLE_H - 1))

        # Draw cards
        self._draw_team_card(surface, is_home=True)
        self._draw_team_card(surface, is_home=False)

        # Bottom bar
        bar_y = self.height - BOTTOM_BAR_H
        pygame.draw.line(surface, (40, 42, 58), (0, bar_y), (self.width, bar_y))
        if self._start_btn:
            self._start_btn.draw(surface)
        if self._back_btn:
            self._back_btn.draw(surface)

        # Draw dropdowns last so they appear on top of roster content
        for dd in self._all_dropdowns():
            if dd:
                dd.draw(surface)

    def _draw_team_card(self, surface: pygame.Surface, is_home: bool):
        card = self._card_rect(is_home)
        card_bg = COLOR_HOME_CARD_BG if is_home else COLOR_AWAY_CARD_BG
        accent = COLOR_HOME_ACCENT if is_home else COLOR_AWAY_ACCENT
        text_col = COLOR_HOME_TEXT if is_home else COLOR_AWAY_TEXT
        ly = _card_layout()
        cx = card.x + INNER_PAD

        # Card background
        pygame.draw.rect(surface, card_bg, card, border_radius=8)
        pygame.draw.rect(surface, accent, card, 2, border_radius=8)

        # Colored top accent strip (inside the rounded rect, clipped to look smooth)
        strip = pygame.Rect(card.x + 8, card.y + 1, card.width - 16, 3)
        pygame.draw.rect(surface, accent, strip, border_radius=2)

        # --- Team header (logo + name + race) ---
        teams = self._teams
        dd = self._home_team_dd if is_home else self._away_team_dd
        team = teams[dd.selected % len(teams)] if teams and dd else None

        cy = ly['cy']
        logo_surf = _load_logo(team.race) if team else None

        if logo_surf:
            surface.blit(logo_surf, (cx, cy))
        else:
            pygame.draw.circle(surface, accent,
                               (cx + LOGO_SIZE // 2, cy + LOGO_SIZE // 2), LOGO_SIZE // 2)
        tx = cx + LOGO_SIZE + 12

        if team:
            name_surf = _font(17, bold=True).render(team.name, True, text_col)
            surface.blit(name_surf, (tx, cy + 6))
            race_surf = _font(11).render(team.race, True, (130, 130, 150))
            surface.blit(race_surf, (tx, cy + 30))

        # Side-by-side labels: Team (left half) | Agent (right half)
        lbl_col = (140, 145, 165)
        surface.blit(_font(11).render('Team',  True, lbl_col), (cx, ly['y_dd_lbl']))
        surface.blit(_font(11).render('Agent', True, lbl_col),
                     (cx + DD_HALF_W + DD_HALF_GAP, ly['y_dd_lbl']))

        # Resources: TV · Rerolls · Apothecary
        if team:
            players = team.players if hasattr(team, 'players') else []
            player_tv = sum(getattr(p.role, 'cost', 0) for p in players if p.role)
            reroll_cost = self._reroll_costs.get(team.race, 0)
            rerolls = getattr(team, 'rerolls', 0)
            apothecaries = getattr(team, 'apothecaries', 0)
            tv = player_tv + rerolls * reroll_cost + apothecaries * 50000
            res_parts = [f'TV: {tv // 1000}k']
            if rerolls:
                res_parts.append(f'Rerolls: {rerolls} ({reroll_cost // 1000}k ea)')
            if apothecaries:
                res_parts.append(f'Apothecary ×{apothecaries}')
            res_surf = _font(11).render('  ·  '.join(res_parts), True, (110, 120, 145))
            surface.blit(res_surf, (cx, ly['y_resources']))

        # Separator
        pygame.draw.line(surface, (48, 52, 72),
                         (cx, ly['y_separator']),
                         (card.right - INNER_PAD, ly['y_separator']))

        # Roster
        if team:
            self._draw_roster(surface, team, cx, ly['y_roster'],
                              card.right - INNER_PAD, card.bottom - 4, is_home)

    def _draw_roster(self, surface: pygame.Surface, team, x: int, y: int,
                     x_max: int, y_max: int, is_home: bool):
        font_hdr = _font(11, bold=True)
        font_row = _font(12)
        hdr_col = (100, 105, 130)
        row_col = (195, 195, 210)
        skill_col = (90, 170, 110)
        tv_col = (170, 155, 90)
        alt_bg = (38, 48, 85) if is_home else (72, 32, 32)
        ROW_H = 22

        total_w = x_max - x  # ~480px
        ICON_W = ROW_H  # square icon = row height

        # Column layout (right-to-left): TV | Skills | AV | AG | ST | MA | Name | Icon
        COL_TV_R     = total_w - 2
        COL_SKILL_R  = COL_TV_R - 32
        COL_SKILL_L  = COL_SKILL_R - 150    # 150px for skills
        COL_AV_R     = COL_SKILL_L - 8
        COL_AG_R     = COL_AV_R - 24
        COL_ST_R     = COL_AG_R - 24
        COL_MA_R     = COL_ST_R - 24
        COL_ICON     = 0
        COL_NR_R     = ICON_W + 20           # right edge of # column
        COL_NAME     = COL_NR_R + 4
        name_max_w   = COL_MA_R - COL_NAME - 8

        text_oy = (ROW_H - font_row.get_height()) // 2  # vertical offset to center text in row

        def blit_right(surf, col_right_rel, row_y):
            surface.blit(surf, (x + col_right_rel - surf.get_width(), row_y + text_oy))

        def truncate(text, font, max_w):
            if font.size(text)[0] <= max_w:
                return text
            while text and font.size(text + '…')[0] > max_w:
                text = text[:-1]
            return text + '…'

        def skill_lines(skills):
            """Split skill list into lines fitting within 150px."""
            names = [s.name.replace('_', ' ').title() for s in skills]
            lines, current = [], ''
            for sk in names:
                test = current + (', ' if current else '') + sk
                if font_row.size(test)[0] <= 150:
                    current = test
                else:
                    if current:
                        lines.append(current)
                    current = sk
            if current:
                lines.append(current)
            return lines

        # Header row
        blit_right(font_hdr.render('#',    True, hdr_col), COL_NR_R,    y)
        surface.blit(font_hdr.render('Name',   True, hdr_col), (x + COL_NAME,    y))
        blit_right(font_hdr.render('MA',       True, hdr_col), COL_MA_R,    y)
        blit_right(font_hdr.render('ST',       True, hdr_col), COL_ST_R,    y)
        blit_right(font_hdr.render('AG',       True, hdr_col), COL_AG_R,    y)
        blit_right(font_hdr.render('AV',       True, hdr_col), COL_AV_R,    y)
        surface.blit(font_hdr.render('Skills', True, hdr_col), (x + COL_SKILL_L, y))
        blit_right(font_hdr.render('TV',       True, hdr_col), COL_TV_R,    y)
        y += ROW_H + 4

        race = team.race if hasattr(team, 'race') else None

        roster = team.players if hasattr(team, 'players') else []
        for idx, player in enumerate(roster):
            role = player.role
            s_lines = skill_lines(role.skills) if role and role.skills else []
            row_lines = max(1, len(s_lines))
            row_h_total = row_lines * ROW_H

            if y + row_h_total > y_max:
                break

            row_rect = pygame.Rect(x - 2, y - 1, total_w + 2, row_h_total)
            if idx % 2 == 1:
                pygame.draw.rect(surface, alt_bg, row_rect, border_radius=2)

            # Player icon
            icon = spr.get_player_surface(player, is_home, False) if hasattr(player, 'team') else None
            if icon is None and race and role:
                # build a minimal proxy to reuse the sprite function
                class _P:
                    pass
                _p = _P(); _p.role = role
                _t = _P(); _t.race = race
                _p.team = _t
                icon = spr.get_player_surface(_p, is_home, False)
            if icon:
                scaled = pygame.transform.smoothscale(icon, (ICON_W, ICON_W))
                surface.blit(scaled, (x + COL_ICON, y))

            # Nr
            blit_right(font_row.render(str(player.nr), True, row_col), COL_NR_R, y)

            # Name
            name = truncate(player.name or '', font_row, name_max_w)
            surface.blit(font_row.render(name, True, row_col), (x + COL_NAME, y + text_oy))

            if role:
                for val, col_r in ((role.ma, COL_MA_R), (role.st, COL_ST_R),
                                   (role.ag, COL_AG_R), (role.av, COL_AV_R)):
                    blit_right(font_row.render(str(val), True, row_col), col_r, y)
                tv = getattr(role, 'cost', 0) // 1000
                blit_right(font_row.render(f'{tv}k', True, tv_col), COL_TV_R, y)

            # Skill lines (vertically centered within each sub-row)
            for i, line in enumerate(s_lines):
                sy = y + i * ROW_H
                surface.blit(font_row.render(line, True, skill_col),
                             (x + COL_SKILL_L, sy + text_oy))

            y += row_h_total

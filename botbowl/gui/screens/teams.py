"""
Teams screen — browse existing teams and create new ones.
"""
from __future__ import annotations
import json
import os
import re
import pygame
from typing import Optional

import botbowl
from botbowl.gui.gui import SCREEN_W, SCREEN_H
from botbowl.core.load import load_all_teams, load_rule_set
from botbowl.core.util import get_data_path
from botbowl.gui.rendering.ui_primitives import (
    Button, TextInput, draw_bordered_rect, get_button_image,
    COLOR_BTN_DEFAULT, COLOR_BTN_NEUTRAL,
    COLOR_TEXT, COLOR_TEXT_DIM, COLOR_BORDER,
)
from botbowl.gui.screens.create_game import (
    Dropdown, PADDING, INNER_PAD, BOTTOM_BAR_H, TITLE_H,
)
from botbowl.gui.screens.team_utils import (
    load_team_logo, calc_team_tv, draw_roster_table,
    TEAM_BUDGET, APOTHECARY_COST, STAFF_COSTS, MIN_PLAYERS, MAX_PLAYERS,
)

_BOARD_SIZES  = [1, 3, 5, 7, 11]
_SIZE_CONFIGS = {1: 'gym-1', 3: 'gym-3', 5: 'gym-5', 7: 'gym-7', 11: 'bot-bowl'}

ROW_H = 44


from botbowl.gui.fonts import get_font as _fancy, get_body_font as _font


def _load_ruleset_for_size(board_size: int):
    config  = botbowl.load_config(_SIZE_CONFIGS[board_size])
    ruleset = load_rule_set(config.ruleset)
    reroll_costs = {r.name: r.reroll_cost for r in ruleset.races}
    return ruleset, reroll_costs


# ─────────────────────────────────────────────────────────────────────────────
# TeamsScreen
# ─────────────────────────────────────────────────────────────────────────────

class TeamsScreen:
    """Browse existing teams by board size, and navigate to the creator."""

    _HEADER_H    = 60
    _DD_Y        = _HEADER_H + 36        # dropdown sits one tab-height below header line
    _DD_H        = 32
    _SECTION_TOP = _DD_Y + _DD_H + 20   # content starts below dropdown

    def __init__(self, app):
        self.app = app
        self._size_idx = 4   # default: 11v11
        self._rows: list = []
        self._reroll_costs: dict = {}
        self._scroll_y = 0
        self._stack_depth = 0   # for detecting when a sub-screen is popped

        dd_w = 160
        self._size_dd = Dropdown(
            pygame.Rect((SCREEN_W - dd_w) // 2, self._DD_Y, dd_w, self._DD_H),
            [f'{s}v{s}' for s in _BOARD_SIZES],
            selected=self._size_idx, font_size=13)
        self._back_btn = Button(
            pygame.Rect(PADDING, SCREEN_H - 54, 120, 42),
            label='Back', font_size=15, bg_image=get_button_image())
        self._new_btn = Button(
            pygame.Rect(SCREEN_W - PADDING - 150, SCREEN_H - 54, 150, 42),
            label='New Team', font_size=15, bg_image=get_button_image())

        self._load_teams()

    def _load_teams(self):
        board_size = _BOARD_SIZES[self._size_idx]
        try:
            ruleset, self._reroll_costs = _load_ruleset_for_size(board_size)
            self._rows = sorted(
                load_all_teams(ruleset, board_size=board_size),
                key=lambda t: t.name.lower())
        except Exception as e:
            print(f'TeamsScreen: could not load teams: {e}')
            self._rows = []
        self._scroll_y = 0

    def update(self):
        depth = len(self.app._screen_stack)
        if self._stack_depth > 0 and depth < self._stack_depth:
            self._load_teams()   # reload after returning from creator/detail
        self._stack_depth = depth

    def handle_event(self, event: pygame.event.Event):
        mouse = pygame.mouse.get_pos()
        self._back_btn.update_hover(mouse)
        self._new_btn.update_hover(mouse)

        if event.type == pygame.MOUSEWHEEL:
            if self._size_dd.handle_scroll(mouse, event.y):
                return
            list_top    = self._SECTION_TOP + 44
            list_bottom = SCREEN_H - 60
            total_h     = len(self._rows) * ROW_H
            max_scroll  = max(0, total_h - (list_bottom - list_top))
            self._scroll_y = max(0, min(max_scroll, self._scroll_y - event.y * 20))

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self._size_dd.open:
                    self._size_dd.close()
                else:
                    self.app.pop_screen()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._back_btn.is_clicked(event.pos):
                self.app.pop_screen()
                return
            if self._new_btn.is_clicked(event.pos):
                self.app.push_screen(
                    TeamCreatorScreen(self.app, board_size=_BOARD_SIZES[self._size_idx]))
                return
            if self._size_dd.handle_click(event.pos):
                self._size_idx = self._size_dd.selected
                self._load_teams()
                return
            if self._size_dd.open:
                self._size_dd.close()
                return

            # Row click → team detail
            list_top    = self._SECTION_TOP + 44
            list_bottom = SCREEN_H - 60
            if list_top <= event.pos[1] < list_bottom and self._rows:
                idx = (event.pos[1] - list_top + self._scroll_y) // ROW_H
                if 0 <= idx < len(self._rows):
                    self.app.push_screen(
                        TeamDetailScreen(self.app, self._rows[idx], self._reroll_costs))

    def draw(self, surface: pygame.Surface):
        surface.fill((15, 15, 20))

        # Title
        title = _fancy(24, bold=True).render('Teams', True, (210, 215, 245))
        surface.blit(title, (PADDING, 16))
        pygame.draw.line(surface, (45, 48, 68),
                         (PADDING, 58), (SCREEN_W - PADDING, 58))

        list_top    = self._SECTION_TOP + 44
        list_bottom = SCREEN_H - 60

        # Column headers
        hdr_y   = list_top - 20
        hdr_col = (100, 105, 130)
        surface.blit(_font(11, bold=True).render('Team', True, hdr_col),
                     (PADDING + 46, hdr_y))
        surface.blit(_font(11, bold=True).render('Race', True, hdr_col),
                     (PADDING + 290, hdr_y))
        tv_hdr = _font(11, bold=True).render('TV', True, hdr_col)
        surface.blit(tv_hdr, (SCREEN_W - PADDING - tv_hdr.get_width(), hdr_y))

        # Clip to list area
        clip = pygame.Rect(0, list_top, SCREEN_W, list_bottom - list_top)
        old_clip = surface.get_clip()
        surface.set_clip(clip)

        y = list_top - self._scroll_y
        for idx, team in enumerate(self._rows):
            row_rect = pygame.Rect(PADDING, y, SCREEN_W - 2 * PADDING, ROW_H - 2)
            bg = (25, 25, 32) if idx % 2 == 0 else (22, 22, 28)
            pygame.draw.rect(surface, bg, row_rect, border_radius=3)

            # Logo
            logo = load_team_logo(team.race, size=32)
            if logo:
                surface.blit(logo, (PADDING + 4, y + (ROW_H - 32) // 2))
            else:
                pygame.draw.circle(surface, (60, 80, 140),
                                   (PADDING + 20, y + ROW_H // 2), 14)

            # Name
            name_s = _font(13).render(team.name, True, COLOR_TEXT)
            surface.blit(name_s, (PADDING + 46, y + (ROW_H - name_s.get_height()) // 2))

            # Race
            race_s = _font(12).render(team.race, True, (125, 130, 155))
            surface.blit(race_s, (PADDING + 290, y + (ROW_H - race_s.get_height()) // 2))

            # TV
            tv   = calc_team_tv(team, self._reroll_costs)
            tv_s = _font(12, bold=True).render(f'{tv // 1000}k', True, (180, 165, 85))
            surface.blit(tv_s, (SCREEN_W - PADDING - tv_s.get_width() - 6,
                                y + (ROW_H - tv_s.get_height()) // 2))
            y += ROW_H

        surface.set_clip(old_clip)

        # Scrollbar
        total_h = len(self._rows) * ROW_H
        list_h  = list_bottom - list_top
        if total_h > list_h:
            sb_x = SCREEN_W - 10
            sb_w = 5
            pygame.draw.rect(surface, (28, 30, 42),
                             pygame.Rect(sb_x, list_top, sb_w, list_h), border_radius=3)
            thumb_h = max(20, int(list_h * list_h / total_h))
            max_scroll = total_h - list_h
            thumb_y = list_top + int((list_h - thumb_h) * self._scroll_y / max_scroll)
            pygame.draw.rect(surface, (75, 80, 110),
                             pygame.Rect(sb_x, thumb_y, sb_w, thumb_h), border_radius=3)

        if not self._rows:
            empty = _font(13).render(
                'No teams found for this board size.', True, COLOR_TEXT_DIM)
            surface.blit(empty, ((SCREEN_W - empty.get_width()) // 2, list_top + 40))

        # Bottom bar
        pygame.draw.line(surface, (45, 48, 68),
                         (0, SCREEN_H - 60), (SCREEN_W, SCREEN_H - 60))
        self._back_btn.draw(surface)
        self._new_btn.draw(surface)

        # Dropdown drawn last so it overlaps the team list when open
        self._size_dd.draw(surface)


# ─────────────────────────────────────────────────────────────────────────────
# TeamDetailScreen
# ─────────────────────────────────────────────────────────────────────────────

_DETAIL_HEADER_H = 90   # space for title + race + logo
_DETAIL_LIST_TOP = _DETAIL_HEADER_H + PADDING


class TeamDetailScreen:
    """Read-only view of a single team's roster and stats."""

    def __init__(self, app, team, reroll_costs: dict):
        self.app           = app
        self._team         = team
        self._reroll_costs = reroll_costs

        self._back_btn = Button(
            pygame.Rect(PADDING, SCREEN_H - 54, 120, 42),
            label='Back', font_size=15, bg_image=get_button_image())

    def update(self):
        pass

    def handle_event(self, event: pygame.event.Event):
        self._back_btn.update_hover(pygame.mouse.get_pos())

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.pop_screen()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._back_btn.is_clicked(event.pos):
                self.app.pop_screen()

    def draw(self, surface: pygame.Surface):
        surface.fill((15, 15, 20))

        team = self._team
        tv   = calc_team_tv(team, self._reroll_costs)

        # Logo
        logo = load_team_logo(team.race, size=54)
        if logo:
            surface.blit(logo, (PADDING, 14))
        else:
            pygame.draw.circle(surface, (60, 80, 140), (PADDING + 27, 41), 26)

        # Team name + race
        name_s = _fancy(20, bold=True).render(team.name, True, (210, 215, 245))
        surface.blit(name_s, (PADDING + 64, 14))
        sub_s = _font(12).render(team.race, True, (100, 105, 130))
        surface.blit(sub_s, (PADDING + 64, 42))

        # Staff chips (TV, Rerolls, Apothecary, Ass. Coaches, Cheerleaders, Fan Factor)
        reroll_cost  = self._reroll_costs.get(team.race, 0)
        rerolls      = getattr(team, 'rerolls', 0)
        apothecaries = getattr(team, 'apothecaries', 0)
        fan_factor   = getattr(team, 'fan_factor', 0)
        ass_coaches  = getattr(team, 'ass_coaches', 0)
        cheerleaders = getattr(team, 'cheerleaders', 0)

        chip_bg     = (22, 22, 30)
        chip_border = (60, 62, 78)
        val_col     = (185, 185, 205)
        chips = [
            ('TV',           f'{tv // 1000}k'),
            ('Rerolls',      str(rerolls)),
            ('Apothecary',   'Yes' if apothecaries else 'No'),
            ('Ass. Coaches', str(ass_coaches)),
            ('Cheerleaders', str(cheerleaders)),
            ('Fan Factor',   str(fan_factor)),
        ]
        font_lbl = _font(10)
        font_val = _font(13, bold=True)
        chip_gap = 8
        chip_sizes = []
        for lbl, val in chips:
            ls = font_lbl.render(lbl, True, (130, 132, 155))
            vs = font_val.render(val, True, val_col)
            cw = max(ls.get_width(), vs.get_width()) + 14
            ch = ls.get_height() + vs.get_height() + 6
            chip_sizes.append((cw, ch, ls, vs))

        total_chips_w = sum(w for w, *_ in chip_sizes) + chip_gap * (len(chip_sizes) - 1)
        chip_x = SCREEN_W - PADDING - total_chips_w
        chip_y = 20
        for cw, ch, ls, vs in chip_sizes:
            rect = pygame.Rect(chip_x, chip_y, cw, ch)
            draw_bordered_rect(surface, chip_bg, chip_border, rect, radius=4, border_w=1)
            surface.blit(ls, (chip_x + (cw - ls.get_width()) // 2, chip_y + 2))
            surface.blit(vs, (chip_x + (cw - vs.get_width()) // 2,
                               chip_y + ls.get_height() + 4))
            chip_x += cw + chip_gap

        pygame.draw.line(surface, (45, 48, 68),
                         (PADDING, _DETAIL_HEADER_H),
                         (SCREEN_W - PADDING, _DETAIL_HEADER_H))

        # Roster table
        draw_roster_table(
            surface, team,
            PADDING, _DETAIL_LIST_TOP,
            SCREEN_W - PADDING, SCREEN_H - 64,
            True,
            show_skills=True,
        )

        # Bottom bar
        pygame.draw.line(surface, (45, 48, 68),
                         (0, SCREEN_H - 60), (SCREEN_W, SCREEN_H - 60))
        self._back_btn.draw(surface)


# ─────────────────────────────────────────────────────────────────────────────
# TeamCreatorScreen helpers
# ─────────────────────────────────────────────────────────────────────────────

class _PlayerDraft:
    """Minimal player-like proxy used while building a new team roster."""

    class _Team:
        pass

    def __init__(self, nr: int, name: str, role):
        self.nr          = nr
        self.name        = name
        self.role        = role
        self.extra_skills = []
        self.extra_ma = self.extra_st = self.extra_ag = self.extra_av = 0
        # Proxy team so sprite system can locate the race
        self.team       = self._Team()
        self.team.race  = role.races[0] if (role and role.races) else ''


class _RosterProxy:
    """Thin wrapper so draw_roster_table gets (race, players)."""

    def __init__(self, race: str, players: list):
        self.race    = race
        self.players = players


# ─────────────────────────────────────────────────────────────────────────────
# TeamCreatorScreen — layout constants
# ─────────────────────────────────────────────────────────────────────────────

_C_HEADER_H    = 60                              # input band below title bar (extra bottom margin)
_CONTENT_Y     = TITLE_H + _C_HEADER_H          # 122
_BOTTOM_BAR_Y  = SCREEN_H - BOTTOM_BAR_H        # 687

# Roster takes full width; buy row + staff live below it
_ROSTER_X      = PADDING                         # 24
_ROSTER_X_MAX  = SCREEN_W - PADDING             # 816
_ROSTER_FULL_W = _ROSTER_X_MAX - _ROSTER_X      # 792

_STAFF_BAR_H   = 58                              # horizontal staff strip height
_BUY_ROW_H     = 34                              # buy-player row height (fixed, not scrolling)
_STAFF_BAR_Y   = _BOTTOM_BAR_Y - _STAFF_BAR_H   # 629
_BUY_ROW_Y     = _STAFF_BAR_Y - _BUY_ROW_H - 4  # 591
_ROSTER_TOP    = _CONTENT_Y                      # 122
_ROSTER_BOTTOM = _BUY_ROW_Y - 4                 # 587
_ROSTER_AREA_H = _ROSTER_BOTTOM - _ROSTER_TOP   # 465

_ROSTER_HDR_H  = 29   # height consumed by the header row (ROW_H + 4)
_BTN_SZ        = 22   # +/- button size in staff bar

# Column positions matching draw_roster_table with show_skills=True, SCREEN_W=840
# x = _ROSTER_X + INNER_PAD = 32;  BTN_COL_W = 24 → eff_x_max = 784, total_w = 752
# avail = 561; pos_w = 140; skill_col_w = 250
_TABLE_X      = _ROSTER_X + INNER_PAD        # 32 — left edge of table content
_COL_NAME_ABS = _TABLE_X + 49               # 81 — absolute x of Name column
_COL_NAME_W   = 179                         # name column pixel width
_COL_POS_ABS  = _TABLE_X + 236             # 268 — absolute x of Position column
_COL_POS_W    = 140                         # position column pixel width

_STAFF_ITEMS = [
    ('rerolls',      'Rerolls'),
    ('apothecaries', 'Apothecary'),
    ('ass_coaches',  'Ass. Coaches'),
    ('cheerleaders', 'Cheerleaders'),
    ('fan_factor',   'Fan Factor'),
]


# ─────────────────────────────────────────────────────────────────────────────
# TeamCreatorScreen
# ─────────────────────────────────────────────────────────────────────────────

class TeamCreatorScreen:
    """Build a new Blood Bowl team from scratch and save it as a JSON file."""

    def __init__(self, app, board_size: int = 11):
        self.app          = app
        self._board_size  = board_size

        # ── Ruleset ──────────────────────────────────────────────────────────
        self._ruleset: Optional[object] = None
        self._reroll_costs: dict = {}
        self._race_obj    = None
        self._races: list[str] = []
        self._load_ruleset()

        # ── Header widgets ────────────────────────────────────────────────────
        dd_y   = TITLE_H + 18   # leave room for labels above
        name_w = 260

        self._name_input = TextInput(
            pygame.Rect(PADDING, dd_y, name_w, 30),
            placeholder='Team name…', max_len=40, font_size=13)

        race_x = PADDING + name_w + 14
        self._race_dd = Dropdown(
            pygame.Rect(race_x, dd_y, 180, 30),
            self._races, selected=0, font_size=13)

        size_labels = [f'{s}v{s}' for s in _BOARD_SIZES]
        size_idx = _BOARD_SIZES.index(board_size) if board_size in _BOARD_SIZES else 4
        self._size_dd = Dropdown(
            pygame.Rect(race_x + 180 + 10, dd_y, 100, 30),
            size_labels, selected=size_idx, font_size=13)

        # ── Roster state ──────────────────────────────────────────────────────
        self._players: list[_PlayerDraft] = []
        self._remove_btns: list[pygame.Rect] = []   # rebuilt each draw
        self._roster_scroll  = 0
        self._roster_content_h = 0         # updated each draw
        self._player_row_screen_ys: list[int] = []  # set in draw

        # ── Inline name editing ───────────────────────────────────────────────
        self._editing_idx: int = -1
        self._editing_input: Optional[TextInput] = None

        # ── Buy-player row widgets (positions updated dynamically in draw) ────
        _BUY_H = _BUY_ROW_H - 8
        self._pos_dd = Dropdown(
            pygame.Rect(_ROSTER_X + INNER_PAD, 200, 280, _BUY_H),
            [], font_size=12)
        self._add_btn = Button(
            pygame.Rect(_ROSTER_X + INNER_PAD + 280 + 6, 200, 120, _BUY_H),
            label='Buy Player', color=COLOR_BTN_NEUTRAL, font_size=12)

        # ── Staff state ───────────────────────────────────────────────────────
        self._staff: dict[str, int] = {k: 0 for k, _ in _STAFF_ITEMS}
        self._staff_minus: list[Button] = []
        self._staff_plus:  list[Button] = []
        self._build_staff_buttons()

        # ── Bottom bar ────────────────────────────────────────────────────────
        btn_y = SCREEN_H - 54
        self._cancel_btn = Button(
            pygame.Rect(PADDING, btn_y, 120, 42),
            label='Cancel', font_size=15, bg_image=get_button_image())
        self._save_btn = Button(
            pygame.Rect(SCREEN_W - PADDING - 150, btn_y, 150, 42),
            label='Save Team', font_size=15, bg_image=get_button_image())

        self._error_msg = ''
        self._update_race()
        self._update_pos_dd()

    def _commit_name_edit(self):
        if self._editing_idx >= 0 and self._editing_input is not None:
            name = self._editing_input.text.strip()
            if name and self._editing_idx < len(self._players):
                self._players[self._editing_idx].name = name
        self._editing_idx   = -1
        self._editing_input = None

    # ── Initialisation helpers ────────────────────────────────────────────────

    def _load_ruleset(self):
        try:
            self._ruleset, self._reroll_costs = _load_ruleset_for_size(self._board_size)
            self._races = sorted(r.name for r in self._ruleset.races)
        except Exception as e:
            print(f'TeamCreatorScreen: could not load ruleset: {e}')
            self._races = []

    def _build_staff_buttons(self):
        # Horizontal staff bar: 5 items spread across full roster width
        self._staff_minus = []
        self._staff_plus  = []
        n       = len(_STAFF_ITEMS)
        item_w  = (_ROSTER_FULL_W - (n - 1) * 4) // n   # ~154px each
        btn_y   = _STAFF_BAR_Y + 28
        for i in range(n):
            ix = _ROSTER_X + i * (item_w + 4)
            mx = ix + INNER_PAD
            px = mx + _BTN_SZ + 16
            self._staff_minus.append(
                Button(pygame.Rect(mx, btn_y, _BTN_SZ, _BTN_SZ),
                       label='−', color=COLOR_BTN_DEFAULT, font_size=12))
            self._staff_plus.append(
                Button(pygame.Rect(px, btn_y, _BTN_SZ, _BTN_SZ),
                       label='+', color=COLOR_BTN_DEFAULT, font_size=12))

    def _update_race(self):
        if self._races and self._ruleset:
            race_name = self._races[self._race_dd.selected % len(self._races)]
            for r in self._ruleset.races:
                if r.name == race_name:
                    self._race_obj = r
                    break

    def _update_pos_dd(self):
        opts, _ = self._available_positions()
        self._pos_dd.options  = opts
        self._pos_dd.selected = 0

    def _available_positions(self) -> tuple[list[str], list]:
        """Return (option_strings, role_objects) for positions still available."""
        if not self._race_obj:
            return [], []
        counts: dict[str, int] = {}
        for p in self._players:
            counts[p.role.name] = counts.get(p.role.name, 0) + 1
        opts, roles = [], []
        for role in self._race_obj.roles:
            if getattr(role, 'star_player', False):
                continue
            remaining = role.quantity - counts.get(role.name, 0)
            if remaining > 0:
                opts.append(f'{role.name}  ({remaining})  –  {role.cost // 1000}k')
                roles.append(role)
        return opts, roles

    # ── Budget helpers ────────────────────────────────────────────────────────

    def _calc_spent(self) -> int:
        player_tv    = sum(p.role.cost for p in self._players if p.role)
        reroll_cost  = self._race_obj.reroll_cost if self._race_obj else 0
        return (player_tv
                + self._staff['rerolls']      * reroll_cost
                + self._staff['apothecaries'] * APOTHECARY_COST
                + self._staff['ass_coaches']  * STAFF_COSTS['ass_coaches']
                + self._staff['cheerleaders'] * STAFF_COSTS['cheerleaders']
                + self._staff['fan_factor']   * STAFF_COSTS['fan_factor'])

    def _staff_max(self, key: str) -> int:
        if key == 'rerolls':
            return 8
        if key == 'apothecaries':
            return 1 if (self._race_obj and self._race_obj.apothecary) else 0
        if key == 'fan_factor':
            return 6
        return 99

    def _staff_cost(self, key: str) -> int:
        if key == 'rerolls':
            return self._race_obj.reroll_cost if self._race_obj else 0
        if key == 'apothecaries':
            return APOTHECARY_COST
        return STAFF_COSTS.get(key, 0)

    # ── Validation & save ─────────────────────────────────────────────────────

    def _validate(self) -> str:
        if not self._name_input.text.strip():
            return 'Team name is required.'
        spent = self._calc_spent()
        if spent > TEAM_BUDGET:
            return f'Over budget! {spent // 1000}k spent, limit {TEAM_BUDGET // 1000}k.'
        min_p = MIN_PLAYERS.get(self._board_size, 1)
        if len(self._players) < min_p:
            return f'Need at least {min_p} players ({len(self._players)} in roster).'
        if len(self._players) > MAX_PLAYERS:
            return f'Maximum {MAX_PLAYERS} players allowed.'
        return ''

    def _save(self):
        err = self._validate()
        if err:
            self._error_msg = err
            return

        name      = self._name_input.text.strip()
        race      = (self._races[self._race_dd.selected % len(self._races)]
                     if self._races else '')
        remaining = TEAM_BUDGET - self._calc_spent()
        slug      = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-') or 'team'

        data = {
            'name':        name,
            'race':        race,
            'treasury':    remaining,
            'apothecaries': self._staff['apothecaries'],
            'rerolls':     self._staff['rerolls'],
            'fan_factor':  self._staff['fan_factor'],
            'ass_coaches': self._staff['ass_coaches'],
            'cheerleaders': self._staff['cheerleaders'],
            'players': [
                {
                    'nr':           p.nr,
                    'name':         p.name,
                    'position':     p.role.name,
                    'extra_skills': [],
                    'extra_ma': 0, 'extra_st': 0, 'extra_ag': 0, 'extra_av': 0,
                    'spp': 0, 'niggling': False, 'mng': False,
                }
                for p in self._players
            ],
        }

        teams_dir = get_data_path(f'teams/{self._board_size}')
        os.makedirs(teams_dir, exist_ok=True)
        out_path = os.path.join(teams_dir, slug + '.json')
        try:
            with open(out_path, 'w') as fh:
                json.dump(data, fh, indent=2)
            self.app.pop_screen()
        except Exception as e:
            self._error_msg = f'Save failed: {e}'

    # ── Event handling ────────────────────────────────────────────────────────

    def update(self):
        pass

    def handle_event(self, event: pygame.event.Event):
        mouse = pygame.mouse.get_pos()
        self._cancel_btn.update_hover(mouse)
        self._save_btn.update_hover(mouse)
        self._add_btn.update_hover(mouse)
        for b in self._staff_minus + self._staff_plus:
            b.update_hover(mouse)

        if self._name_input.handle_event(event):
            self._error_msg = ''
        if self._editing_input is not None:
            if self._editing_input.handle_event(event):
                pass  # text changed
            if (event.type == pygame.KEYDOWN and
                    event.key in (pygame.K_RETURN, pygame.K_KP_ENTER)):
                self._commit_name_edit()
                return

        if event.type == pygame.MOUSEWHEEL:
            mouse = pygame.mouse.get_pos()
            for dd in (self._race_dd, self._size_dd, self._pos_dd):
                if dd.handle_scroll(mouse, event.y):
                    return
            # Scroll roster table
            roster_area = pygame.Rect(_ROSTER_X, _ROSTER_TOP,
                                      _ROSTER_FULL_W, _BUY_ROW_Y - _ROSTER_TOP)
            if roster_area.collidepoint(mouse):
                self._commit_name_edit()
                self._roster_scroll = max(0, self._roster_scroll - event.y * 20)
                return

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.pop_screen()
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Commit any active name edit if clicking outside the editing input
            if (self._editing_input is not None and
                    not self._editing_input.rect.collidepoint(event.pos)):
                self._commit_name_edit()

            # Close other dropdowns when one is clicked
            all_dds = [self._race_dd, self._size_dd, self._pos_dd]
            for dd in all_dds:
                if dd.rect.collidepoint(event.pos):
                    for other in all_dds:
                        if other is not dd:
                            other.close()
                    break

            # ── Race dropdown ────────────────────────────────────────────────
            if self._race_dd.handle_click(event.pos):
                self._update_race()
                self._players.clear()
                self._editing_idx = -1; self._editing_input = None
                self._staff  = {k: 0 for k, _ in _STAFF_ITEMS}
                self._update_pos_dd()
                self._error_msg = ''

            # ── Board-size dropdown ──────────────────────────────────────────
            if self._size_dd.handle_click(event.pos):
                self._board_size = _BOARD_SIZES[self._size_dd.selected]
                self._load_ruleset()
                self._race_dd.options  = self._races
                self._race_dd.selected = 0
                self._update_race()
                self._players.clear()
                self._editing_idx = -1; self._editing_input = None
                self._staff  = {k: 0 for k, _ in _STAFF_ITEMS}
                self._update_pos_dd()
                self._error_msg = ''

            # ── Position dropdown ────────────────────────────────────────────
            self._pos_dd.handle_click(event.pos)

            # ── Buy player ───────────────────────────────────────────────────
            if self._add_btn.is_clicked(event.pos):
                _, roles = self._available_positions()
                sel = self._pos_dd.selected % max(1, len(roles)) if roles else -1
                if 0 <= sel < len(roles):
                    role = roles[sel]
                    nr   = max((p.nr for p in self._players), default=0) + 1
                    self._players.append(_PlayerDraft(nr, f'Player {nr}', role))
                    self._update_pos_dd()
                    self._error_msg = ''

            # ── Remove player ────────────────────────────────────────────────
            roster_clip = pygame.Rect(
                _ROSTER_X, _ROSTER_TOP, _ROSTER_FULL_W, _ROSTER_BOTTOM - _ROSTER_TOP + 24)
            if roster_clip.collidepoint(event.pos):
                for i, btn_r in enumerate(self._remove_btns):
                    if btn_r.collidepoint(event.pos) and i < len(self._players):
                        if self._editing_idx == i:
                            self._editing_idx = -1
                            self._editing_input = None
                        elif self._editing_idx > i:
                            self._editing_idx -= 1
                        self._players.pop(i)
                        self._update_pos_dd()
                        self._error_msg = ''
                        break

            # ── Click on player name → inline edit ───────────────────────────
            rows_top = _ROSTER_TOP + _ROSTER_HDR_H
            name_rect = pygame.Rect(_COL_NAME_ABS, rows_top,
                                    _COL_NAME_W, _ROSTER_BOTTOM - rows_top)
            if name_rect.collidepoint(event.pos):
                for i, row_y in enumerate(self._player_row_screen_ys):
                    if rows_top <= row_y < _ROSTER_BOTTOM and row_y <= event.pos[1] < row_y + 25:
                        inp = TextInput(
                            pygame.Rect(_COL_NAME_ABS, row_y, _COL_NAME_W, 23),
                            placeholder='Name…', max_len=40, font_size=12)
                        inp.text   = self._players[i].name
                        inp.active = True
                        self._editing_idx   = i
                        self._editing_input = inp
                        break

            # ── Staff +/− ────────────────────────────────────────────────────
            for i, (key, _) in enumerate(_STAFF_ITEMS):
                if self._staff_minus[i].is_clicked(event.pos):
                    self._staff[key] = max(0, self._staff[key] - 1)
                    self._error_msg  = ''
                if self._staff_plus[i].is_clicked(event.pos):
                    if self._staff[key] < self._staff_max(key):
                        self._staff[key] += 1
                    self._error_msg = ''

            # ── Bottom bar ───────────────────────────────────────────────────
            if self._cancel_btn.is_clicked(event.pos):
                self.app.pop_screen()
                return
            if self._save_btn.is_clicked(event.pos):
                self._save()
                return

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface):
        surface.fill((10, 10, 16))
        self._draw_title_bar(surface)
        self._draw_header_inputs(surface)
        self._draw_roster_column(surface)
        self._draw_buy_row(surface)
        self._draw_staff_bar(surface)
        self._draw_bottom_bar(surface)
        # Draw dropdowns last; open one always on top
        _all_dds = [self._race_dd, self._size_dd, self._pos_dd]
        for _dd in _all_dds:
            if not _dd.open:
                _dd.draw(surface)
        for _dd in _all_dds:
            if _dd.open:
                _dd.draw(surface)

    def _draw_title_bar(self, surface: pygame.Surface):
        title_s = _fancy(20, bold=True).render('New Team', True, (210, 215, 240))
        surface.blit(title_s, (PADDING, 16))

        # Gold crowns counter (right-aligned)
        spent     = self._calc_spent()
        remaining = TEAM_BUDGET - spent
        over      = remaining < 0
        gc_col    = (220, 80, 80) if over else (195, 170, 75)
        gc_label  = (f'-{(-remaining) // 1000}k gc' if over
                     else f'{remaining // 1000}k gc')
        gc_s = _fancy(15, bold=True).render(gc_label, True, gc_col)
        surface.blit(gc_s, (SCREEN_W - PADDING - gc_s.get_width(), 20))

        pygame.draw.line(surface, (38, 42, 62),
                         (0, TITLE_H - 1), (SCREEN_W, TITLE_H - 1))

    def _draw_header_inputs(self, surface: pygame.Surface):
        lbl_y   = TITLE_H + 4
        lbl_col = (115, 120, 140)
        surface.blit(_font(10).render('Name',  True, lbl_col), (PADDING, lbl_y))
        surface.blit(_font(10).render('Race',  True, lbl_col),
                     (self._race_dd.rect.x, lbl_y))
        surface.blit(_font(10).render('Board', True, lbl_col),
                     (self._size_dd.rect.x, lbl_y))
        self._name_input.draw(surface)
        pygame.draw.line(surface, (38, 42, 62),
                         (0, _CONTENT_Y - 2), (SCREEN_W, _CONTENT_Y - 2))

    def _draw_roster_column(self, surface: pygame.Surface):
        race  = (self._races[self._race_dd.selected % len(self._races)]
                 if self._races else '')
        proxy = _RosterProxy(race, self._players)

        x     = _ROSTER_X + INNER_PAD
        x_max = _ROSTER_X_MAX - INNER_PAD

        # ── Fixed header (drawn without clip so it always shows) ──────────────
        # Use remove_btns=[] so BTN_COL_W=24 matches the player rows → columns align
        empty_proxy = _RosterProxy(race, [])
        draw_roster_table(surface, empty_proxy, x, _ROSTER_TOP, x_max,
                          _ROSTER_TOP + _ROSTER_HDR_H + 4,
                          True, remove_btns=[], show_skills=True)

        # ── Scrollable player rows ────────────────────────────────────────────
        rows_top = _ROSTER_TOP + _ROSTER_HDR_H
        rows_area = pygame.Rect(_ROSTER_X, rows_top,
                                _ROSTER_X_MAX - _ROSTER_X, _ROSTER_BOTTOM - rows_top)
        old_clip = surface.get_clip()
        surface.set_clip(rows_area)

        self._remove_btns = []
        self._player_row_screen_ys = []
        end_y = draw_roster_table(
            surface, proxy,
            x, rows_top - self._roster_scroll,
            x_max, 99999,
            True,
            remove_btns=self._remove_btns,
            show_skills=True,
            skip_header=True,
            out_row_ys=self._player_row_screen_ys,
        )
        # Track content height and clamp scroll
        self._roster_content_h = end_y - (rows_top - self._roster_scroll)
        rows_area_h = _ROSTER_BOTTOM - rows_top
        max_scroll = max(0, self._roster_content_h - rows_area_h)
        self._roster_scroll = min(self._roster_scroll, max_scroll)

        surface.set_clip(old_clip)

        # Draw inline name editor outside clip (so it renders over the table)
        if self._editing_input is not None and self._editing_idx >= 0:
            if self._editing_idx < len(self._player_row_screen_ys):
                row_y = self._player_row_screen_ys[self._editing_idx]
                rows_top_abs = _ROSTER_TOP + _ROSTER_HDR_H
                if rows_top_abs <= row_y < _ROSTER_BOTTOM:
                    self._editing_input.rect.y = row_y
                    self._editing_input.draw(surface)

    def _draw_buy_row(self, surface: pygame.Surface):
        """Fixed buy-player strip between roster and staff bar."""
        can_buy = (len(self._players) < MAX_PLAYERS and
                   bool(self._available_positions()[0]))
        row_rect = pygame.Rect(_ROSTER_X, _BUY_ROW_Y, _ROSTER_FULL_W, _BUY_ROW_H)
        draw_bordered_rect(surface, (18, 26, 38), (48, 58, 78), row_rect, radius=3, border_w=1)

        # Update y positions (x is fixed)
        by = _BUY_ROW_Y + (_BUY_ROW_H - self._add_btn.rect.height) // 2
        self._pos_dd.rect.y  = by
        self._add_btn.rect.y = by
        self._add_btn.disabled = not can_buy
        self._add_btn.draw(surface)

        # "Max players reached" hint
        if not can_buy and len(self._players) >= MAX_PLAYERS:
            hint = _font(10).render('Max 16 players', True, (100, 100, 120))
            surface.blit(hint, (self._add_btn.rect.right + 8,
                                by + (self._add_btn.rect.height - hint.get_height()) // 2))

    def _draw_staff_bar(self, surface: pygame.Surface):
        """Draw the horizontal staff strip below the roster table."""
        bar_rect = pygame.Rect(_ROSTER_X, _STAFF_BAR_Y, _ROSTER_FULL_W, _STAFF_BAR_H)
        draw_bordered_rect(surface, (18, 18, 24), (48, 50, 68),
                           bar_rect, radius=4, border_w=1)

        # Divider line above bar
        pygame.draw.line(surface, (40, 44, 60),
                         (_ROSTER_X, _STAFF_BAR_Y - 2),
                         (_ROSTER_X_MAX, _STAFF_BAR_Y - 2))

        n      = len(_STAFF_ITEMS)
        item_w = (_ROSTER_FULL_W - (n - 1) * 4) // n

        for i, (key, label) in enumerate(_STAFF_ITEMS):
            ix        = _ROSTER_X + i * (item_w + 4)
            val       = self._staff[key]
            item_cost = self._staff_cost(key) * val
            is_apo    = (key == 'apothecaries')
            allowed   = (not is_apo) or (self._race_obj and self._race_obj.apothecary)

            # Label + cost on same row, left-aligned
            lbl_col = (100, 100, 120) if (is_apo and not allowed) else (145, 150, 175)
            lbl_s   = _font(10).render(label, True, lbl_col)
            cost_s  = _font(10).render(f'  {item_cost // 1000}k', True, (155, 145, 85))
            surface.blit(lbl_s,  (ix + INNER_PAD, _STAFF_BAR_Y + 6))
            surface.blit(cost_s, (ix + INNER_PAD + lbl_s.get_width(), _STAFF_BAR_Y + 6))

            # Disabled state
            self._staff_minus[i].disabled = (val <= 0) or not allowed
            self._staff_plus[i].disabled  = (val >= self._staff_max(key)) or not allowed
            self._staff_minus[i].draw(surface)
            self._staff_plus[i].draw(surface)

            # Value between buttons
            val_s = _font(12, bold=True).render(str(val), True, (200, 205, 225))
            mx    = self._staff_minus[i].rect
            px    = self._staff_plus[i].rect
            vx    = (mx.right + px.left) // 2 - val_s.get_width() // 2
            vy    = mx.y + (mx.height - val_s.get_height()) // 2
            surface.blit(val_s, (vx, vy))

            # N/A indicator for apothecary
            if is_apo and not allowed:
                na = _font(9).render('N/A', True, (80, 80, 95))
                surface.blit(na, (ix + INNER_PAD, _STAFF_BAR_Y + 38))

        # Error message
        if self._error_msg:
            err_s = _font(11).render(self._error_msg, True, (220, 80, 80))
            surface.blit(err_s, (_ROSTER_X + INNER_PAD,
                                  _STAFF_BAR_Y + _STAFF_BAR_H - 18))

    def _draw_bottom_bar(self, surface: pygame.Surface):
        pygame.draw.line(surface, (40, 42, 58),
                         (0, _BOTTOM_BAR_Y), (SCREEN_W, _BOTTOM_BAR_Y))
        self._cancel_btn.draw(surface)
        self._save_btn.draw(surface)

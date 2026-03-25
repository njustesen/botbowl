"""
Game creation screen — mode selection, team/agent picker, roster preview.
"""
from __future__ import annotations
import pygame
from typing import Optional

import botbowl
from botbowl.core.load import load_all_teams, load_team_by_filename, load_rule_set
from botbowl.core.model import Agent, Action
from botbowl.core.table import ActionType
from botbowl.ai.registry import list_bots
from botbowl.gui.rendering.ui_primitives import (
    Button, COLOR_BTN_DEFAULT, COLOR_BTN_HOME, COLOR_BTN_AWAY,
    COLOR_BTN_NEUTRAL, COLOR_TEXT, COLOR_TEXT_DIM, COLOR_PANEL_BG, COLOR_BORDER
)
from botbowl.gui import sprites as spr

CREATE_W = 900
CREATE_H = 620

PADDING = 20
ROW_H = 34


def _font(size: int = 13, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont('Arial', size, bold=bold)


_GAME_MODES = [
    {'label': '1v1', 'config': 'gym-1', 'board_size': 1},
    {'label': '3v3', 'config': 'gym-3', 'board_size': 3},
    {'label': '5v5', 'config': 'gym-5', 'board_size': 5},
    {'label': '7v7', 'config': 'gym-7', 'board_size': 7},
    {'label': '11v11', 'config': 'bot-bowl', 'board_size': 11},
]


class Dropdown:
    """Simple dropdown widget."""

    def __init__(self, rect: pygame.Rect, options: list[str],
                 selected: int = 0, font_size: int = 12):
        self.rect = rect
        self.options = options
        self.selected = selected
        self.font_size = font_size
        self.open = False

    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, (45, 45, 55), self.rect, border_radius=3)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, 1, border_radius=3)
        font = _font(self.font_size)
        label = self.options[self.selected] if self.options else ''
        surf = font.render(label[:40], True, COLOR_TEXT)
        surface.blit(surf, (self.rect.x + 6, self.rect.y + (self.rect.height - surf.get_height()) // 2))
        # Arrow
        arrow = font.render('▾', True, (160, 160, 160))
        surface.blit(arrow, (self.rect.right - 18, self.rect.y + (self.rect.height - arrow.get_height()) // 2))

        if self.open:
            item_h = self.rect.height
            for i, opt in enumerate(self.options):
                item_rect = pygame.Rect(self.rect.x, self.rect.bottom + i * item_h,
                                        self.rect.width, item_h)
                bg = (60, 60, 80) if i == self.selected else (40, 40, 52)
                pygame.draw.rect(surface, bg, item_rect)
                pygame.draw.rect(surface, COLOR_BORDER, item_rect, 1)
                item_surf = font.render(opt[:40], True, COLOR_TEXT)
                surface.blit(item_surf, (item_rect.x + 6,
                                         item_rect.y + (item_h - item_surf.get_height()) // 2))

    def handle_click(self, pos: tuple) -> bool:
        """Returns True if selection changed."""
        if self.rect.collidepoint(pos):
            self.open = not self.open
            return False
        if self.open:
            item_h = self.rect.height
            for i in range(len(self.options)):
                item_rect = pygame.Rect(self.rect.x, self.rect.bottom + i * item_h,
                                        self.rect.width, item_h)
                if item_rect.collidepoint(pos):
                    changed = (i != self.selected)
                    self.selected = i
                    self.open = False
                    return changed
            self.open = False
        return False

    @property
    def value(self) -> Optional[str]:
        return self.options[self.selected] if self.options else None


class CreateGameScreen:
    """Two-step game creation: mode selection then team/agent selection."""

    def __init__(self, app):
        self.app = app
        self.width = CREATE_W
        self.height = CREATE_H
        self._resize_display()

        self._step = 0  # 0 = mode, 1 = team/agent
        self._selected_mode = 4  # default 11v11
        self._mode_buttons: list[Button] = []
        self._build_mode_buttons()

        # Step 1 state
        self._teams: list = []
        self._team_names: list[str] = []
        self._bot_names: list[str] = ['Human'] + list_bots()
        self._home_team_dd: Optional[Dropdown] = None
        self._away_team_dd: Optional[Dropdown] = None
        self._home_agent_dd: Optional[Dropdown] = None
        self._away_agent_dd: Optional[Dropdown] = None
        self._start_btn: Optional[Button] = None
        self._back_btn: Optional[Button] = None
        self._preview_team: Optional[object] = None  # Team being previewed

    def _resize_display(self):
        current = pygame.display.get_surface()
        if current is None or current.get_size() != (self.width, self.height):
            pygame.display.set_mode((self.width, self.height))

    def _build_mode_buttons(self):
        self._mode_buttons = []
        total_w = len(_GAME_MODES) * 120 + (len(_GAME_MODES) - 1) * 20
        start_x = (self.width - total_w) // 2
        y = self.height // 2 - 30
        for i, mode in enumerate(_GAME_MODES):
            x = start_x + i * 140
            color = COLOR_BTN_NEUTRAL if i == self._selected_mode else COLOR_BTN_DEFAULT
            btn = Button(
                pygame.Rect(x, y, 120, 60),
                label=mode['label'],
                color=color,
                font_size=20
            )
            self._mode_buttons.append(btn)

    def _load_teams_for_mode(self):
        mode = _GAME_MODES[self._selected_mode]
        config = botbowl.load_config(mode['config'])
        ruleset = botbowl.load_rule_set(config.ruleset)
        self._teams = load_all_teams(ruleset, board_size=mode['board_size'])
        self._team_names = [t.name for t in self._teams]

        dd_w, dd_h = 200, 28
        x_home = PADDING
        x_away = self.width // 2 + PADDING
        y_team = 100
        y_agent = y_team + dd_h + 12

        self._home_team_dd = Dropdown(
            pygame.Rect(x_home, y_team, dd_w, dd_h), self._team_names)
        self._away_team_dd = Dropdown(
            pygame.Rect(x_away, y_team, dd_w, dd_h), self._team_names,
            selected=min(1, len(self._team_names) - 1))

        self._home_agent_dd = Dropdown(
            pygame.Rect(x_home, y_agent, dd_w, dd_h), self._bot_names)
        self._away_agent_dd = Dropdown(
            pygame.Rect(x_away, y_agent, dd_w, dd_h), self._bot_names)

        self._start_btn = Button(
            pygame.Rect(self.width // 2 - 80, self.height - 60, 160, 40),
            label='Start Game', color=COLOR_BTN_NEUTRAL, font_size=15)
        self._back_btn = Button(
            pygame.Rect(PADDING, self.height - 60, 100, 40),
            label='← Back', color=COLOR_BTN_DEFAULT, font_size=13)

    def update(self):
        pass

    def handle_event(self, event: pygame.event.Event):
        mouse = pygame.mouse.get_pos()

        if self._step == 0:
            for btn in self._mode_buttons:
                btn.update_hover(mouse)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, btn in enumerate(self._mode_buttons):
                    if btn.is_clicked(event.pos):
                        self._selected_mode = i
                        self._step = 1
                        self._load_teams_for_mode()
                        return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.app.pop_screen()

        elif self._step == 1:
            for dd in [self._home_team_dd, self._away_team_dd,
                       self._home_agent_dd, self._away_agent_dd]:
                if dd:
                    dd.update_hover(mouse) if hasattr(dd, 'update_hover') else None

            if self._start_btn:
                self._start_btn.update_hover(mouse)
            if self._back_btn:
                self._back_btn.update_hover(mouse)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._back_btn and self._back_btn.is_clicked(event.pos):
                    self._step = 0
                    return
                if self._start_btn and self._start_btn.is_clicked(event.pos):
                    self._start_game()
                    return
                # Handle dropdowns (close others when one opens)
                for dd in [self._home_team_dd, self._away_team_dd,
                           self._home_agent_dd, self._away_agent_dd]:
                    if dd:
                        changed = dd.handle_click(event.pos)

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
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

    def draw(self, surface: pygame.Surface):
        surface.fill((15, 15, 20))

        if self._step == 0:
            self._draw_mode_select(surface)
        else:
            self._draw_team_select(surface)

    def _draw_mode_select(self, surface: pygame.Surface):
        title = _font(24, bold=True).render('Select Game Mode', True, COLOR_TEXT)
        surface.blit(title, ((self.width - title.get_width()) // 2, 80))

        for i, btn in enumerate(self._mode_buttons):
            # Highlight selected
            if i == self._selected_mode:
                pygame.draw.rect(surface, COLOR_BTN_NEUTRAL,
                                 btn.rect.inflate(4, 4), border_radius=6)
            btn.draw(surface)

    def _draw_team_select(self, surface: pygame.Surface):
        title = _font(18, bold=True).render('Select Teams & Agents', True, COLOR_TEXT)
        surface.blit(title, (PADDING, 20))

        mode = _GAME_MODES[self._selected_mode]
        mode_lbl = _font(13).render(f'Mode: {mode["label"]}', True, (120, 160, 120))
        surface.blit(mode_lbl, (PADDING, 50))

        # Labels
        lf = _font(12, bold=True)
        surface.blit(lf.render('Home Team', True, (100, 130, 220)), (PADDING, 78))
        surface.blit(lf.render('Home Agent', True, (100, 130, 220)), (PADDING, 78 + 40))
        x_away = self.width // 2 + PADDING
        surface.blit(lf.render('Away Team', True, (220, 100, 80)), (x_away, 78))
        surface.blit(lf.render('Away Agent', True, (220, 100, 80)), (x_away, 78 + 40))

        # Dropdowns
        for dd in [self._home_team_dd, self._away_team_dd,
                   self._home_agent_dd, self._away_agent_dd]:
            if dd:
                dd.draw(surface)

        # Roster preview (selected home team)
        if self._home_team_dd and self._teams:
            team = self._teams[self._home_team_dd.selected % len(self._teams)]
            self._draw_roster(surface, team, PADDING, 200, is_home=True)

        if self._away_team_dd and self._teams:
            team = self._teams[self._away_team_dd.selected % len(self._teams)]
            self._draw_roster(surface, team, self.width // 2 + PADDING, 200, is_home=False)

        if self._start_btn:
            self._start_btn.draw(surface)
        if self._back_btn:
            self._back_btn.draw(surface)

    def _draw_roster(self, surface: pygame.Surface, team, x: int, y: int,
                     is_home: bool):
        font = _font(11)
        header = _font(12, bold=True)
        color = (100, 130, 220) if is_home else (220, 100, 80)

        name_surf = header.render(f'{team.name} ({team.race})', True, color)
        surface.blit(name_surf, (x, y))
        y += 20

        # Resource info
        res_parts = []
        if hasattr(team, 'rerolls'):
            res_parts.append(f'RR:{team.rerolls}')
        if hasattr(team, 'apothecary') and team.apothecary:
            res_parts.append('Apo')
        if res_parts:
            res_surf = font.render('  '.join(res_parts), True, (160, 160, 160))
            surface.blit(res_surf, (x, y))
            y += 16

        # Player list header
        hdr = font.render('Nr  Name              MA ST AG AV', True, (120, 120, 140))
        surface.blit(hdr, (x, y))
        y += 14
        pygame.draw.line(surface, (50, 50, 70),
                         (x, y), (x + 350, y))
        y += 4

        # Player rows (limit to ~12)
        roster = team.players if hasattr(team, 'players') else []
        for player in roster[:12]:
            role = player.role.name if player.role else '?'
            ma = player.role.ma if player.role else 0
            st = player.role.st if player.role else 0
            ag = player.role.ag if player.role else 0
            av = player.role.av if player.role else 0
            skills = ', '.join(s.name for s in (player.role.skills or []))
            line1 = f'{player.nr:2d}  {player.name[:16]:<16}  {ma}  {st}  {ag}  {av}'
            p_surf = font.render(line1, True, (200, 200, 200))
            surface.blit(p_surf, (x, y))
            y += 13
            if skills:
                sk_surf = _font(10).render(f'    {skills[:50]}', True, (100, 180, 100))
                surface.blit(sk_surf, (x, y))
                y += 12

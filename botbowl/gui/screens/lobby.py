"""
Lobby screen — lists active games, saved games, and replays.
"""
from __future__ import annotations
import pygame
from typing import Optional

from botbowl.gui.save_load import list_saves, load_game, delete_save, list_replays, load_replay, delete_replay
from botbowl.gui.rendering.ui_primitives import (
    Button, COLOR_BTN_DEFAULT, COLOR_BTN_HOME, COLOR_BTN_AWAY,
    COLOR_BTN_NEUTRAL, COLOR_PANEL_BG, COLOR_BORDER, COLOR_TEXT,
    COLOR_TEXT_DIM
)

LOBBY_W = 1080
LOBBY_H = 700

ROW_H = 36
HEADER_H = 50
SECTION_GAP = 30
PADDING = 20


def _font(size: int = 14, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont('Arial', size, bold=bold)


class LobbyScreen:
    """
    Lobby: lists saved games and replays, provides New Game button.
    Active in-memory games are tracked by the app (self.app.active_games).
    """

    def __init__(self, app):
        self.app = app
        self.width = LOBBY_W
        self.height = LOBBY_H
        self._resize_display()

        self._saves: list[str] = []
        self._replays: list[str] = []
        self._buttons: list[Button] = []
        self._scroll_y = 0
        self._refresh()

    def _resize_display(self):
        current = pygame.display.get_surface()
        if current is None or current.get_size() != (self.width, self.height):
            pygame.display.set_mode((self.width, self.height))

    def _refresh(self):
        self._saves = list_saves()
        self._replays = list_replays()
        self._build_buttons()

    def _build_buttons(self):
        self._buttons = []
        x_base = PADDING
        y = HEADER_H + PADDING

        # New Game button
        self._new_game_btn = Button(
            pygame.Rect(self.width - 170 - PADDING, 13, 170, 38),
            label='+ New Game',
            color=COLOR_BTN_NEUTRAL,
            font_size=14
        )

        btn_del_x = self.width - PADDING - 86
        btn_act_x = btn_del_x - 90

        # Saved games section
        self._save_rows: list[tuple] = []  # (name, load_btn, del_btn)
        y_saves = y + 30  # after section header
        for name in self._saves:
            load_btn = Button(
                pygame.Rect(btn_act_x, y_saves, 82, ROW_H - 6),
                label='Load', color=COLOR_BTN_HOME, font_size=12)
            del_btn = Button(
                pygame.Rect(btn_del_x, y_saves, 82, ROW_H - 6),
                label='Delete', color=COLOR_BTN_AWAY, font_size=12)
            self._save_rows.append((name, load_btn, del_btn))
            y_saves += ROW_H

        # Replays section
        y_replays = y_saves + SECTION_GAP + 30
        self._replay_rows: list[tuple] = []  # (name, play_btn, del_btn)
        for name in self._replays:
            play_btn = Button(
                pygame.Rect(btn_act_x, y_replays, 82, ROW_H - 6),
                label='Play', color=COLOR_BTN_NEUTRAL, font_size=12)
            del_btn = Button(
                pygame.Rect(btn_del_x, y_replays, 82, ROW_H - 6),
                label='Delete', color=COLOR_BTN_AWAY, font_size=12)
            self._replay_rows.append((name, play_btn, del_btn))
            y_replays += ROW_H

    def update(self):
        pass

    def handle_event(self, event: pygame.event.Event):
        mouse = pygame.mouse.get_pos()
        self._new_game_btn.update_hover(mouse)
        for _, lb, db in self._save_rows:
            lb.update_hover(mouse)
            db.update_hover(mouse)
        for _, pb, db in self._replay_rows:
            pb.update_hover(mouse)
            db.update_hover(mouse)

        if event.type == pygame.MOUSEWHEEL:
            self._scroll_y = max(0, self._scroll_y - event.y * 20)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # New game
            if self._new_game_btn.is_clicked(event.pos):
                from botbowl.gui.screens.create_game import CreateGameScreen
                self.app.push_screen(CreateGameScreen(self.app))
                return

            # Saved games
            for name, lb, db in self._save_rows:
                if lb.is_clicked(event.pos):
                    self._load_saved_game(name)
                    return
                if db.is_clicked(event.pos):
                    delete_save(name)
                    self._refresh()
                    return

            # Replays
            for name, pb, db in self._replay_rows:
                if pb.is_clicked(event.pos):
                    self._load_replay(name)
                    return
                if db.is_clicked(event.pos):
                    delete_replay(name)
                    self._refresh()
                    return

    def _load_saved_game(self, name: str):
        try:
            game = load_game(name)
            from botbowl.gui.screens.game_screen import GameScreen
            screen = GameScreen(
                self.app, game,
                home_agent=game.home_agent,
                away_agent=game.away_agent,
                spectating=game.home_agent.human and game.away_agent.human,
                ai_delay_ms=self.app.ai_delay_ms
            )
            self.app.push_screen(screen)
        except Exception as e:
            print(f'Error loading save {name}: {e}')

    def _load_replay(self, name: str):
        try:
            replay = load_replay(name)
            from botbowl.gui.screens.replay_screen import ReplayScreen
            screen = ReplayScreen(self.app, replay)
            self.app.push_screen(screen)
        except Exception as e:
            print(f'Error loading replay {name}: {e}')

    def draw(self, surface: pygame.Surface):
        surface.fill((15, 15, 20))

        # Title
        title_font = _font(26, bold=True)
        title = title_font.render('botbowl', True, (210, 215, 245))
        surface.blit(title, (PADDING, 12))

        sub_font = _font(12)
        sub = sub_font.render('Blood Bowl Simulator', True, (100, 105, 130))
        surface.blit(sub, (PADDING, 42))

        # Accent line under title
        accent_surf = pygame.Surface((200, 2), pygame.SRCALPHA)
        for px in range(200):
            r = int(70 + (210 - 70) * px / 200)
            g = int(110 + (75 - 110) * px / 200)
            b = int(210 + (55 - 210) * px / 200)
            accent_surf.set_at((px, 0), (r, g, b, 200))
            accent_surf.set_at((px, 1), (r, g, b, 200))
        surface.blit(accent_surf, (PADDING, 58))

        self._new_game_btn.draw(surface)

        y = HEADER_H + PADDING - self._scroll_y

        # Saved games section
        self._draw_section_header(surface, 'Saved Games', y)
        y += 30

        if not self._saves:
            empty = _font(12).render('No saved games.', True, COLOR_TEXT_DIM)
            surface.blit(empty, (PADDING, y + 8))
            y += ROW_H
        else:
            for name, lb, db in self._save_rows:
                self._draw_row(surface, name, lb, db, y)
                y += ROW_H

        y += SECTION_GAP

        # Replays section
        self._draw_section_header(surface, 'Replays', y)
        y += 30

        if not self._replays:
            empty = _font(12).render('No replays found.', True, COLOR_TEXT_DIM)
            surface.blit(empty, (PADDING, y + 8))
        else:
            for name, pb, db in self._replay_rows:
                self._draw_row(surface, name, pb, db, y)
                y += ROW_H

    def _draw_section_header(self, surface: pygame.Surface, title: str, y: int):
        font = _font(15, bold=True)
        surf = font.render(title, True, (155, 160, 200))
        surface.blit(surf, (PADDING, y))
        pygame.draw.line(surface, (45, 48, 68),
                         (PADDING, y + 22), (self.width - PADDING, y + 22))

    def _draw_row(self, surface: pygame.Surface, name: str,
                  btn1: Button, btn2: Button, y: int):
        # Row background
        row_rect = pygame.Rect(PADDING, y, self.width - PADDING * 2 - 1, ROW_H - 2)
        pygame.draw.rect(surface, (25, 25, 32), row_rect, border_radius=3)

        font = _font(12)
        label = name[:60] + ('...' if len(name) > 60 else '')
        surf = font.render(label, True, COLOR_TEXT)
        surface.blit(surf, (PADDING + 8, y + (ROW_H - surf.get_height()) // 2))

        btn1.draw(surface)
        btn2.draw(surface)

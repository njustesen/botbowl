"""
Replay List screen — lists saved replays with play/delete actions.
"""
from __future__ import annotations
import pygame

from botbowl.gui.gui import SCREEN_W, SCREEN_H
from botbowl.gui.save_load import list_replays, load_replay, delete_replay
from botbowl.gui.rendering.ui_primitives import (
    Button, get_button_image,
    COLOR_TEXT, COLOR_TEXT_DIM
)

ROW_H = 36
HEADER_H = 70
PADDING = 20
SECTION_TOP = HEADER_H + PADDING


from botbowl.gui.fonts import get_font as _fancy, get_body_font as _font


class ReplayListScreen:
    """Lists saved replays with Play and Delete actions."""

    def __init__(self, app):
        self.app = app
        self._replays: list[str] = []
        self._replay_rows: list[tuple] = []  # (name, play_btn, del_btn)
        self._scroll_y = 0

        self._back_btn = Button(
            pygame.Rect(PADDING, SCREEN_H - 54, 120, 42),
            label='Back', font_size=15, bg_image=get_button_image()
        )
        self._refresh()

    def _refresh(self):
        self._replays = list_replays()
        self._build_rows()

    def _build_rows(self):
        self._replay_rows = []
        btn_del_x = SCREEN_W - PADDING - 86
        btn_act_x = btn_del_x - 90
        y = SECTION_TOP + 30  # below section header
        for name in self._replays:
            play_btn = Button(
                pygame.Rect(btn_act_x, y, 82, ROW_H - 6),
                label='Play', font_size=12, bg_image=get_button_image('green'))
            del_btn = Button(
                pygame.Rect(btn_del_x, y, 82, ROW_H - 6),
                label='Delete', font_size=12, bg_image=get_button_image('red'))
            self._replay_rows.append((name, play_btn, del_btn))
            y += ROW_H

    def update(self):
        pass

    def handle_event(self, event: pygame.event.Event):
        mouse = pygame.mouse.get_pos()
        self._back_btn.update_hover(mouse)
        for _, pb, db in self._replay_rows:
            pb.update_hover(mouse)
            db.update_hover(mouse)

        if event.type == pygame.MOUSEWHEEL:
            self._scroll_y = max(0, self._scroll_y - event.y * 20)

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.pop_screen()
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._back_btn.is_clicked(event.pos):
                self.app.pop_screen()
                return
            for name, pb, db in self._replay_rows:
                if pb.is_clicked(event.pos):
                    self._play_replay(name)
                    return
                if db.is_clicked(event.pos):
                    delete_replay(name)
                    self._refresh()
                    return

    def _play_replay(self, name: str):
        try:
            replay = load_replay(name)
            from botbowl.gui.screens.replay_screen import ReplayScreen
            self.app.push_screen(ReplayScreen(self.app, replay))
        except Exception as e:
            print(f'Error loading replay {name}: {e}')

    def draw(self, surface: pygame.Surface):
        surface.fill((15, 15, 20))

        # Page title
        title = _fancy(24, bold=True).render('Replays', True, (210, 215, 245))
        surface.blit(title, (PADDING, 16))
        pygame.draw.line(surface, (45, 48, 68), (PADDING, 58), (SCREEN_W - PADDING, 58))

        y = SECTION_TOP - self._scroll_y

        # Section header
        hdr = _fancy(13, bold=True).render('Saved Replays', True, (155, 160, 200))
        surface.blit(hdr, (PADDING, y))
        pygame.draw.line(surface, (45, 48, 68),
                         (PADDING, y + 20), (SCREEN_W - PADDING, y + 20))
        y += 30

        if not self._replays:
            empty = _font(12).render('No replays found.', True, COLOR_TEXT_DIM)
            surface.blit(empty, (PADDING, y + 8))
        else:
            for name, pb, db in self._replay_rows:
                row_rect = pygame.Rect(PADDING, y, SCREEN_W - PADDING * 2 - 1, ROW_H - 2)
                pygame.draw.rect(surface, (25, 25, 32), row_rect, border_radius=3)
                label = name[:70] + ('...' if len(name) > 70 else '')
                surf = _font(12).render(label, True, COLOR_TEXT)
                surface.blit(surf, (PADDING + 8, y + (ROW_H - surf.get_height()) // 2))
                pb.draw(surface)
                db.draw(surface)
                y += ROW_H

        self._back_btn.draw(surface)

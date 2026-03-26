"""
Main menu screen — classic game menu (New Game, Load Game, Replays, Exit).
"""
from __future__ import annotations
import os
import sys
import pygame

from botbowl.gui.gui import SCREEN_W, SCREEN_H
from botbowl.gui.rendering.ui_primitives import (
    Button, COLOR_BTN_NEUTRAL, COLOR_BTN_DEFAULT, COLOR_TEXT, get_button_image
)

_LOGO_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'img', 'botbowl.png'))

LOGO_H = 170        # displayed logo height
LOGO_Y = 55         # vertical position of logo
BTN_W = 340
BTN_H = 56
BTN_GAP = 14
MENU_START_Y = LOGO_Y + LOGO_H + 45 + BTN_H  # buttons start below the logo

# (label, image variant)
_MENU_ITEMS = [
    ('New Game',  'default'),
    ('Load Game', 'default'),
    ('Teams',     'default'),
    ('Replays',   'default'),
    ('Exit',      'red'),
]


class MainMenuScreen:
    """Classic four-option main menu."""

    def __init__(self, app):
        self.app = app
        self._logo: pygame.Surface = self._load_logo()
        self._buttons: list[Button] = []
        self._selected = 0  # keyboard navigation index
        self._build_buttons()

    def _load_logo(self) -> pygame.Surface:
        if not os.path.exists(_LOGO_PATH):
            return None
        try:
            import numpy as np
            raw   = pygame.image.load(_LOGO_PATH).convert_alpha()
            alpha = pygame.surfarray.array_alpha(raw)
            # Crop to bounding box of non-transparent pixels
            visible = alpha > 10
            cols = np.any(visible, axis=1)
            rows = np.any(visible, axis=0)
            if cols.any() and rows.any():
                x0, x1 = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
                y0, y1 = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
                pad = 4
                x0 = max(0, x0 - pad); x1 = min(raw.get_width(),  x1 + pad)
                y0 = max(0, y0 - pad); y1 = min(raw.get_height(), y1 + pad)
                raw = raw.subsurface(pygame.Rect(x0, y0, x1 - x0, y1 - y0)).copy()
            aspect = raw.get_width() / raw.get_height()
            logo_w = min(SCREEN_W - 60, int(LOGO_H * aspect))
            return pygame.transform.smoothscale(raw, (logo_w, LOGO_H))
        except Exception:
            return None

    def _build_buttons(self):
        self._buttons = []
        bx = (SCREEN_W - BTN_W) // 2
        for i, (label, variant) in enumerate(_MENU_ITEMS):
            y = MENU_START_Y + i * (BTN_H + BTN_GAP)
            btn = Button(pygame.Rect(bx, y, BTN_W, BTN_H),
                         label=label, font_size=17, bg_image=get_button_image(variant))
            btn.focused = (i == self._selected)
            self._buttons.append(btn)

    def _set_selected(self, index: int):
        self._buttons[self._selected].focused = False
        self._selected = index
        self._buttons[self._selected].focused = True

    def update(self):
        pass

    def handle_event(self, event: pygame.event.Event):
        mouse = pygame.mouse.get_pos()
        for btn in self._buttons:
            btn.update_hover(mouse)

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_DOWN, pygame.K_TAB):
                self._set_selected((self._selected + 1) % len(self._buttons))
            elif event.key == pygame.K_UP:
                self._set_selected((self._selected - 1) % len(self._buttons))
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._activate(self._selected)
            elif event.key == pygame.K_ESCAPE:
                sys.exit(0)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, btn in enumerate(self._buttons):
                if btn.is_clicked(event.pos):
                    self._activate(i)
                    return

    def _activate(self, index: int):
        if index == 0:  # New Game
            from botbowl.gui.screens.create_game import CreateGameScreen
            self.app.push_screen(CreateGameScreen(self.app))
        elif index == 1:  # Load Game
            from botbowl.gui.screens.load_game import LoadGameScreen
            self.app.push_screen(LoadGameScreen(self.app))
        elif index == 2:  # Teams
            from botbowl.gui.screens.teams import TeamsScreen
            self.app.push_screen(TeamsScreen(self.app))
        elif index == 3:  # Replays
            from botbowl.gui.screens.replay_list import ReplayListScreen
            self.app.push_screen(ReplayListScreen(self.app))
        elif index == 4:  # Exit
            sys.exit(0)

    def draw(self, surface: pygame.Surface):
        surface.fill((15, 15, 20))

        # BotBowl logo
        if self._logo is not None:
            lx = (SCREEN_W - self._logo.get_width()) // 2
            surface.blit(self._logo, (lx, LOGO_Y))

        # Menu buttons — highlight keyboard-selected item with a ring
        for i, btn in enumerate(self._buttons):
            btn.draw(surface)


# Keep LobbyScreen as an alias so existing imports still work
LobbyScreen = MainMenuScreen

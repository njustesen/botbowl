"""
Minimal screen that shows the Exit Game modal immediately — for layout iteration.
Used with: python -m botbowl.gui --screen modal --auto-screenshot /tmp/modal.png
"""
from __future__ import annotations
import pygame
from botbowl.gui.gui import SCREEN_W, SCREEN_H
from botbowl.gui.rendering.ui_primitives import Modal


class ModalTestScreen:
    def __init__(self, app):
        self.app = app
        screen_size = (SCREEN_W, SCREEN_H)
        self._modal = Modal(
            screen_size, 'Exit Game?', None,
            ok_label='Exit', cancel_label='Cancel',
            description='Any unsaved data will be lost.',
            ok_variant='red',
        )

    def update(self):
        pass

    def handle_event(self, event: pygame.event.Event):
        result = self._modal.handle_event(event)
        if result in ('ok', 'cancel'):
            self.app.pop_screen()

    def draw(self, surface: pygame.Surface):
        surface.fill((30, 45, 30))  # fake game board background
        self._modal.draw(surface)

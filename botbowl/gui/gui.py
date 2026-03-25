"""
Main application controller — screen stack, pygame init, main loop.
"""
from __future__ import annotations
import pygame
import sys


class App:
    """
    Simple screen-stack application.
    Each screen implements: update(), handle_event(event), draw(surface).
    """

    def __init__(self, title: str = 'botbowl', fps: int = 60,
                 ai_delay_ms: int = 50):
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption(title)

        self.fps = fps
        self.ai_delay_ms = ai_delay_ms
        self._screen_stack: list = []
        self._running = False
        self._clock = pygame.time.Clock()

    def push_screen(self, screen):
        self._screen_stack.append(screen)

    def pop_screen(self):
        if self._screen_stack:
            self._screen_stack.pop()

    def replace_screen(self, screen):
        if self._screen_stack:
            self._screen_stack.pop()
        self._screen_stack.append(screen)

    @property
    def current_screen(self):
        return self._screen_stack[-1] if self._screen_stack else None

    def run(self):
        self._running = True
        while self._running:
            if not self._screen_stack:
                self._running = False
                break

            surface = pygame.display.get_surface()
            if surface is None:
                break

            screen = self.current_screen

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    break
                screen.handle_event(event)

            screen.update()
            screen.draw(surface)
            pygame.display.flip()
            self._clock.tick(self.fps)

        pygame.quit()

    def quit(self):
        self._running = False

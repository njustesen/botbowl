"""
Reusable pygame UI primitives: Button, ScrollPanel, Modal, TextInput.
"""
from __future__ import annotations
import pygame
from typing import Optional, Any

# Colors
COLOR_BTN_DEFAULT = (80, 80, 90)
COLOR_BTN_HOME = (40, 80, 160)
COLOR_BTN_AWAY = (160, 60, 40)
COLOR_BTN_NEUTRAL = (60, 110, 60)
COLOR_BTN_DISABLED = (50, 50, 50)
COLOR_BTN_HOVER = (120, 120, 140)
COLOR_TEXT = (230, 230, 230)
COLOR_TEXT_DIM = (120, 120, 120)
COLOR_TEXT_DISABLED = (100, 100, 100)
COLOR_PANEL_BG = (30, 30, 35)
COLOR_BORDER = (80, 80, 100)
COLOR_MODAL_BG = (20, 20, 25, 220)
COLOR_INPUT_BG = (50, 50, 60)
COLOR_INPUT_ACTIVE = (70, 70, 90)


def _get_font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont('Arial', size, bold=bold)


class Button:
    """A clickable button with optional image."""

    def __init__(self, rect: pygame.Rect, label: str = '',
                 action: Any = None, image: Optional[pygame.Surface] = None,
                 color: tuple = COLOR_BTN_DEFAULT, disabled: bool = False,
                 font_size: int = 14, tooltip: str = ''):
        self.rect = rect
        self.label = label
        self.action = action
        self.image = image
        self.color = color
        self.disabled = disabled
        self.font_size = font_size
        self.tooltip = tooltip
        self._hovered = False

    def draw(self, surface: pygame.Surface):
        if self.disabled:
            bg = COLOR_BTN_DISABLED
            txt_color = COLOR_TEXT_DISABLED
        elif self._hovered:
            bg = COLOR_BTN_HOVER
            txt_color = COLOR_TEXT
        else:
            bg = self.color
            txt_color = COLOR_TEXT

        pygame.draw.rect(surface, bg, self.rect, border_radius=4)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, 1, border_radius=4)

        if self.image:
            img_rect = self.image.get_rect(center=self.rect.center)
            surface.blit(self.image, img_rect)
        if self.label:
            font = _get_font(self.font_size)
            txt_surf = font.render(self.label, True, txt_color)
            txt_rect = txt_surf.get_rect(center=self.rect.center)
            surface.blit(txt_surf, txt_rect)

    def update_hover(self, mouse_pos: tuple):
        self._hovered = self.rect.collidepoint(mouse_pos) and not self.disabled

    def is_clicked(self, pos: tuple) -> bool:
        return not self.disabled and self.rect.collidepoint(pos)


class ScrollPanel:
    """A scrollable panel that renders a list of pre-rendered surface lines."""

    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self._scroll = 0
        self._content_height = 0

    def draw(self, surface: pygame.Surface, lines: list[pygame.Surface]):
        # Background
        pygame.draw.rect(surface, COLOR_PANEL_BG, self.rect)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, 1)

        # Clip to panel
        clip = surface.subsurface(self.rect)
        y = -self._scroll
        for line_surf in lines:
            if y + line_surf.get_height() > 0 and y < self.rect.height:
                clip.blit(line_surf, (4, y))
            y += line_surf.get_height() + 2

        self._content_height = max(0, y + self._scroll)
        # Clamp scroll now that content height is known
        max_scroll = max(0, self._content_height - self.rect.height)
        self._scroll = min(self._scroll, max_scroll)

    def scroll(self, delta: int):
        max_scroll = max(0, self._content_height - self.rect.height)
        self._scroll = max(0, min(self._scroll + delta, max_scroll))

    def scroll_to_top(self):
        self._scroll = 0

    def scroll_to_bottom(self):
        self._scroll = 999_999  # clamped to actual max on next draw()


class TextInput:
    """Simple single-line text input widget."""

    def __init__(self, rect: pygame.Rect, placeholder: str = '',
                 max_len: int = 40, font_size: int = 14):
        self.rect = rect
        self.placeholder = placeholder
        self.max_len = max_len
        self.font_size = font_size
        self.text = ''
        self.active = False

    def draw(self, surface: pygame.Surface):
        bg = COLOR_INPUT_ACTIVE if self.active else COLOR_INPUT_BG
        pygame.draw.rect(surface, bg, self.rect, border_radius=3)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, 1, border_radius=3)
        font = _get_font(self.font_size)
        display = self.text if self.text else self.placeholder
        color = COLOR_TEXT if self.text else (100, 100, 100)
        txt_surf = font.render(display, True, color)
        surface.blit(txt_surf, (self.rect.x + 6, self.rect.y + (self.rect.height - txt_surf.get_height()) // 2))
        if self.active:
            cursor_x = self.rect.x + 6 + font.size(self.text)[0]
            cursor_y = self.rect.y + 4
            pygame.draw.line(surface, COLOR_TEXT,
                             (cursor_x, cursor_y),
                             (cursor_x, self.rect.bottom - 4), 1)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Returns True if input changed."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
                return True
            elif event.unicode and len(self.text) < self.max_len:
                self.text += event.unicode
                return True
        return False


class Modal:
    """Simple modal overlay with title, body widget, and OK/Cancel buttons."""

    def __init__(self, screen_size: tuple, title: str, body_widget,
                 ok_label: str = 'OK', cancel_label: str = 'Cancel'):
        w, h = 400, 200
        x = (screen_size[0] - w) // 2
        y = (screen_size[1] - h) // 2
        self.rect = pygame.Rect(x, y, w, h)
        self.title = title
        self.body_widget = body_widget
        self.error_message = ''

        btn_w, btn_h = 100, 32
        ok_x = self.rect.right - btn_w - 16
        cancel_x = ok_x - btn_w - 10
        btn_y = self.rect.bottom - btn_h - 12

        self.ok_button = Button(
            pygame.Rect(ok_x, btn_y, btn_w, btn_h),
            label=ok_label, color=COLOR_BTN_NEUTRAL)
        self.cancel_button = Button(
            pygame.Rect(cancel_x, btn_y, btn_w, btn_h),
            label=cancel_label, color=COLOR_BTN_DEFAULT)

    def draw(self, surface: pygame.Surface):
        # Dim overlay
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surface.blit(overlay, (0, 0))

        # Modal background
        pygame.draw.rect(surface, (40, 40, 50), self.rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, 1, border_radius=6)

        # Title
        font = _get_font(16, bold=True)
        title_surf = font.render(self.title, True, COLOR_TEXT)
        surface.blit(title_surf, (self.rect.x + 12, self.rect.y + 12))

        # Body widget
        if self.body_widget:
            self.body_widget.draw(surface)

        # Error
        if self.error_message:
            err_font = _get_font(12)
            err_surf = err_font.render(self.error_message, True, (220, 80, 80))
            surface.blit(err_surf, (self.rect.x + 12, self.rect.bottom - 55))

        self.ok_button.draw(surface)
        self.cancel_button.draw(surface)

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """Returns 'ok', 'cancel', or None."""
        if self.body_widget:
            self.body_widget.handle_event(event)
        mouse = pygame.mouse.get_pos()
        self.ok_button.update_hover(mouse)
        self.cancel_button.update_hover(mouse)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.ok_button.is_clicked(event.pos):
                return 'ok'
            if self.cancel_button.is_clicked(event.pos):
                return 'cancel'
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                return 'ok'
            if event.key == pygame.K_ESCAPE:
                return 'cancel'
        return None

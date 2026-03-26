"""
Reusable pygame UI primitives: Button, ScrollPanel, Modal, TextInput.
"""
from __future__ import annotations
import pygame
from typing import Optional, Any

from botbowl.gui.fonts import get_font, get_body_font
from botbowl.core.util import get_data_path

# Colors
COLOR_BTN_DEFAULT = (80, 80, 90)
COLOR_BTN_HOME = (40, 80, 160)
COLOR_BTN_AWAY = (160, 60, 40)
COLOR_BTN_NEUTRAL = (60, 110, 60)
COLOR_BTN_CONFIRM = (40, 95, 60)   # confirmatory UI actions (not game-team semantic)
COLOR_BTN_DANGER = (105, 38, 30)   # destructive UI actions (not game-team semantic)
COLOR_BTN_DISABLED = (50, 50, 50)
COLOR_BTN_HOVER = (120, 120, 140)
COLOR_TEXT = (230, 230, 230)
COLOR_TEXT_IMG = (240, 220, 170)   # warm cream/gold — readable over metal textures
COLOR_TEXT_DIM = (120, 120, 120)
COLOR_TEXT_DISABLED = (100, 100, 100)
COLOR_PANEL_BG = (30, 30, 35)
COLOR_BORDER = (80, 80, 100)
COLOR_MODAL_BG = (20, 20, 25, 220)
COLOR_INPUT_BG = (50, 50, 60)
COLOR_INPUT_ACTIVE = (70, 70, 90)

_popup_image: Optional[pygame.Surface] = None
_popup_image_loaded = False
_btn_images: dict = {}


def _get_popup_image() -> Optional[pygame.Surface]:
    """Return the popup frame image cropped to its non-transparent bounding box."""
    global _popup_image, _popup_image_loaded
    if not _popup_image_loaded:
        _popup_image_loaded = True
        try:
            path = get_data_path("img/popup.png")
            raw = pygame.image.load(path).convert_alpha()
            # Crop to bounding box of visible pixels (popup.png has large transparent margins)
            try:
                import numpy as np
                alpha = pygame.surfarray.array_alpha(raw)  # shape: (w, h)
                cols = np.any(alpha > 10, axis=1)  # visible columns (x-axis)
                rows = np.any(alpha > 10, axis=0)  # visible rows (y-axis)
                if cols.any() and rows.any():
                    x0, x1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
                    y0, y1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
                    raw = raw.subsurface(pygame.Rect(x0, y0, x1 - x0, y1 - y0)).copy()
            except Exception:
                pass  # numpy unavailable — use full image
            _popup_image = raw
        except Exception:
            _popup_image = None
    return _popup_image


def get_button_image(variant: str = 'default') -> Optional[pygame.Surface]:
    """Return a cached button background image. variant: 'default', 'green', 'red', 'orange'."""
    if variant not in _btn_images:
        filename = {
            'default': 'button.png',
            'green':   'button-green.png',
            'red':     'button-red.png',
            'orange':  'button-orange.png',
        }.get(variant, 'button.png')
        try:
            path = get_data_path(f"img/buttons/{filename}")
            _btn_images[variant] = pygame.image.load(path).convert_alpha()
        except Exception:
            _btn_images[variant] = None
    return _btn_images[variant]


def draw_text_with_shadow(surface: pygame.Surface, text: str,
                          font: pygame.font.Font, color: tuple,
                          x: int, y: int,
                          shadow_color: tuple = (0, 0, 0),
                          shadow_offset: int = 2):
    """Blit text with a drop shadow for readability on textured backgrounds."""
    surface.blit(font.render(text, True, shadow_color), (x + shadow_offset, y + shadow_offset))
    surface.blit(font.render(text, True, color), (x, y))


def draw_bordered_rect(surface: pygame.Surface, bg: tuple, border: tuple,
                       rect: pygame.Rect, radius: int = 0, border_w: int = 1):
    """Draw a filled rounded rect with a clean border using two filled draws.

    Avoids the double-line artifact that `pygame.draw.rect(..., width>0)` can
    produce when combined with border_radius.
    """
    pygame.draw.rect(surface, border, rect, border_radius=radius)
    inner = rect.inflate(-2 * border_w, -2 * border_w)
    pygame.draw.rect(surface, bg, inner, border_radius=max(0, radius - border_w))


class Button:
    """A clickable button with optional image or bg_image background."""

    def __init__(self, rect: pygame.Rect, label: str = '',
                 action: Any = None, image: Optional[pygame.Surface] = None,
                 color: tuple = COLOR_BTN_DEFAULT, disabled: bool = False,
                 font_size: int = 14, tooltip: str = '',
                 bg_image: Optional[pygame.Surface] = None):
        self.rect = rect
        self.label = label
        self.action = action
        self.image = image
        self.color = color
        self.disabled = disabled
        self.font_size = font_size
        self.tooltip = tooltip
        self.bg_image = bg_image
        self.focused = False   # keyboard focus — shows orange highlight image
        self._hovered = False
        self._scaled_bg: dict = {}  # keyed by (variant, w, h)

    def _get_scaled_bg(self, img: pygame.Surface) -> pygame.Surface:
        sz = (self.rect.width, self.rect.height)
        key = (id(img), sz[0], sz[1])
        if key not in self._scaled_bg:
            self._scaled_bg[key] = pygame.transform.smoothscale(img, sz)
        return self._scaled_bg[key]

    def draw(self, surface: pygame.Surface):
        if self.bg_image is not None:
            # Image-backed button — image never changes regardless of focus/hover
            scaled = self._get_scaled_bg(self.bg_image)
            if self.disabled:
                dim = scaled.copy()
                dim.fill((80, 80, 80, 0), special_flags=pygame.BLEND_RGBA_MULT)
                surface.blit(dim, self.rect.topleft)
            else:
                surface.blit(scaled, self.rect.topleft)
            if self.label:
                if self.disabled:
                    txt_color = COLOR_TEXT_DISABLED
                elif self._hovered or self.focused:
                    txt_color = (255, 255, 255)
                else:
                    txt_color = COLOR_TEXT_IMG
                font = get_font(self.font_size, bold=True)
                cx = self.rect.centerx
                cy = self.rect.centery
                tw, th = font.size(self.label)
                tx = cx - tw // 2
                ty = cy - th // 2
                draw_text_with_shadow(surface, self.label, font, txt_color,
                                      tx, ty, shadow_offset=2)
        else:
            # Standard colored-rect button
            if self.disabled:
                bg = COLOR_BTN_DISABLED
                txt_color = COLOR_TEXT_DISABLED
            elif self._hovered:
                bg = COLOR_BTN_HOVER
                txt_color = COLOR_TEXT
            else:
                bg = self.color
                txt_color = COLOR_TEXT

            draw_bordered_rect(surface, bg, COLOR_BORDER, self.rect, radius=4, border_w=1)

            if self.image:
                img_rect = self.image.get_rect(center=self.rect.center)
                surface.blit(self.image, img_rect)
            if self.label:
                font = get_body_font(self.font_size)
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
        font = get_body_font(self.font_size)
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
                 ok_label: str = 'OK', cancel_label: str = 'Cancel',
                 size: tuple = (360, 250), description: str = '',
                 ok_variant: str = 'default'):
        w, h = size
        x = (screen_size[0] - w) // 2
        y = (screen_size[1] - h) // 2
        self.rect = pygame.Rect(x, y, w, h)
        self.title = title
        self.description = description
        self.body_widget = body_widget
        self.error_message = ''

        # Buttons — centered horizontally, near bottom of inner frame area
        btn_w, btn_h = 130, 42
        gap = 10
        total_btn_w = btn_w * 2 + gap
        cancel_x = self.rect.centerx - total_btn_w // 2
        ok_x = cancel_x + btn_w + gap
        btn_y = self.rect.y + int(self.rect.height * 0.66)

        self.ok_button = Button(
            pygame.Rect(ok_x, btn_y, btn_w, btn_h),
            label=ok_label, font_size=14,
            bg_image=get_button_image(ok_variant))
        self.cancel_button = Button(
            pygame.Rect(cancel_x, btn_y, btn_w, btn_h),
            label=cancel_label, font_size=14,
            bg_image=get_button_image())

    def draw(self, surface: pygame.Surface):
        # Dim overlay
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surface.blit(overlay, (0, 0))

        # Modal background
        popup_img = _get_popup_image()
        if popup_img is not None:
            # Scale to fill the modal rect exactly (modal is sized to match popup aspect ratio)
            scaled = pygame.transform.smoothscale(popup_img, (self.rect.width, self.rect.height))
            surface.blit(scaled, self.rect.topleft)
        else:
            # Fallback: plain dark rect
            pygame.draw.rect(surface, (35, 32, 28), self.rect, border_radius=4)

        # Title — centered horizontally, positioned in upper third of inner frame area
        title_font = get_font(20, bold=True)
        content_top = self.rect.y + int(self.rect.height * 0.24)
        title_surf = title_font.render(self.title, True, COLOR_TEXT_IMG)
        title_shadow = title_font.render(self.title, True, (0, 0, 0))
        tx = self.rect.centerx - title_surf.get_width() // 2
        surface.blit(title_shadow, (tx + 2, content_top + 2))
        surface.blit(title_surf, (tx, content_top))

        # Description — centered
        if self.description:
            desc_font = get_body_font(15)
            desc_surf = desc_font.render(self.description, True, (215, 215, 215))
            dx = self.rect.centerx - desc_surf.get_width() // 2
            surface.blit(desc_surf, (dx, content_top + 44))

        # Body widget
        if self.body_widget:
            self.body_widget.draw(surface)

        # Error
        if self.error_message:
            err_font = get_body_font(12)
            err_surf = err_font.render(self.error_message, True, (220, 80, 80))
            err_y = self.ok_button.rect.top - 22
            surface.blit(err_surf, (self.rect.centerx - err_surf.get_width() // 2, err_y))

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

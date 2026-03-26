"""
Central font provider for the botbowl pygame GUI.

Usage:
    from botbowl.gui.fonts import get_font, get_body_font

    font = get_font(24, bold=True)       # Cinzel — for titles, buttons, headers
    font = get_body_font(12)             # Arial  — for table data, logs, stats
"""
from __future__ import annotations
import os
import pygame

from botbowl.core.util import get_data_path

_FONT_REGULAR = get_data_path("fonts/Cinzel-Regular.ttf")
_FONT_BOLD = get_data_path("fonts/Cinzel-Bold.ttf")

_cache: dict = {}
_cinzel_ok: bool | None = None


def _cinzel_available() -> bool:
    global _cinzel_ok
    if _cinzel_ok is None:
        _cinzel_ok = os.path.exists(_FONT_REGULAR) and os.path.exists(_FONT_BOLD)
    return _cinzel_ok


def get_font(size: int, bold: bool = False) -> pygame.font.Font:
    """Cinzel serif font — for titles, screen headers, large button labels."""
    key = ('fancy', size, bold)
    if key in _cache:
        return _cache[key]
    if _cinzel_available():
        path = _FONT_BOLD if bold else _FONT_REGULAR
        try:
            font = pygame.font.Font(path, size + 1)
            _cache[key] = font
            return font
        except Exception:
            pass
    font = pygame.font.SysFont('Arial', size, bold=bold)
    _cache[key] = font
    return font


def get_body_font(size: int, bold: bool = False) -> pygame.font.Font:
    """Normal sans-serif font — for table data, event log, stat values, inputs."""
    key = ('body', size, bold)
    if key in _cache:
        return _cache[key]
    font = pygame.font.SysFont('Arial', size, bold=bold)
    _cache[key] = font
    return font

"""
Event log panel rendering.
"""
from __future__ import annotations
import pygame
from botbowl.gui.assets import (get_log_text, get_log_dice_labels,
                                 get_log_roll_outcomes, get_log_entry_color)
from botbowl.gui.rendering.ui_primitives import ScrollPanel
from botbowl.gui import sprites as spr

# Colors
PANEL_BG          = (22, 22, 28)
CARD_BG           = (28, 28, 38)
CARD_BG_HOME      = (22, 32, 55)   # subtle blue tint for home-team events
CARD_BG_AWAY      = (50, 25, 25)   # subtle red tint for away-team events
CARD_BORDER       = (52, 52, 68)
COLOR_TEXT        = (190, 190, 200)
COLOR_OUTCOME_OK  = (100, 210, 110)
COLOR_OUTCOME_BAD = (210, 80, 80)
HEADER_BG         = (30, 30, 40)
HEADER_TEXT       = (160, 160, 180)

# Layout
DICE_COL_W = 90     # right column width reserved for dice + outcome labels
PAD        = 5      # card inner padding
DIE_SIZE   = 20     # px per die face
FONT_SIZE  = 11
HEADER_H   = 18
LINE_GAP   = 1      # px between text lines within a card
CARD_GAP   = 2      # px between cards in scroll panel


def _font(size: int = FONT_SIZE, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont('Arial', size, bold=bold)


def _wrap_text(font: pygame.font.Font, text: str, max_w: int) -> list[str]:
    """Word-wrap text to fit within max_w pixels. Returns list of line strings."""
    words = text.split()
    lines: list[str] = []
    current = ''
    for word in words:
        test = (current + ' ' + word).strip()
        if font.size(test)[0] <= max_w:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or ['']


class LogPanelRenderer:
    """Scrollable event log panel — terminal style (oldest top, newest bottom)."""

    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        content_rect = pygame.Rect(rect.x, rect.y + HEADER_H,
                                   rect.width, rect.height - HEADER_H)
        self.scroll_panel = ScrollPanel(content_rect)
        self._last_report_count = 0
        self._lines: list[pygame.Surface] = []

    def draw(self, surface: pygame.Surface, game):
        reports = game.state.reports
        if len(reports) != self._last_report_count:
            self._rebuild_lines(reports, game)
            self._last_report_count = len(reports)
            self.scroll_panel.scroll_to_bottom()

        self.scroll_panel.draw(surface, self._lines)

        # Header
        header_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, HEADER_H)
        pygame.draw.rect(surface, HEADER_BG, header_rect)
        hsurf = _font(FONT_SIZE, bold=True).render('Event Log', True, HEADER_TEXT)
        surface.blit(hsurf, (self.rect.x + 6, self.rect.y + 2))

    def handle_scroll(self, delta: int):
        self.scroll_panel.scroll(delta * 20)

    def _rebuild_lines(self, reports, game):
        self._lines = []
        font = _font()
        small_font = _font(FONT_SIZE - 1)
        line_h = font.get_height() + LINE_GAP
        card_w = self.rect.width - 8   # 4px margin each side in ScrollPanel
        text_w = card_w - DICE_COL_W - PAD * 3
        die_size = (DIE_SIZE, DIE_SIZE)
        home_team = game.state.home_team

        for outcome in reports:   # oldest → newest (terminal order)
            log_str = get_log_text(outcome)
            if not log_str:
                continue

            dice = get_log_dice_labels(outcome)
            roll_outcomes = get_log_roll_outcomes(outcome)
            text_color = get_log_entry_color(outcome) or COLOR_TEXT

            # Team-colored card background — prefer outcome.team, fall back to player's team
            team = outcome.team
            if team is None and outcome.player is not None:
                team = getattr(outcome.player, 'team', None)
            if team is not None:
                card_bg = CARD_BG_HOME if team is home_team else CARD_BG_AWAY
            else:
                card_bg = CARD_BG

            # Pre-compute right-column surfaces
            die_imgs = []
            for die_type, value in dice:
                if die_type == 'd6':
                    die_imgs.append(spr.get_d6_surface(value, die_size))
                elif die_type == 'd8':
                    die_imgs.append(spr.get_d8_surface(value, die_size))
                elif die_type == 'block':
                    die_imgs.append(spr.get_block_die_surface(value, die_size))

            label_surfs = []
            for label, ok in roll_outcomes:
                color = COLOR_OUTCOME_OK if ok else COLOR_OUTCOME_BAD
                label_surfs.append(small_font.render(f'({label})', True, color))

            # Right row: outcome labels | gap | dice — all on one row, right-aligned
            dice_w = len(die_imgs) * DIE_SIZE + max(0, len(die_imgs) - 1) * 2
            labels_w = sum(s.get_width() for s in label_surfs) + max(0, len(label_surfs) - 1) * 3
            gap = 4 if (die_imgs and label_surfs) else 0
            right_row_w = labels_w + gap + dice_w
            right_row_h = max(DIE_SIZE if die_imgs else 0,
                              small_font.get_height() if label_surfs else 0)

            # Card height
            wrapped = _wrap_text(font, log_str, text_w)
            text_h = len(wrapped) * line_h
            card_h = max(text_h, right_row_h) + PAD * 2

            # Build card surface
            card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            pygame.draw.rect(card_surf, card_bg, (0, 0, card_w, card_h), border_radius=3)
            pygame.draw.rect(card_surf, CARD_BORDER, (0, 0, card_w, card_h), 1, border_radius=3)

            # Text (left side)
            for i, line in enumerate(wrapped):
                t_surf = font.render(line, True, text_color)
                card_surf.blit(t_surf, (PAD, PAD + i * line_h))

            # Right row: outcome labels before dice, right-aligned, vertically centered
            if die_imgs or label_surfs:
                ry = (card_h - right_row_h) // 2
                x = card_w - PAD - right_row_w

                for lsurf in label_surfs:
                    y_off = (right_row_h - lsurf.get_height()) // 2
                    card_surf.blit(lsurf, (x, ry + y_off))
                    x += lsurf.get_width() + 3

                x += gap

                for img in die_imgs:
                    y_off = (right_row_h - DIE_SIZE) // 2
                    card_surf.blit(img, (x, ry + y_off))
                    x += DIE_SIZE + 2

            self._lines.append(card_surf)

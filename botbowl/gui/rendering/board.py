"""
Pitch board rendering: tiles, grid, highlights, paths, block dice on squares.
"""
from __future__ import annotations
import pygame
from typing import Optional

from botbowl.core.model import Tile, TwoPlayerArena, Square
from botbowl.gui.sprites import get_pitch_surface

# Tile colors
COLOR_FIELD = (90, 160, 90)
COLOR_WING = (70, 130, 70)
COLOR_SCRIMMAGE = (50, 110, 50)
COLOR_HOME_TD = (60, 80, 180)
COLOR_AWAY_TD = (180, 80, 60)
COLOR_CROWD = (40, 40, 40)

# Highlight colors (RGBA)
HL_MOVE = (0, 220, 0, 100)
HL_SELECTED = (0, 180, 255, 130)
HL_ACTIVE = (255, 220, 0, 120)
HL_BLOCK = (220, 60, 60, 100)
HL_PASS = (60, 60, 220, 100)
HL_FOUL = (180, 60, 180, 100)
HL_SETUP = (0, 200, 180, 80)

COLOR_GRID = (0, 0, 0, 60)
COLOR_GRID_DOT = (0, 0, 0, 80)
COLOR_PATH_LINE = (255, 255, 100, 180)
COLOR_PATH_DOT = (255, 255, 80)
COLOR_PATH_TEXT = (30, 30, 30)

# Action type → highlight color mapping
from botbowl.core.table import ActionType

_ACTION_HL = {
    ActionType.MOVE: HL_MOVE,
    ActionType.BLOCK: HL_BLOCK,
    ActionType.PASS: HL_PASS,
    ActionType.HANDOFF: HL_PASS,
    ActionType.FOUL: HL_FOUL,
    ActionType.PLACE_PLAYER: HL_SETUP,
    ActionType.PLACE_BALL: HL_SETUP,
    ActionType.START_MOVE: HL_MOVE,
    ActionType.START_BLOCK: HL_BLOCK,
    ActionType.START_BLITZ: HL_BLOCK,
    ActionType.START_PASS: HL_PASS,
    ActionType.START_HANDOFF: HL_PASS,
    ActionType.START_FOUL: HL_FOUL,
    ActionType.LEAP: HL_MOVE,
    ActionType.STAB: HL_BLOCK,
    ActionType.HYPNOTIC_GAZE: HL_BLOCK,
    ActionType.THROW_BOMB: HL_PASS,
    ActionType.THROW_TEAM_MATE: HL_PASS,
}


def sq_to_px(sq: Square, tile_size: int, pitch_offset: tuple) -> tuple:
    """Convert a board Square to pixel (x, y) top-left of that tile."""
    ox, oy = pitch_offset
    return (ox + sq.x * tile_size, oy + sq.y * tile_size)


def px_to_sq(pos: tuple, tile_size: int, pitch_offset: tuple,
             arena_width: int, arena_height: int) -> Optional[Square]:
    """Convert pixel position to board Square, or None if outside board."""
    ox, oy = pitch_offset
    x = (pos[0] - ox) // tile_size
    y = (pos[1] - oy) // tile_size
    if 0 <= x < arena_width and 0 <= y < arena_height:
        return Square(x, y)
    return None


def highlight_color_for_action(action_type) -> tuple:
    return _ACTION_HL.get(action_type, HL_MOVE)


class BoardRenderer:
    """Renders the pitch tiles, grid, highlights, and path visualization."""

    def __init__(self, tile_size: int, pitch_offset: tuple):
        self.tile_size = tile_size
        self.pitch_offset = pitch_offset

    def draw_board(self, surface: pygame.Surface, game, weather_name: str,
                   grid_mode: str = 'none'):
        """Draw board background tiles."""
        ts = self.tile_size
        ox, oy = self.pitch_offset
        arena = game.arena

        # Try to draw pitch image (inner area, no crowd border)
        try:
            pitch_surf = get_pitch_surface(weather_name, arena.width, arena.height, ts)
            # Pitch image covers squares [1..w-2][1..h-2] (inner, no crowd)
            surface.blit(pitch_surf, (ox + ts, oy + ts))
        except Exception:
            pass

        # Draw crowd border and endzone tiles on top (as solid colored rects)
        for y in range(arena.height):
            for x in range(arena.width):
                tile = arena.board[y][x]
                px = ox + x * ts
                py = oy + y * ts
                rect = pygame.Rect(px, py, ts, ts)

                if tile == Tile.CROWD:
                    pygame.draw.rect(surface, COLOR_CROWD, rect)
                elif tile in TwoPlayerArena.home_td_tiles:
                    col = (*COLOR_HOME_TD[:3], 140)
                    self._draw_alpha_rect(surface, col, rect)
                elif tile in TwoPlayerArena.away_td_tiles:
                    col = (*COLOR_AWAY_TD[:3], 140)
                    self._draw_alpha_rect(surface, col, rect)

        # Center line
        cx = ox + (arena.width * ts) // 2
        pygame.draw.line(surface, (0, 0, 0, 100),
                         (cx, oy), (cx, oy + arena.height * ts), 1)

        # Grid overlay
        if grid_mode == 'full':
            self._draw_full_grid(surface, arena.width, arena.height)
        elif grid_mode == 'dots':
            self._draw_dot_grid(surface, arena.width, arena.height)

    def _draw_alpha_rect(self, surface: pygame.Surface, color: tuple,
                         rect: pygame.Rect):
        tmp = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        tmp.fill(color[:4] if len(color) == 4 else color)
        surface.blit(tmp, rect.topleft)

    def _draw_full_grid(self, surface: pygame.Surface, w: int, h: int):
        ts = self.tile_size
        ox, oy = self.pitch_offset
        grid_surf = pygame.Surface((w * ts, h * ts), pygame.SRCALPHA)
        grid_surf.fill((0, 0, 0, 0))
        for x in range(w + 1):
            pygame.draw.line(grid_surf, COLOR_GRID, (x * ts, 0), (x * ts, h * ts))
        for y in range(h + 1):
            pygame.draw.line(grid_surf, COLOR_GRID, (0, y * ts), (w * ts, y * ts))
        surface.blit(grid_surf, (ox, oy))

    def _draw_dot_grid(self, surface: pygame.Surface, w: int, h: int):
        ts = self.tile_size
        ox, oy = self.pitch_offset
        for x in range(w + 1):
            for y in range(h + 1):
                pygame.draw.circle(surface, COLOR_GRID_DOT,
                                   (ox + x * ts, oy + y * ts), 1)

    @staticmethod
    def _prob_color(p: float, alpha: int = 255) -> tuple:
        """Continuous red→yellow→green color for probability p in [0, 1].

        Returns (R, G, B) when alpha=255, else (R, G, B, A).
        """
        p = max(0.0, min(1.0, p))
        if p <= 0.5:
            t = p * 2
            r, g, b = int(200 + t * 10), int(50 + t * 140), int(50 - t * 10)
        else:
            t = (p - 0.5) * 2
            r, g, b = int(210 - t * 160), int(190 + t * 10), int(40 + t * 40)
        return (r, g, b) if alpha == 255 else (r, g, b, alpha)

    def draw_highlights(self, surface: pygame.Surface,
                        squares: list, color: tuple,
                        probs: dict = None):
        """Draw semi-transparent colored overlays on squares.

        If *probs* is provided (dict[Square, float]) each square is colored
        on a continuous red→yellow→green scale by probability.
        """
        if not squares:
            return
        ts = self.tile_size
        ox, oy = self.pitch_offset
        tmp = pygame.Surface((ts, ts), pygame.SRCALPHA)
        for sq in squares:
            if probs is not None and sq in probs:
                sq_color = self._prob_color(probs[sq], alpha=110)
            else:
                sq_color = color
            tmp.fill(sq_color)
            px = ox + sq.x * ts
            py = oy + sq.y * ts
            surface.blit(tmp, (px, py))

    @staticmethod
    def _roll_color(step_rolls: list) -> tuple:
        """Return a continuous-scale RGB color for a path node's roll requirements."""
        if not step_rolls:
            return COLOR_PATH_DOT
        max_roll = max(step_rolls)
        p = (7 - max_roll) / 6
        return BoardRenderer._prob_color(p)

    def draw_path(self, surface: pygame.Surface, path, steps_taken: int = 0):
        """Draw movement path: connecting lines, per-step roll requirements, overall probability."""
        if path is None or not path.steps:
            return
        ts = self.tile_size
        ox, oy = self.pitch_offset
        small_font = pygame.font.SysFont('Arial', max(7, ts // 4), bold=True)
        steps = path.steps
        rolls = path.rolls if (path.rolls is not None) else [[] for _ in steps]

        # Draw connecting lines
        for i in range(1, len(steps)):
            prev = steps[i - 1]
            curr = steps[i]
            p1 = (ox + prev.x * ts + ts // 2, oy + prev.y * ts + ts // 2)
            p2 = (ox + curr.x * ts + ts // 2, oy + curr.y * ts + ts // 2)
            pygame.draw.line(surface, COLOR_PATH_LINE, p1, p2, 2)

        # Draw step dots — color by this step's roll difficulty
        for i, step in enumerate(steps):
            cx = ox + step.x * ts + ts // 2
            cy = oy + step.y * ts + ts // 2
            r = max(6, ts // 5)

            step_rolls = rolls[i] if i < len(rolls) else []
            dot_color = self._roll_color(step_rolls)

            pygame.draw.circle(surface, dot_color, (cx, cy), r)
            pygame.draw.circle(surface, (0, 0, 0), (cx, cy), r, 1)

            if step_rolls:
                label = f'{max(step_rolls)}+'
            else:
                label = str(i + 1)
            lbl_surf = small_font.render(label, True, COLOR_PATH_TEXT)
            surface.blit(lbl_surf, (cx - lbl_surf.get_width() // 2,
                                    cy - lbl_surf.get_height() // 2))

        # Overall probability % on the final step — larger, centered on the square
        if path.prob is not None and steps:
            final = steps[-1]
            pct = int(round(path.prob * 100))
            pct_text = f'{pct}%'
            big_font = pygame.font.SysFont('Arial', max(10, ts // 2), bold=True)
            prob_rgb = self._prob_color(path.prob)
            p_surf = big_font.render(pct_text, True, prob_rgb)
            sh_surf = big_font.render(pct_text, True, (0, 0, 0))
            fx = ox + final.x * ts + (ts - p_surf.get_width()) // 2
            fy = oy + final.y * ts + (ts - p_surf.get_height()) // 2
            surface.blit(sh_surf, (fx + 1, fy + 1))
            surface.blit(p_surf, (fx, fy))

    def draw_prob_labels(self, surface: pygame.Surface, highlight_probs: dict):
        """Draw always-visible probability % text centred on each highlighted square."""
        if not highlight_probs:
            return
        ts = self.tile_size
        ox, oy = self.pitch_offset
        font = pygame.font.SysFont('Arial', max(10, ts // 2 - 2), bold=True)
        for sq, prob in highlight_probs.items():
            pct = int(round(prob * 100))
            text = f'{pct}%'
            p_surf = font.render(text, True, (255, 255, 255))
            sh = font.render(text, True, (0, 0, 0))
            px = ox + sq.x * ts + (ts - p_surf.get_width()) // 2
            py = oy + sq.y * ts + (ts - p_surf.get_height()) // 2
            surface.blit(sh, (px + 1, py + 1))
            surface.blit(p_surf, (px, py))

    def draw_hover_highlight(self, surface: pygame.Surface, sq,
                             color: tuple = (255, 255, 255, 70)):
        """Draw a bright overlay on the hovered square."""
        if sq is None:
            return
        ts = self.tile_size
        ox, oy = self.pitch_offset
        tmp = pygame.Surface((ts, ts), pygame.SRCALPHA)
        tmp.fill(color)
        surface.blit(tmp, (ox + sq.x * ts, oy + sq.y * ts))

    def draw_block_dice_overlays(self, surface: pygame.Surface,
                                 position_dice_pairs: list):
        """Draw block dice count badges on each block target square.

        position_dice_pairs: list of (Square, int) where the int is the
        signed dice count (positive = attacker favoured, negative = defender).
        """
        for sq, dice in position_dice_pairs:
            against = dice < 0
            self.draw_block_dice_indicator(surface, sq, abs(dice), against)

    def draw_block_dice_indicator(self, surface: pygame.Surface,
                                  sq: Square, n_dice: int, against: bool):
        """Draw small colored squares indicating block dice count on a target square.

        Red squares = defender has the advantage (rolls more dice).
        Green squares = attacker has the advantage.
        """
        ts = self.tile_size
        ox, oy = self.pitch_offset
        color = (210, 50, 50) if against else (50, 200, 80)
        size = max(5, ts // 5)
        gap = 2
        total_w = n_dice * (size + gap) - gap
        # Centre horizontally, place at bottom of square
        bx = ox + sq.x * ts + (ts - total_w) // 2
        by = oy + sq.y * ts + ts - size - 3
        for i in range(n_dice):
            rect = pygame.Rect(bx + i * (size + gap), by, size, size)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, (0, 0, 0), rect, 1)

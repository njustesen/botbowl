"""
Pitch board rendering: tiles, grid, highlights, paths, block dice on squares.
"""
from __future__ import annotations
import math
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

COLOR_PATH = (0, 0, 0, 128)       # 50% black for lines and dots

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

    def draw_board(self, surface: pygame.Surface, game, weather_name: str):
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

        # Draw crowd border, bench columns, and endzone tiles on top (as solid colored rects)
        for y in range(arena.height):
            for x in range(arena.width):
                tile = arena.board[y][x]
                px = ox + x * ts
                py = oy + y * ts
                rect = pygame.Rect(px, py, ts, ts)

                if tile == Tile.CROWD or x == 0 or x == arena.width - 1:
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

    def _draw_alpha_rect(self, surface: pygame.Surface, color: tuple,
                         rect: pygame.Rect):
        tmp = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        tmp.fill(color[:4] if len(color) == 4 else color)
        surface.blit(tmp, rect.topleft)

    @staticmethod
    def _prob_color(p: float, alpha: int = 255) -> tuple:
        """Green → yellow → red → dark-red gradient for probability p in [0, 1]."""
        p = max(0.0, min(1.0, p))
        # Anchor points: (p_value, R, G, B)
        anchors = [
            (1.00,  60, 200,  60),   # green
            (0.67, 220, 210,  20),   # yellow
            (0.33, 220,  50,  30),   # red
            (0.00,  80,   0,   0),   # dark red
        ]
        # Find the two anchors that bracket p
        for i in range(len(anchors) - 1):
            p_hi, r_hi, g_hi, b_hi = anchors[i]
            p_lo, r_lo, g_lo, b_lo = anchors[i + 1]
            if p >= p_lo:
                t = (p - p_lo) / (p_hi - p_lo)
                r = int(r_lo + t * (r_hi - r_lo))
                g = int(g_lo + t * (g_hi - g_lo))
                b = int(b_lo + t * (b_hi - b_lo))
                return (r, g, b) if alpha == 255 else (r, g, b, alpha)
        r, g, b = anchors[-1][1], anchors[-1][2], anchors[-1][3]
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

    def draw_path(self, surface: pygame.Surface, path, steps_taken: int = 0,
                  player_square=None):
        """Draw movement path as 50% black lines and dots — no outlines or effects."""
        if path is None or not path.steps:
            return
        ts = self.tile_size
        ox, oy = self.pitch_offset
        steps = path.steps

        def cp(sq):
            return (ox + sq.x * ts + ts // 2, oy + sq.y * ts + ts // 2)

        line_w = max(2, ts // 10)
        dot_r  = max(5, ts // 6)

        # Draw onto SRCALPHA surface so everything is truly 50% transparent black
        path_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

        if player_square is not None:
            pygame.draw.line(path_surf, COLOR_PATH, cp(player_square), cp(steps[0]), line_w)

        for i in range(1, len(steps)):
            pygame.draw.line(path_surf, COLOR_PATH, cp(steps[i - 1]), cp(steps[i]), line_w)

        for step in steps:
            pygame.draw.circle(path_surf, COLOR_PATH, cp(step), dot_r)

        surface.blit(path_surf, (0, 0))

    def draw_path_prob(self, surface: pygame.Surface, path, cursor_pos: tuple):
        """Draw the path probability % just above the mouse cursor, on top of everything."""
        if path is None or path.prob is None:
            return
        ts = self.tile_size
        pct_text = f'{int(round(path.prob * 100))}%'
        font_size = max(13, ts // 2 + 2)
        font = pygame.font.SysFont('Arial', font_size, bold=True)
        mx, my = cursor_pos
        p_surf = font.render(pct_text, True, (255, 255, 255))
        sh_surf = font.render(pct_text, True, (0, 0, 0))
        fx = mx - p_surf.get_width() // 2
        fy = my - p_surf.get_height() - 6
        surface.blit(sh_surf, (fx + 1, fy + 1))
        surface.blit(p_surf, (fx, fy))

    def draw_pass_prob(self, surface: pygame.Surface, rolls: list, cursor_pos: tuple):
        """Draw pass probability % just above the mouse cursor — same style as draw_path_prob."""
        if not rolls:
            return
        ts = self.tile_size
        p = 1.0
        for r in rolls:
            p *= (7 - r) / 6
        pct_text = f'{int(round(p * 100))}%'
        font_size = max(13, ts // 2 + 2)
        font = pygame.font.SysFont('Arial', font_size, bold=True)
        mx, my = cursor_pos
        p_surf = font.render(pct_text, True, (255, 255, 255))
        sh_surf = font.render(pct_text, True, (0, 0, 0))
        fx = mx - p_surf.get_width() // 2
        fy = my - p_surf.get_height() - 6
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

    def draw_roll_labels(self, surface: pygame.Surface, highlight_rolls: dict):
        """Draw dice roll requirement chips (e.g. '3+') centered on each highlighted square.

        Layout: 1 roll → centered; 2 rolls → side-by-side, row vertically centered;
        3–4 rolls → 2 per row, group vertically centered.
        """
        if not highlight_rolls:
            return
        ts = self.tile_size
        ox, oy = self.pitch_offset
        font = pygame.font.SysFont('Arial', max(8, ts // 4), bold=True)
        pad = max(2, ts // 12)
        gap = max(1, ts // 20)

        for sq, rolls in highlight_rolls.items():
            if not rolls:
                continue
            surfs = [font.render(f'{r}+', True, (255, 255, 255)) for r in rolls]
            chip_dims = [(s.get_width() + pad * 2, s.get_height() + pad) for s in surfs]

            # Group into rows of 2
            rows = [chip_dims[i:i + 2] for i in range(0, len(chip_dims), 2)]
            row_widths = [sum(c[0] for c in row) + gap * (len(row) - 1) for row in rows]
            row_heights = [max(c[1] for c in row) for row in rows]
            total_h = sum(row_heights) + gap * (len(rows) - 1)

            sq_cx = ox + sq.x * ts + ts // 2
            sq_cy = oy + sq.y * ts + ts // 2
            cy = sq_cy - total_h // 2

            chip_idx = 0
            for r_i, row in enumerate(rows):
                cx = sq_cx - row_widths[r_i] // 2
                for cw, ch in row:
                    badge = pygame.Surface((cw, ch), pygame.SRCALPHA)
                    badge.fill((0, 0, 0, 140))
                    surface.blit(badge, (cx, cy))
                    surface.blit(surfs[chip_idx], (cx + pad, cy + pad // 2))
                    cx += cw + gap
                    chip_idx += 1
                cy += row_heights[r_i] + gap

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

    def draw_pass_arrow(self, surface: pygame.Surface, p1: tuple, p2: tuple,
                        prob: float, half_width: int = 10,
                        pass_label: str = None, rolls: list = None):
        """Draw a probability-colored arrow (shaft + arrowhead) from p1 → p2.

        p1, p2      : pixel (x, y) centre-points (passer → target)
        prob        : success probability in [0, 1], controls fill color
        half_width  : shaft half-width in pixels (fixed, not tile-relative)
        pass_label  : abbreviated pass type label ('QP', 'SP', 'LP', 'LB', 'HM')
        rolls       : pass rolls list; rolls[0] used for roll-target text (e.g. '3+')
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 4:
            return

        ux, uy   = dx / dist, dy / dist   # unit vector along arrow
        perp_x, perp_y = -uy, ux          # perpendicular unit vector

        # Inset start and end by 1/3 of a tile so the arrow doesn't overlap player sprites
        margin = self.tile_size / 3
        ax = p1[0] + ux * margin   # effective start
        ay = p1[1] + uy * margin
        bx = p2[0] - ux * margin   # effective end
        by = p2[1] - uy * margin

        # Recalculate effective distance; bail if too short after inset
        eff_dist = math.hypot(bx - ax, by - ay)
        if eff_dist < 4:
            return

        fill = self._prob_color(prob, alpha=160)

        tmp = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

        # Arrowhead dimensions (fixed pixels, independent of tile size)
        ah_len = half_width * 3    # arrowhead length
        ah_hw  = half_width * 2    # arrowhead half-width at base

        # Shaft ends where arrowhead begins
        sx = bx - ux * ah_len
        sy = by - uy * ah_len

        # Shaft rectangle: effective start → shaft-end
        shaft_len = math.hypot(sx - ax, sy - ay)
        if shaft_len > 1:
            shaft = [
                (ax + perp_x * half_width, ay + perp_y * half_width),
                (sx + perp_x * half_width, sy + perp_y * half_width),
                (sx - perp_x * half_width, sy - perp_y * half_width),
                (ax - perp_x * half_width, ay - perp_y * half_width),
            ]
            pygame.draw.polygon(tmp, fill, shaft)

        # Arrowhead triangle: tip at effective end
        head = [
            (bx, by),
            (sx + perp_x * ah_hw, sy + perp_y * ah_hw),
            (sx - perp_x * ah_hw, sy - perp_y * ah_hw),
        ]
        pygame.draw.polygon(tmp, fill, head)

        surface.blit(tmp, (0, 0))

        # Text label centered on the shaft, rotated to match the arrow angle
        if pass_label or rolls:
            parts = []
            if pass_label:
                parts.append(pass_label)
            if rolls:
                parts.append(f'{rolls[0]}+')
            label = '  '.join(parts)

            font_size = max(10, int(self.tile_size * 0.55))
            font = pygame.font.SysFont('Arial', font_size, bold=True)
            text_surf = font.render(label, True, (255, 255, 255))
            sh_surf   = font.render(label, True, (0, 0, 0))

            # Angle in pygame coords (y-axis flipped → negate dy for standard math angle)
            angle_deg = math.degrees(math.atan2(-dy, dx))
            if angle_deg < -90 or angle_deg > 90:   # keep text readable (not upside-down)
                angle_deg += 180

            rot_text = pygame.transform.rotate(text_surf, angle_deg)
            rot_sh   = pygame.transform.rotate(sh_surf,   angle_deg)

            # Centre on shaft midpoint (midpoint between effective start and shaft-end)
            cx = int((ax + sx) / 2)
            cy = int((ay + sy) / 2)
            fx = cx - rot_text.get_width()  // 2
            fy = cy - rot_text.get_height() // 2
            surface.blit(rot_sh,   (fx + 1, fy + 1))
            surface.blit(rot_text, (fx, fy))

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

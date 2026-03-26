"""
Shared team display helpers — reused by CreateGameScreen and TeamsScreen.
"""
from __future__ import annotations
import os
import pygame
from typing import Optional

from botbowl.gui.rendering.ui_primitives import draw_bordered_rect
from botbowl.gui import sprites as spr

# ── BB2016 roster-building constants ─────────────────────────────────────────
TEAM_BUDGET     = 1_000_000
APOTHECARY_COST = 50_000
STAFF_COSTS = {
    'ass_coaches':  10_000,
    'cheerleaders': 10_000,
    'fan_factor':   10_000,
}
MIN_PLAYERS = {1: 1, 3: 3, 5: 5, 7: 7, 11: 11}
MAX_PLAYERS  = 16

# ── Logo loader ───────────────────────────────────────────────────────────────
_LOGO_CACHE: dict = {}
_IMG_BASE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'img', 'teamlogos'))


def load_team_logo(race: str, size: int = 54) -> Optional[pygame.Surface]:
    """Load a team race logo scaled to *size* × *size* pixels."""
    key = (race.lower(), size)
    if key in _LOGO_CACHE:
        return _LOGO_CACHE[key]
    slug = race.lower()
    candidates = [
        slug + '.png',
        slug + 's.png',                      # e.g. goblin → goblins
        slug.replace(' ', '_') + '.png',
        slug.replace(' ', '-') + '.png',
        slug.replace(' ', '') + '.png',
    ]
    path = next((os.path.join(_IMG_BASE, c) for c in candidates
                 if os.path.exists(os.path.join(_IMG_BASE, c))), None)
    if path is None:
        _LOGO_CACHE[key] = None
        return None
    try:
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.smoothscale(img, (size, size))
        _LOGO_CACHE[key] = img
        return img
    except Exception:
        _LOGO_CACHE[key] = None
        return None


# ── TV calculation ────────────────────────────────────────────────────────────
def calc_team_tv(team, reroll_costs: dict) -> int:
    """Compute total team value (TV) from players, rerolls, and apothecaries."""
    player_tv = sum(getattr(p.role, 'cost', 0) for p in team.players if p.role)
    reroll_cost = reroll_costs.get(team.race, 0)
    return player_tv + team.rerolls * reroll_cost + team.apothecaries * 50_000


# ── Font helper ───────────────────────────────────────────────────────────────
from botbowl.gui.fonts import get_body_font as _font


# ── Roster table ──────────────────────────────────────────────────────────────
def draw_roster_table(surface: pygame.Surface, team, x: int, y: int,
                      x_max: int, y_max: int, is_home: bool, *,
                      remove_btns: Optional[list] = None,
                      show_skills: bool = True,
                      skip_header: bool = False,
                      out_row_ys: Optional[list] = None) -> int:
    """
    Draw a player roster table at position (x, y).

    Args:
        remove_btns: If a list is provided, a [×] button is drawn for each
                     player row and its pygame.Rect is appended to the list
                     (in the same order as team.players).
        show_skills: If False, omit the skills column (useful for narrow layouts).
    Returns:
        The y coordinate immediately after the last drawn row.
    """
    BTN_COL_W = 24 if remove_btns is not None else 0
    eff_x_max = x_max - BTN_COL_W

    font_hdr = _font(11, bold=True)
    font_row = _font(12)
    hdr_col  = (100, 105, 130)
    row_col  = (195, 195, 210)
    tv_col   = (170, 155, 90)
    alt_bg   = (32, 32, 38)
    ROW_H    = 25

    total_w = eff_x_max - x
    ICON_W  = ROW_H

    # Column layout (right-to-left)
    # Fixed right section: TV + [skills +] stats
    STAT_W    = 24
    STAT_GAP  = 8
    TV_W      = 34
    stats_w   = 4 * STAT_W + STAT_GAP   # 104

    COL_TV_R  = total_w - 2
    if show_skills:
        # Adaptive skills width so name/position columns never collide
        fixed_total = ICON_W + 24 + 4 + TV_W + stats_w  # icon+nr+gap + TV+stats = 191
        avail       = max(0, total_w - fixed_total)
        skill_col_w = max(60, min(250, int(avail * 0.50)))
        pos_w       = max(60, min(140, int(avail * 0.30)))
    else:
        skill_col_w = 0
        fixed_total = ICON_W + 24 + 4 + TV_W + stats_w
        avail       = max(0, total_w - fixed_total)
        pos_w       = max(60, min(160, int(avail * 0.40)))

    COL_SKILL_R = COL_TV_R - TV_W + 2
    COL_SKILL_L = COL_SKILL_R - skill_col_w if show_skills else COL_SKILL_R
    COL_AV_R    = COL_SKILL_L - (STAT_GAP if show_skills else 0)
    COL_AG_R    = COL_AV_R - STAT_W
    COL_ST_R    = COL_AG_R - STAT_W
    COL_MA_R    = COL_ST_R - STAT_W
    COL_MA_C    = COL_MA_R - STAT_W // 2
    COL_ST_C    = COL_ST_R - STAT_W // 2
    COL_AG_C    = COL_AG_R - STAT_W // 2
    COL_AV_C    = COL_AV_R - STAT_W // 2
    COL_POS_R   = COL_MA_R - STAT_W // 2
    COL_POS_L   = COL_POS_R - pos_w
    COL_NR_R    = ICON_W + 20
    COL_NAME    = COL_NR_R + 4
    name_max_w  = max(20, COL_POS_L - COL_NAME - 8)

    text_oy = (ROW_H - font_row.get_height()) // 2
    hdr_oy  = (ROW_H - font_hdr.get_height()) // 2

    def blit_right(surf, col_right_rel, row_y, oy=None):
        surface.blit(surf, (x + col_right_rel - surf.get_width(),
                            row_y + (text_oy if oy is None else oy)))

    def blit_center(surf, col_center_x, row_y, oy=None):
        surface.blit(surf, (x + col_center_x - surf.get_width() // 2,
                            row_y + (text_oy if oy is None else oy)))

    def truncate(text, font, max_w):
        if font.size(text)[0] <= max_w:
            return text
        while text and font.size(text + '…')[0] > max_w:
            text = text[:-1]
        return text + '…'

    def skill_lines(skills):
        names = [s.name.replace('_', ' ').title() for s in skills]
        lines, current = [], ''
        for sk in names:
            test = current + (', ' if current else '') + sk
            if font_row.size(test)[0] <= skill_col_w:
                current = test
            else:
                if current:
                    lines.append(current)
                current = sk
        if current:
            lines.append(current)
        return lines

    # Header row
    if not skip_header:
        blit_right( font_hdr.render('#',         True, hdr_col), COL_NR_R,    y, oy=hdr_oy)
        surface.blit(font_hdr.render('Name',     True, hdr_col), (x + COL_NAME,    y + hdr_oy))
        surface.blit(font_hdr.render('Position', True, hdr_col), (x + COL_POS_L,   y + hdr_oy))
        blit_center(font_hdr.render('MA',        True, hdr_col), COL_MA_C,    y, oy=hdr_oy)
        blit_center(font_hdr.render('ST',        True, hdr_col), COL_ST_C,    y, oy=hdr_oy)
        blit_center(font_hdr.render('AG',        True, hdr_col), COL_AG_C,    y, oy=hdr_oy)
        blit_center(font_hdr.render('AV',        True, hdr_col), COL_AV_C,    y, oy=hdr_oy)
        if show_skills:
            surface.blit(font_hdr.render('Skills', True, hdr_col), (x + COL_SKILL_L, y + hdr_oy))
        blit_right( font_hdr.render('TV',        True, hdr_col), COL_TV_R,    y, oy=hdr_oy)
        y += ROW_H + 4

    race   = team.race if hasattr(team, 'race') else None
    roster = team.players if hasattr(team, 'players') else []

    for idx, player in enumerate(roster):
        if out_row_ys is not None:
            out_row_ys.append(y)
        role = player.role
        base_skills  = (role.skills if role and role.skills else []) if show_skills else []
        extra_skills = (getattr(player, 'extra_skills', None) or []) if show_skills else []
        s_lines = ([(ln, (210, 215, 240)) for ln in skill_lines(base_skills)] +
                   [(ln, (100, 210, 100)) for ln in skill_lines(extra_skills)])
        row_lines   = max(1, len(s_lines))
        row_h_total = row_lines * ROW_H

        if y + row_h_total > y_max:
            break

        row_rect = pygame.Rect(x - 2, y - 1, total_w + 2, row_h_total)
        if idx % 2 == 1:
            pygame.draw.rect(surface, alt_bg, row_rect, border_radius=2)

        # Player icon — try sprite, fall back to team logo for unsupported races
        icon = None
        _player_race = (player.team.race
                        if (hasattr(player, 'team') and hasattr(player.team, 'race'))
                        else None) or race
        _role_name = role.name if role else ''
        from botbowl.gui.sprites import player_icons as _pi
        _has_sprite = (_player_race in _pi and _role_name in _pi[_player_race])
        if _has_sprite:
            icon = spr.get_player_surface(player, is_home, False)
        if icon is None and _player_race:
            icon = load_team_logo(_player_race, size=ICON_W)
        if icon:
            scaled = pygame.transform.smoothscale(icon, (ICON_W, ICON_W))
            surface.blit(scaled, (x, y))

        blit_right(font_row.render(str(player.nr), True, row_col), COL_NR_R, y)
        name = truncate(player.name or '', font_row, name_max_w)
        surface.blit(font_row.render(name, True, row_col), (x + COL_NAME, y + text_oy))

        if role:
            pos_text = truncate(getattr(role, 'name', '') or '', font_row, 120)
            surface.blit(font_row.render(pos_text, True, (155, 155, 180)),
                         (x + COL_POS_L, y + text_oy))
            for base, extra, col_c in (
                (role.ma, getattr(player, 'extra_ma', 0), COL_MA_C),
                (role.st, getattr(player, 'extra_st', 0), COL_ST_C),
                (role.ag, getattr(player, 'extra_ag', 0), COL_AG_C),
                (role.av, getattr(player, 'extra_av', 0), COL_AV_C),
            ):
                sc = (100, 210, 100) if extra > 0 else (210, 80, 80) if extra < 0 else row_col
                blit_center(font_row.render(str(base + extra), True, sc), col_c, y)
            cost = getattr(role, 'cost', 0) // 1000
            blit_right(font_row.render(f'{cost}k', True, tv_col), COL_TV_R, y)

        if show_skills:
            for i, (line, line_col) in enumerate(s_lines):
                sy = y + i * ROW_H
                surface.blit(font_row.render(line, True, line_col),
                             (x + COL_SKILL_L, sy + text_oy))

        # Optional remove button
        if remove_btns is not None:
            btn_rect = pygame.Rect(eff_x_max + 2, y + 2, BTN_COL_W - 4, ROW_H - 4)
            pygame.draw.rect(surface, (90, 35, 35), btn_rect, border_radius=3)
            x_surf = _font(11, bold=True).render('×', True, (220, 100, 100))
            surface.blit(x_surf, (btn_rect.x + (btn_rect.width  - x_surf.get_width())  // 2,
                                   btn_rect.y + (btn_rect.height - x_surf.get_height()) // 2))
            remove_btns.append(btn_rect)

        y += row_h_total

    return y

"""
HUD rendering: scoreboard, rerolls, resources, turn markers, clock, status.
"""
from __future__ import annotations
import pygame
from typing import Optional

from botbowl.gui import sprites as spr
from botbowl.gui.assets import get_procedure_label, get_turn_label, prettify
from botbowl.core.table import Skill, SkillCategory

COLOR_BG = (20, 20, 25)
COLOR_HOME = (50, 100, 200)
COLOR_AWAY = (200, 80, 50)
COLOR_NEUTRAL = (80, 80, 80)
COLOR_TEXT = (220, 220, 220)
COLOR_TEXT_DIM = (120, 120, 120)
COLOR_REROLL_USED = (80, 80, 90)
COLOR_REROLL_AVAIL = (80, 160, 200)
COLOR_REROLL_BORDER = (60, 60, 70)
COLOR_TURN_MARKER = (60, 100, 180)
COLOR_TURN_MARKER_INACTIVE = (40, 50, 60)
COLOR_CLOCK_NORMAL = (180, 180, 200)
COLOR_CLOCK_BUZZER = (220, 60, 60)

# Skill chip colors: (bg_color, text_color)
_CHIP = {
    SkillCategory.General:       ((50, 100, 200),  (255, 255, 255)),
    SkillCategory.Agility:       ((200, 180, 30),  (20, 20, 20)),
    SkillCategory.Strength:      ((180, 40, 40),   (255, 255, 255)),
    SkillCategory.Passing:       ((220, 220, 220), (20, 20, 20)),
    SkillCategory.Mutation:      ((120, 50, 180),  (255, 255, 255)),
    SkillCategory.Extraordinary: ((40, 40, 40),    (200, 80, 80)),
}
_INJURY_CHIP = ((20, 20, 20), (200, 60, 60))   # black bg, red text

# Manual skill → category mapping (BB2016)
_SKILL_CAT = {
    # General
    Skill.BLOCK: SkillCategory.General,
    Skill.WRESTLE: SkillCategory.General,
    Skill.GUARD: SkillCategory.General,
    Skill.DAUNTLESS: SkillCategory.General,
    Skill.DIRTY_PLAYER: SkillCategory.General,
    Skill.FEND: SkillCategory.General,
    Skill.FRENZY: SkillCategory.General,
    Skill.KICK: SkillCategory.General,
    Skill.PRO: SkillCategory.General,
    Skill.STAND_FIRM: SkillCategory.General,
    Skill.STRIP_BALL: SkillCategory.General,
    Skill.SURE_HANDS: SkillCategory.General,
    Skill.TACKLE: SkillCategory.General,
    Skill.PASS_BLOCK: SkillCategory.General,
    Skill.KICK_OFF_RETURN: SkillCategory.General,
    Skill.DUMP_OFF: SkillCategory.General,
    Skill.MULTIPLE_BLOCK: SkillCategory.General,
    Skill.PILING_ON: SkillCategory.General,
    Skill.JUMP_UP: SkillCategory.General,
    Skill.SHADOWING: SkillCategory.General,
    Skill.DIVING_TACKLE: SkillCategory.General,
    Skill.FAN_FAVOURITE: SkillCategory.General,
    Skill.SWIFT_REACTION: SkillCategory.General,
    Skill.TIMMMBER: SkillCategory.General,
    # Agility
    Skill.CATCH: SkillCategory.Agility,
    Skill.DODGE: SkillCategory.Agility,
    Skill.LEAP: SkillCategory.Agility,
    Skill.SIDE_STEP: SkillCategory.Agility,
    Skill.SPRINT: SkillCategory.Agility,
    Skill.SURE_FEET: SkillCategory.Agility,
    Skill.DIVING_CATCH: SkillCategory.Agility,
    Skill.SNEAKY_GIT: SkillCategory.Agility,
    Skill.NERVES_OF_STEEL: SkillCategory.Agility,
    Skill.LONG_LEGS: SkillCategory.Agility,
    # Strength
    Skill.BREAK_TACKLE: SkillCategory.Strength,
    Skill.CLAWS: SkillCategory.Strength,
    Skill.GRAB: SkillCategory.Strength,
    Skill.JUGGERNAUT: SkillCategory.Strength,
    Skill.MIGHTY_BLOW: SkillCategory.Strength,
    Skill.STRONG_ARM: SkillCategory.Strength,
    Skill.THICK_SKULL: SkillCategory.Strength,
    # Passing
    Skill.ACCURATE: SkillCategory.Passing,
    Skill.HAIL_MARY_PASS: SkillCategory.Passing,
    Skill.PASS: SkillCategory.Passing,
    Skill.SAFE_THROW: SkillCategory.Passing,
    # Mutation
    Skill.BIG_HAND: SkillCategory.Mutation,
    Skill.DISTURBING_PRESENCE: SkillCategory.Mutation,
    Skill.EXTRA_ARMS: SkillCategory.Mutation,
    Skill.FOUL_APPEARANCE: SkillCategory.Mutation,
    Skill.HORNS: SkillCategory.Mutation,
    Skill.MONSTROUS_MOUTH: SkillCategory.Mutation,
    Skill.PREHENSILE_TAIL: SkillCategory.Mutation,
    Skill.TENTACLES: SkillCategory.Mutation,
    Skill.TWO_HEADS: SkillCategory.Mutation,
    Skill.VERY_LONG_LEGS: SkillCategory.Mutation,
    # Extraordinary
    Skill.ALWAYS_HUNGRY: SkillCategory.Extraordinary,
    Skill.ANIMOSITY: SkillCategory.Extraordinary,
    Skill.BALL_AND_CHAIN: SkillCategory.Extraordinary,
    Skill.BLOOD_LUST: SkillCategory.Extraordinary,
    Skill.BOMBARDIER: SkillCategory.Extraordinary,
    Skill.BONE_HEAD: SkillCategory.Extraordinary,
    Skill.CHAINSAW: SkillCategory.Extraordinary,
    Skill.DECAY: SkillCategory.Extraordinary,
    Skill.HYPNOTIC_GAZE: SkillCategory.Extraordinary,
    Skill.LONER: SkillCategory.Extraordinary,
    Skill.NO_HANDS: SkillCategory.Extraordinary,
    Skill.NURGLES_ROT: SkillCategory.Extraordinary,
    Skill.REALLY_STUPID: SkillCategory.Extraordinary,
    Skill.REGENERATION: SkillCategory.Extraordinary,
    Skill.RIGHT_STUFF: SkillCategory.Extraordinary,
    Skill.SECRET_WEAPON: SkillCategory.Extraordinary,
    Skill.STAB: SkillCategory.Extraordinary,
    Skill.STAKES: SkillCategory.Extraordinary,
    Skill.STUNTY: SkillCategory.Extraordinary,
    Skill.SWOOP: SkillCategory.Extraordinary,
    Skill.TAKE_ROOT: SkillCategory.Extraordinary,
    Skill.THROW_TEAM_MATE: SkillCategory.Extraordinary,
    Skill.TITCHY: SkillCategory.Extraordinary,
    Skill.WILD_ANIMAL: SkillCategory.Extraordinary,
}


def _draw_chip(surface: pygame.Surface, text: str, x: int, y: int,
               font: pygame.font.Font, bg: tuple, fg: tuple,
               pad_x: int = 4, pad_y: int = 2, radius: int = 3) -> int:
    """Draw a rounded-rect chip with text. Returns chip width."""
    ts = font.render(text, True, fg)
    w = ts.get_width() + pad_x * 2
    h = ts.get_height() + pad_y * 2
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surface, bg, rect, border_radius=radius)
    surface.blit(ts, (x + pad_x, y + pad_y))
    return w


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont('Arial', size, bold=bold)


class HUDRenderer:
    """Renders the scoreboard bar and all HUD elements."""

    def __init__(self, rect: pygame.Rect):
        """rect: the full scoreboard bar area."""
        self.rect = rect

    def draw(self, surface: pygame.Surface, game):
        pygame.draw.rect(surface, COLOR_BG, self.rect)
        pygame.draw.rect(surface, (40, 40, 50), self.rect, 1)

        w = self.rect.width
        cx = self.rect.x + w // 2
        home = game.state.home_team
        away = game.state.away_team
        rounds = game.config.rounds if hasattr(game.config, 'rounds') else 8

        gap = 1
        center_w = 110
        team_info_w = 140       # reserved on each side for logo + name
        rr_box_size = 12        # reroll indicator box size
        rr_box_gap = 2          # gap between reroll boxes
        boxes_per_side = rounds + 1  # N turn boxes + 1 score box
        center_h = self.rect.height - 4   # center box uses full bar height
        # Turn boxes capped at 30px so resource rows always fit below the clock bar
        max_box_h = min(center_h, 30)
        center_y = self.rect.y + 2
        avail = (w - 2 * team_info_w - center_w - 4) // 2
        box_w = max(18, (avail - (boxes_per_side - 1) * gap) // boxes_per_side)
        box_side = min(box_w, max_box_h)  # square: constrained by whichever is smaller
        my = center_y  # turn boxes top-aligned with center box
        nr_font = _font(max(11, box_side - 10), bold=True)
        sc_font = _font(min(box_side - 6, 22), bold=True)

        # ── Away side: [1][2]...[N] [score] ──────────────────────────────
        away_total_w = boxes_per_side * box_side + (boxes_per_side - 1) * gap
        away_x = cx - center_w // 2 - gap - away_total_w
        away_turn = away.state.turn
        away_active = (game.state.current_team == away)

        for i in range(1, rounds + 1):
            bx = away_x + (i - 1) * (box_side + gap)
            is_current = away_active and (i == away_turn)
            is_past = (i < away_turn)
            bg = COLOR_AWAY if is_current else ((55, 28, 18) if is_past else (32, 32, 42))
            rect = pygame.Rect(bx, my, box_side, box_side)
            pygame.draw.rect(surface, bg, rect)
            pygame.draw.rect(surface, (70, 70, 85), rect, 1)
            fg = (255, 255, 255) if is_current else ((140, 80, 60) if is_past else (110, 110, 120))
            nr_s = nr_font.render(str(i), True, fg)
            surface.blit(nr_s, (bx + (box_side - nr_s.get_width()) // 2,
                                my + (box_side - nr_s.get_height()) // 2))

        sc_bx = away_x + rounds * (box_side + gap)
        sc_rect = pygame.Rect(sc_bx, center_y, box_side, center_h)
        pygame.draw.rect(surface, COLOR_AWAY, sc_rect)
        sc_s = sc_font.render(str(away.state.score), True, (255, 255, 255))
        surface.blit(sc_s, (sc_bx + (box_side - sc_s.get_width()) // 2,
                            center_y + (center_h - sc_s.get_height()) // 2))

        # ── Home side: [score] [1][2]...[N] ──────────────────────────────
        home_x = cx + center_w // 2 + gap
        home_turn = home.state.turn
        home_active = (game.state.current_team == home)

        sc_rect2 = pygame.Rect(home_x, center_y, box_side, center_h)
        pygame.draw.rect(surface, COLOR_HOME, sc_rect2)
        sc_s2 = sc_font.render(str(home.state.score), True, (255, 255, 255))
        surface.blit(sc_s2, (home_x + (box_side - sc_s2.get_width()) // 2,
                             center_y + (center_h - sc_s2.get_height()) // 2))

        for i in range(1, rounds + 1):
            bx = home_x + i * (box_side + gap)
            is_current = home_active and (i == home_turn)
            is_past = (i < home_turn)
            bg = COLOR_HOME if is_current else ((18, 28, 55) if is_past else (32, 32, 42))
            rect = pygame.Rect(bx, my, box_side, box_side)
            pygame.draw.rect(surface, bg, rect)
            pygame.draw.rect(surface, (70, 70, 85), rect, 1)
            fg = (255, 255, 255) if is_current else ((60, 80, 140) if is_past else (110, 110, 120))
            nr_s = nr_font.render(str(i), True, fg)
            surface.blit(nr_s, (bx + (box_side - nr_s.get_width()) // 2,
                                my + (box_side - nr_s.get_height()) // 2))

        # ── Center: weather icon + phase label ───────────────────────────
        center_rect = pygame.Rect(cx - center_w // 2, center_y, center_w, center_h)
        pygame.draw.rect(surface, (28, 30, 36), center_rect, border_radius=3)
        pygame.draw.rect(surface, (65, 65, 78), center_rect, 1, border_radius=3)

        weather_name = game.state.weather.name if game.state.weather else 'NICE'
        icon_size = (center_h - 42, center_h - 42)
        weather_surf = spr.get_weather_icon(weather_name, icon_size)
        proc_str = 'Game Over' if game.state.game_over else get_procedure_label(game)
        proc_s = _font(14, bold=True).render(proc_str.upper(), True, (180, 180, 200))
        total_center_h = proc_s.get_height() + 2 + icon_size[1]
        cy_off = center_y + (center_h - total_center_h) // 2
        surface.blit(proc_s, (cx - proc_s.get_width() // 2, cy_off))
        surface.blit(weather_surf, (cx - icon_size[0] // 2,
                                    cy_off + proc_s.get_height() + 2))

        # ── Team clocks: bar below turn boxes + MM:SS below score box ─────
        clock_font = _font(9, bold=True)
        bar_h = 4
        bar_y = my + box_side + 2

        for team, is_away in ((away, True), (home, False)):
            clock = next((c for c in game.state.clocks if c.team is team), None)
            if clock is None:
                continue
            seconds_left = clock.get_seconds_left()
            ratio_left = max(0.0, min(1.0, 1.0 - clock.get_ratio_done()))
            mm, ss = int(seconds_left) // 60, int(seconds_left) % 60
            clock_str = f'{mm:02d}:{ss:02d}'
            color = COLOR_CLOCK_BUZZER if seconds_left < 10 else (
                COLOR_AWAY if is_away else COLOR_HOME)

            if is_away:
                bar_x = away_x
                bar_w = rounds * (box_side + gap) - gap
                sc_x = sc_bx
            else:
                bar_x = home_x + box_side + gap
                bar_w = rounds * (box_side + gap) - gap
                sc_x = home_x

            # Background bar
            pygame.draw.rect(surface, (40, 40, 50),
                             pygame.Rect(bar_x, bar_y, bar_w, bar_h), border_radius=2)
            # Filled portion (remaining time, both bars deplete inward toward score box)
            fill_w = int(bar_w * ratio_left)
            if fill_w > 0:
                if is_away:
                    pygame.draw.rect(surface, color,
                                     pygame.Rect(bar_x + bar_w - fill_w, bar_y,
                                                 fill_w, bar_h), border_radius=2)
                else:
                    pygame.draw.rect(surface, color,
                                     pygame.Rect(bar_x, bar_y, fill_w, bar_h), border_radius=2)

            # Clock text below score box
            clk_s = clock_font.render(clock_str, True, color)
            surface.blit(clk_s, (sc_x + (box_side - clk_s.get_width()) // 2, bar_y))

        # ── Rerolls + resources: two rows per team ───────────────────────
        rr_margin = 8
        GAP = 3   # gap between items and between rows
        rr_row_y = bar_y + bar_h + 4
        res_row_y = rr_row_y + rr_box_size + GAP

        # Away: both rows right-aligned before score box
        away_anchor = sc_bx - rr_margin
        self._draw_reroll_boxes(surface, away, away_anchor, rr_row_y,
                                rr_box_size, GAP, is_home=False, game=game,
                                align_right=True)
        self._draw_resource_row(surface, away, away_anchor, res_row_y,
                                GAP, align_right=True, flip=False)

        # Home: both rows left-aligned after score box
        home_anchor = home_x + box_side + rr_margin
        self._draw_reroll_boxes(surface, home, home_anchor, rr_row_y,
                                rr_box_size, GAP, is_home=True, game=game,
                                align_right=False, flip=True)
        self._draw_resource_row(surface, home, home_anchor, res_row_y,
                                GAP, align_right=False, flip=True)

        # ── Team info: icon + agent name (row 1) + team name (row 2) ──────
        logo_margin = 4
        logo_size = (center_h - 2 * logo_margin, center_h - 2 * logo_margin)
        name_f = _font(22, bold=True)
        agent_f = _font(13)
        pad = 4
        text_margin = 10  # gap between logo and text column
        row2_y = my + box_side + 11   # team name sits below the turn boxes

        # Away (left edge): [logo] [agent name on turn-box row / team name below]
        away_logo = spr.get_team_logo(away.race, logo_size)
        surface.blit(away_logo, (self.rect.x + pad + logo_margin, center_y + logo_margin))
        tx = self.rect.x + pad + center_h + text_margin
        if game.away_agent:
            ag_s = agent_f.render(game.away_agent.name, True, COLOR_TEXT_DIM)
            surface.blit(ag_s, (tx, my + (box_side - ag_s.get_height()) // 2))
        away_name_s = name_f.render(away.name, True, COLOR_AWAY)
        surface.blit(away_name_s, (tx, row2_y))

        # Home (right edge): [agent name on turn-box row / team name below] [logo]
        home_logo = spr.get_team_logo(home.race, logo_size)
        lx = self.rect.x + w - pad - center_h
        surface.blit(home_logo, (lx + logo_margin, center_y + logo_margin))
        if game.home_agent:
            ag_s2 = agent_f.render(game.home_agent.name, True, COLOR_TEXT_DIM)
            surface.blit(ag_s2, (lx - pad - ag_s2.get_width(),
                                 my + (box_side - ag_s2.get_height()) // 2))
        home_name_s = name_f.render(home.name, True, COLOR_HOME)
        surface.blit(home_name_s, (lx - pad - home_name_s.get_width(), row2_y))

    def _draw_reroll_boxes(self, surface: pygame.Surface, team, anchor_x: int, y: int,
                           box_size: int, gap: int, is_home: bool, game,
                           align_right: bool, flip: bool = False):
        """Draw reroll indicator boxes for a team.
        flip=True mirrors the used/available ordering (used boxes on the right)."""
        total = team.state.rerolls_start
        if total == 0:
            return
        used = total - team.state.rerolls
        color = COLOR_HOME if is_home else COLOR_AWAY
        avail_action = any(
            str(a.action_type).endswith('USE_REROLL') and a.team == team
            for a in game.state.available_actions
        )
        total_w = total * (box_size + gap) - gap
        ix = anchor_x - total_w if align_right else anchor_x
        for i in range(total):
            bx = ix + i * (box_size + gap)
            rect = pygame.Rect(bx, y, box_size, box_size)
            effective_i = (total - 1 - i) if flip else i
            if effective_i < used:
                pygame.draw.rect(surface, COLOR_REROLL_USED, rect)
            else:
                # Highlight the next-to-use box
                next_used_i = (total - 1 - used) if flip else used
                fill = COLOR_REROLL_AVAIL if (avail_action and i == next_used_i) else color
                pygame.draw.rect(surface, fill, rect)
            pygame.draw.rect(surface, COLOR_REROLL_BORDER, rect, 1)

    def _draw_resource_row(self, surface: pygame.Surface, team, anchor_x: int, y: int,
                           gap: int, align_right: bool, flip: bool):
        """Draw apothecary/bribe/wizard icons in a row. flip reverses the order."""
        ICON_SIZE = 12
        icons = []
        for _ in range(getattr(team.state, 'apothecaries', 0)):
            icons.append('apothecary')
        for _ in range(getattr(team.state, 'bribes', 0)):
            icons.append('bribe')
        if getattr(team.state, 'wizard_available', False):
            icons.append('wizard')
        if not icons:
            return
        if flip:
            icons = list(reversed(icons))
        total_w = len(icons) * (ICON_SIZE + gap) - gap
        ix = anchor_x - total_w if align_right else anchor_x
        for name in icons:
            surf = spr.get_resource_icon(name, (ICON_SIZE, ICON_SIZE))
            surface.blit(surf, (ix, y))
            ix += ICON_SIZE + gap

    def _draw_clock(self, surface: pygame.Surface, game, cx: int):
        """Draw MM:SS clock if active team has time remaining."""
        if not game.state.clocks:
            return
        for clock in game.state.clocks:
            if clock.is_running() if hasattr(clock, 'is_running') else False:
                remaining = clock.get_remaining() if hasattr(clock, 'get_remaining') else 0
                mm = int(remaining) // 60
                ss = int(remaining) % 60
                clock_str = f'{mm:02d}:{ss:02d}'
                color = COLOR_CLOCK_BUZZER if remaining < 10 else COLOR_CLOCK_NORMAL
                font = _font(14, bold=True)
                surf = font.render(clock_str, True, color)
                surface.blit(surf, (cx - surf.get_width() // 2, self.rect.y + 50))
                break


class ActionBarRenderer:
    """Renders the action button bar below the pitch."""

    def __init__(self, rect: pygame.Rect):
        self.rect = rect

    def draw(self, surface: pygame.Surface, buttons: list):
        pygame.draw.rect(surface, (15, 15, 20), self.rect)
        pygame.draw.rect(surface, (50, 50, 60), self.rect, 1)
        for btn in buttons:
            btn.draw(surface)


class PlayerInfoRenderer:
    """Renders detailed player info for the selected/hovered player."""

    def __init__(self, rect: pygame.Rect):
        self.rect = rect

    def draw(self, surface: pygame.Surface, player, game):
        pygame.draw.rect(surface, (20, 20, 28), self.rect)
        pygame.draw.rect(surface, (50, 50, 60), self.rect, 1)

        if player is None:
            hint = _font(12).render('Click a player', True, (80, 80, 80))
            surface.blit(hint, (self.rect.x + 8, self.rect.y + 8))
            return

        is_home = (player.team == game.state.home_team)
        team_color = COLOR_HOME if is_home else COLOR_AWAY

        # Three-row layout:
        #   Row 1: icon + player name / role
        #   Row 2: attribute table (MA ST AG AV)
        #   Row 3: skill + state chips
        ROW1_H = 42
        ATTR_H = 17   # header row
        ATTR_V = 20   # value row
        ROW2_H = ATTR_H + ATTR_V
        row1_y = self.rect.y + 2
        row2_y = row1_y + ROW1_H + 1
        row3_y = row2_y + ROW2_H + 2

        # ── Row 1: icon + name/role ───────────────────────────────────────
        icon_size = ROW1_H - 2
        sprite = spr.get_player_surface(player, is_home, False, (icon_size, icon_size))
        surface.blit(sprite, (self.rect.x + 2, row1_y))
        tx = self.rect.x + icon_size + 6
        surface.blit(
            _font(12, bold=True).render(f'{player.nr}. {player.name}', True, (220, 220, 220)),
            (tx, row1_y + 1))
        surface.blit(
            _font(14).render(player.role.name if player.role else '', True, (160, 160, 160)),
            (tx, row1_y + 17))

        # ── Row 2: attribute table ────────────────────────────────────────
        COL_W = max(28, (self.rect.width - 4) // 4)
        stats = [('MA', player.get_ma()), ('ST', player.get_st()),
                 ('AG', player.get_ag()), ('AV', player.get_av())]
        lbl_font = _font(10, bold=True)
        val_font = _font(12, bold=True)
        for i, (label, value) in enumerate(stats):
            cx = self.rect.x + 2 + i * COL_W
            hdr = pygame.Rect(cx, row2_y, COL_W, ATTR_H)
            pygame.draw.rect(surface, team_color, hdr)
            pygame.draw.rect(surface, (0, 0, 0), hdr, 1)
            ls = lbl_font.render(label, True, (255, 255, 255))
            surface.blit(ls, (cx + (COL_W - ls.get_width()) // 2,
                               row2_y + (ATTR_H - ls.get_height()) // 2))
            vr = pygame.Rect(cx, row2_y + ATTR_H, COL_W, ATTR_V)
            pygame.draw.rect(surface, (35, 35, 45), vr)
            pygame.draw.rect(surface, (60, 60, 80), vr, 1)
            vs = val_font.render(str(value), True, (220, 220, 220))
            surface.blit(vs, (cx + (COL_W - vs.get_width()) // 2,
                               row2_y + ATTR_H + (ATTR_V - vs.get_height()) // 2))

        # ── Row 3: skill + state chips ────────────────────────────────────
        chip_font = _font(9, bold=True)
        chip_gap = 3
        cx = self.rect.x + 4
        cy = row3_y
        max_x = self.rect.right - 2

        for skill in player.get_skills():
            cat = _SKILL_CAT.get(skill)
            bg, fg = _CHIP.get(cat, _CHIP[SkillCategory.Extraordinary]) if cat else _CHIP[SkillCategory.Extraordinary]
            label = prettify(skill.name)
            w = chip_font.size(label)[0] + 8
            if cx + w > max_x:
                break
            w = _draw_chip(surface, label, cx, cy, chip_font, bg, fg)
            cx += w + chip_gap

        for label in (['Prone'] if not player.state.up else []) + \
                     (['Stunned'] if player.state.stunned else []) + \
                     (['Used'] if player.state.used else []):
            w = chip_font.size(label)[0] + 8
            if cx + w > max_x:
                break
            w = _draw_chip(surface, label, cx, cy, chip_font, *_INJURY_CHIP)
            cx += w + chip_gap

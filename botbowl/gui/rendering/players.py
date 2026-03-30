"""
Player and ball rendering: on-pitch sprites, state overlays, roster columns.
"""
from __future__ import annotations
import math
import pygame
from typing import Optional

from botbowl.gui import sprites as spr
from botbowl.gui.rendering.board import sq_to_px

# Roster column status icon colors / labels
COLOR_KO = (240, 200, 0)       # Yellow bell for KO'd
COLOR_CAS = (220, 50, 50)      # Red ! for casualty
COLOR_EJECTED = (50, 100, 220) # Blue ! for ejected

COLOR_BORDER_ACTIVE = (255, 220, 0)
COLOR_BORDER_SELECTED = (0, 200, 80)


class PlayerRenderer:

    def __init__(self, tile_size: int, pitch_offset: tuple):
        self.tile_size = tile_size
        self.pitch_offset = pitch_offset

    def draw_players(self, surface: pygame.Surface, game,
                     selected_player=None, ui_state=None):
        """Draw all on-pitch players with state overlays."""
        ts = self.tile_size
        active = game.get_active_player()
        home_team = game.state.home_team
        away_team = game.state.away_team

        pinned_home = ui_state.pinned_home_player if ui_state else None
        pinned_away = ui_state.pinned_away_player if ui_state else None

        for y in range(game.state.pitch.height):
            for x in range(game.state.pitch.width):
                player = game.state.pitch.board[y][x]
                if player is None:
                    continue
                is_home = (player.team == home_team)
                is_active = (player is active)
                # Show selection border for: action-selected player OR either pinned player
                is_selected = (player is selected_player or
                               player is pinned_home or
                               player is pinned_away)

                sq = player.position
                px, py = sq_to_px(sq, ts, self.pitch_offset)
                self._draw_player_at(surface, player, px, py,
                                     is_home, is_active, is_selected, ts)

    def _draw_player_at(self, surface: pygame.Surface, player,
                        px: int, py: int,
                        is_home: bool, is_active: bool, is_selected: bool,
                        ts: int):
        sprite = spr.get_player_surface(player, is_home, is_active)
        iw, ih = sprite.get_size()
        sx = px + (ts - iw) // 2
        sy = py + (ts - ih) // 2

        # Darken used players by multiplying RGB channels (alpha preserved → only icon pixels affected)
        if player.state.used and not is_active:
            blit_sprite = sprite.copy()
            dark = pygame.Surface((iw, ih))
            dark.fill((130, 130, 130))
            blit_sprite.blit(dark, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        else:
            blit_sprite = sprite

        surface.blit(blit_sprite, (sx, sy))

        # Pulsating white brightness for selected player (alpha preserved → only icon pixels affected)
        if is_selected and not player.state.used:
            t = pygame.time.get_ticks()
            phase = (math.sin(t / 350.0) + 1) / 2  # 0.0 → 1.0, ~1.4s cycle
            white_amount = int(110 * phase)          # 0 → 110 brightness boost
            brightened = sprite.copy()
            white = pygame.Surface((iw, ih))
            white.fill((white_amount, white_amount, white_amount))
            brightened.blit(white, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
            surface.blit(brightened, (sx, sy))

        # State indicators (top-left overlay, natural size)
        if player.state.stunned:
            surface.blit(spr.get_state_surface('stunned'), (px, py))
        elif not player.state.up:
            surface.blit(spr.get_state_surface('prone'), (px, py))

        # Skill-based state indicators (top-right overlay)
        if hasattr(player.state, 'bone_headed') and player.state.bone_headed:
            surface.blit(spr.get_state_surface('bonehead'), (px + ts // 2, py))
        if hasattr(player.state, 'really_stupid') and player.state.really_stupid:
            surface.blit(spr.get_state_surface('reallystupid'), (px + ts // 2, py))
        if hasattr(player.state, 'wild_animal') and player.state.wild_animal:
            surface.blit(spr.get_state_surface('wildanimal'), (px + ts // 2, py))
        if hasattr(player.state, 'taken_root') and player.state.taken_root:
            surface.blit(spr.get_state_surface('takenroot'), (px + ts // 2, py))
        if hasattr(player.state, 'hypnotized') and player.state.hypnotized:
            surface.blit(spr.get_state_surface('hypnotized'), (px + ts // 2, py + ts // 2))


    def draw_ball(self, surface: pygame.Surface, game):
        """Draw the ball on pitch."""
        ts = self.tile_size
        for ball in game.state.pitch.balls:
            if ball.position is None:
                continue
            # Check if ball is carried (player on that square)
            player = game.state.pitch.board[ball.position.y][ball.position.x]
            is_carried = player is not None
            px, py = sq_to_px(ball.position, ts, self.pitch_offset)
            if is_carried:
                # Ball overlay at full tile size when carried
                ball_surf = spr.get_ball_surface(is_carried, (ts, ts))
                surface.blit(ball_surf, (px, py))
            else:
                # Full-tile ball when on the ground
                ball_surf = spr.get_ball_surface(is_carried, (ts, ts))
                surface.blit(ball_surf, (px, py))

    def draw_bench_on_board(self, surface: pygame.Surface, game, team,
                            is_home: bool, selected_player=None,
                            show_numbers: bool = False):
        """
        Draw non-fielded players on the crowd column of the pitch board.
        Away (is_home=False): column x=0, starting at y=1, going down.
        Home (is_home=True): column x=arena.width-1, starting at y=1, going down.
        Status icons: yellow dot=KO, red!=casualty, blue!=ejected, none=reserve.
        """
        ts = self.tile_size
        ox, oy = self.pitch_offset
        arena_h = game.arena.height

        off_pitch = [p for p in team.players if p.position is None]
        if not off_pitch:
            return

        ko_set = set(id(p) for p in game.get_knocked_out(team))
        cas_set = set(id(p) for p in game.get_casualties(team))
        ej_set = set(id(p) for p in game.get_dungeon(team))

        icon_font = pygame.font.SysFont('Arial', 11, bold=True)
        nr_font = pygame.font.SysFont('Arial', max(7, ts // 4), bold=True) if show_numbers else None

        x_tile = game.arena.width - 1 if is_home else 0
        y_tile = 1

        for player in off_pitch:
            if y_tile > arena_h - 2:
                break

            px = ox + x_tile * ts
            py = oy + y_tile * ts

            sprite = spr.get_player_surface(player, is_home, False)
            iw, ih = sprite.get_size()
            surface.blit(sprite, (px + (ts - iw) // 2, py + (ts - ih) // 2))

            pid = id(player)
            if pid in ko_set:
                pygame.draw.circle(surface, COLOR_KO, (px + ts - 5, py + 5), 5)
            elif pid in cas_set:
                icon = icon_font.render('!', True, COLOR_CAS)
                surface.blit(icon, (px + ts - 8, py))
            elif pid in ej_set:
                icon = icon_font.render('!', True, COLOR_EJECTED)
                surface.blit(icon, (px + ts - 8, py))

            if show_numbers:
                nr_surf = nr_font.render(str(player.nr), True, (255, 255, 255))
                shadow = nr_font.render(str(player.nr), True, (0, 0, 0))
                nr_rect = nr_surf.get_rect(bottomright=(px + ts - 2, py + ts - 2))
                surface.blit(shadow, (nr_rect.x + 1, nr_rect.y + 1))
                surface.blit(nr_surf, nr_rect)

            y_tile += 1

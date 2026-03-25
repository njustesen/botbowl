"""
Action button building and rendering.
Handles action bar buttons, player action dots, block dice buttons.
"""
from __future__ import annotations
import pygame
from typing import Optional

from botbowl.core.table import ActionType
from botbowl.gui.assets import prettify
from botbowl.gui import sprites as spr
from botbowl.gui.rendering.ui_primitives import (
    Button, COLOR_BTN_HOME, COLOR_BTN_AWAY, COLOR_BTN_DEFAULT,
    COLOR_BTN_NEUTRAL
)
from botbowl.gui.rendering.board import sq_to_px

# Action types that are handled by clicking on pitch squares (positional)
POSITIONAL_ACTIONS = {
    ActionType.MOVE,
    ActionType.BLOCK,
    ActionType.PASS,
    ActionType.HANDOFF,
    ActionType.FOUL,
    ActionType.PLACE_PLAYER,
    ActionType.PLACE_BALL,
    ActionType.LEAP,
    ActionType.STAB,
    ActionType.HYPNOTIC_GAZE,
    ActionType.THROW_TEAM_MATE,
    ActionType.THROW_BOMB,
    ActionType.SELECT_PLAYER,
    ActionType.PUSH,
    ActionType.FOLLOW_UP,
    ActionType.PICKUP_TEAM_MATE,
}

# Action types that are select-player START actions (show action dots on selected player)
START_ACTIONS = {
    ActionType.START_MOVE,
    ActionType.START_BLOCK,
    ActionType.START_BLITZ,
    ActionType.START_PASS,
    ActionType.START_HANDOFF,
    ActionType.START_FOUL,
    ActionType.START_THROW_BOMB,
}

# Block dice result actions
BLOCK_DICE_ACTIONS = {
    ActionType.SELECT_ATTACKER_DOWN,
    ActionType.SELECT_DEFENDER_DOWN,
    ActionType.SELECT_BOTH_DOWN,
    ActionType.SELECT_PUSH,
    ActionType.SELECT_DEFENDER_STUMBLES,
}

_BLOCK_DIE_NAMES = {
    ActionType.SELECT_ATTACKER_DOWN: 'ATTACKER_DOWN',
    ActionType.SELECT_DEFENDER_DOWN: 'DEFENDER_DOWN',
    ActionType.SELECT_BOTH_DOWN: 'BOTH_DOWN',
    ActionType.SELECT_PUSH: 'PUSH',
    ActionType.SELECT_DEFENDER_STUMBLES: 'DEFENDER_STUMBLES',
}


def _btn_color(action_type, game) -> tuple:
    """Determine button color based on action team."""
    # Check which team this action belongs to
    for ac in game.state.available_actions:
        if ac.action_type == action_type:
            if ac.team == game.state.home_team:
                return COLOR_BTN_HOME
            elif ac.team == game.state.away_team:
                return COLOR_BTN_AWAY
            break
    return COLOR_BTN_DEFAULT


def build_action_buttons(available_actions: list, game,
                         bar_rect: pygame.Rect) -> list[Button]:
    """Build action bar buttons for the current available actions."""
    buttons = []
    btn_h = bar_rect.height - 8
    x = bar_rect.x + 4
    y = bar_rect.y + 4
    btn_gap = 4

    home_team = game.state.home_team
    away_team = game.state.away_team

    for ac in available_actions:
        at = ac.action_type

        # Skip purely positional actions — handled by board clicks
        if at in POSITIONAL_ACTIONS:
            continue
        # Skip START_* actions — handled by player_dot clicks
        if at in START_ACTIONS:
            continue

        # Determine color
        if ac.team == home_team:
            color = COLOR_BTN_HOME
        elif ac.team == away_team:
            color = COLOR_BTN_AWAY
        else:
            color = COLOR_BTN_NEUTRAL

        if at in BLOCK_DICE_ACTIONS:
            # Show die image button
            die_name = _BLOCK_DIE_NAMES[at]
            die_size = (btn_h, btn_h)
            die_img = spr.get_block_die_surface(die_name, die_size)
            btn = Button(
                rect=pygame.Rect(x, y, btn_h + 4, btn_h),
                action=ac,
                image=die_img,
                color=color,
                tooltip=prettify(at.name)
            )
        else:
            label = _action_label(at)
            btn_w = max(60, len(label) * 8 + 16)
            btn = Button(
                rect=pygame.Rect(x, y, btn_w, btn_h),
                label=label,
                action=ac,
                color=color,
                font_size=12,
                tooltip=prettify(at.name)
            )

        buttons.append(btn)
        x += btn.rect.width + btn_gap

    return buttons


def build_player_action_dots(player, available_actions: list,
                              tile_size: int, pitch_offset: tuple) -> list[Button]:
    """Build small circular action icon buttons around a selected player's square."""
    if player is None or player.position is None:
        return []

    sq = player.position
    px, py = sq_to_px(sq, tile_size, pitch_offset)
    ts = tile_size

    # Positions around the player square (N, NE, E, SE, S, SW, W, NW)
    offsets = [
        (ts // 2, -ts), (ts, -ts), (ts, 0), (ts, ts),
        (ts // 2, ts), (-ts // 4, ts), (-ts // 4, 0), (-ts // 4, -ts)
    ]

    dot_size = max(20, ts * 2 // 3)
    buttons = []
    oi = 0

    for ac in available_actions:
        at = ac.action_type
        if at not in START_ACTIONS:
            continue
        # Check this action applies to the selected player
        if player not in (ac.players or []):
            continue

        if oi >= len(offsets):
            break
        ox, oy = offsets[oi]
        bx = px + ox
        by = py + oy
        icon = spr.get_action_icon(at.name, (dot_size, dot_size))
        btn = Button(
            rect=pygame.Rect(bx, by, dot_size, dot_size),
            action=ac,
            image=icon,
            color=(40, 40, 50),
            tooltip=_action_label(at)
        )
        buttons.append(btn)
        oi += 1

    return buttons


def _action_label(action_type: ActionType) -> str:
    label_map = {
        ActionType.START_GAME: 'Start Game',
        ActionType.END_TURN: 'End Turn',
        ActionType.END_PLAYER_TURN: 'End Player Turn',
        ActionType.END_SETUP: 'End Setup',
        ActionType.USE_REROLL: 'Re-roll',
        ActionType.DONT_USE_REROLL: "Don't Re-roll",
        ActionType.USE_SKILL: 'Use Skill',
        ActionType.DONT_USE_SKILL: "Don't Use Skill",
        ActionType.USE_APOTHECARY: 'Use Apothecary',
        ActionType.DONT_USE_APOTHECARY: "Don't Use Apothecary",
        ActionType.USE_BRIBE: 'Use Bribe',
        ActionType.DONT_USE_BRIBE: "Don't Use Bribe",
        ActionType.HEADS: 'Heads',
        ActionType.TAILS: 'Tails',
        ActionType.KICK: 'Kick',
        ActionType.RECEIVE: 'Receive',
        ActionType.SELECT_ATTACKER_DOWN: 'Att. Down',
        ActionType.SELECT_DEFENDER_DOWN: 'Def. Down',
        ActionType.SELECT_BOTH_DOWN: 'Both Down',
        ActionType.SELECT_PUSH: 'Push',
        ActionType.SELECT_DEFENDER_STUMBLES: 'Stumbles',
        ActionType.STAND_UP: 'Stand Up',
        ActionType.CONTINUE: 'Continue',
        ActionType.FOLLOW_UP: 'Follow Up',
        ActionType.SELECT_NONE: 'Skip',
        ActionType.UNDO: 'Undo',
        ActionType.SETUP_FORMATION_WEDGE: 'Wedge',
        ActionType.SETUP_FORMATION_LINE: 'Line',
        ActionType.SETUP_FORMATION_SPREAD: 'Spread',
        ActionType.SETUP_FORMATION_ZONE: 'Zone',
    }
    return label_map.get(action_type, prettify(action_type.name))

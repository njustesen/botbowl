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
from botbowl.gui.fonts import get_font
from botbowl.gui.rendering.ui_primitives import (
    Button, COLOR_BTN_HOME, COLOR_BTN_AWAY, COLOR_BTN_DEFAULT,
    COLOR_BTN_NEUTRAL, get_button_image
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

# Formation actions — auto-triggered, hidden from the action bar
FORMATION_ACTIONS = {
    ActionType.SETUP_FORMATION_WEDGE,
    ActionType.SETUP_FORMATION_LINE,
    ActionType.SETUP_FORMATION_ZONE,
    ActionType.SETUP_FORMATION_SPREAD,
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
    btn_h = bar_rect.height - 4
    btn_gap = 8
    y = bar_rect.y + (bar_rect.height - btn_h) // 2

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
        # Skip formation actions — auto-triggered, not shown as buttons
        if at in FORMATION_ACTIONS:
            continue

        if at in BLOCK_DICE_ACTIONS:
            # Show die image button
            die_name = _BLOCK_DIE_NAMES[at]
            die_size = (btn_h, btn_h)
            die_img = spr.get_block_die_surface(die_name, die_size)
            btn = Button(
                rect=pygame.Rect(0, y, btn_h + 4, btn_h),
                action=ac,
                image=die_img,
                bg_image=get_button_image('default'),
                tooltip=prettify(at.name)
            )
        else:
            label = _action_label(at, game)
            text_w = get_font(14, bold=True).size(label)[0]
            btn_w = text_w + 48
            btn = Button(
                rect=pygame.Rect(0, y, btn_w, btn_h),
                label=label,
                action=ac,
                bg_image=get_button_image('default'),
                font_size=14,
                tooltip=prettify(at.name)
            )

        buttons.append(btn)

    # Center the button group horizontally in the bar
    if buttons:
        total_w = sum(b.rect.width for b in buttons) + btn_gap * (len(buttons) - 1)
        x = bar_rect.centerx - total_w // 2
        for b in buttons:
            b.rect.x = x
            x += b.rect.width + btn_gap

    return buttons


def build_player_action_dots(player, available_actions: list,
                              tile_size: int, pitch_offset: tuple) -> list[Button]:
    """Build action icon buttons in a horizontal panel above the selected player."""
    if player is None or player.position is None:
        return []

    sq = player.position
    px, py = sq_to_px(sq, tile_size, pitch_offset)
    ts = tile_size

    # Collect actions relevant to this player
    player_actions = [
        ac for ac in available_actions
        if ac.action_type in START_ACTIONS and player in (ac.players or [])
    ]
    if not player_actions:
        return []

    # Load icons at natural size first to determine layout dimensions
    icons = [spr.get_action_icon(ac.action_type.name) for ac in player_actions]
    icon_size = max(img.get_height() for img in icons)

    padding = 4
    gap = 3
    n = len(player_actions)
    panel_w = n * icon_size + (n - 1) * gap + 2 * padding

    # Center panel horizontally over the player square
    panel_x = (px + ts // 2) - panel_w // 2

    # Place above the player square; fall back to below if too close to top edge
    panel_y_above = py - icon_size - 2 * padding - 4
    ox, oy = pitch_offset
    panel_y = panel_y_above if panel_y_above >= oy else py + ts + 4

    buttons = []
    for i, (ac, icon) in enumerate(zip(player_actions, icons)):
        at = ac.action_type
        iw, ih = icon.get_size()
        bx = panel_x + padding + i * (icon_size + gap) + (icon_size - iw) // 2
        by = panel_y + padding + (icon_size - ih) // 2
        btn = Button(
            rect=pygame.Rect(panel_x + padding + i * (icon_size + gap), panel_y + padding, icon_size, icon_size),
            action=ac,
            image=icon,
            color=(40, 40, 50),
            tooltip=_action_label(at)
        )
        buttons.append(btn)

    return buttons


def _action_label(action_type: ActionType, game=None) -> str:
    end_turn_label = 'End Turn'
    if game is not None and action_type == ActionType.END_TURN:
        if game.is_quick_snap():
            end_turn_label = 'End Quick Snap'
        elif game.is_blitz():
            end_turn_label = 'End Blitz'

    label_map = {
        ActionType.START_GAME: 'Start Game',
        ActionType.END_TURN: end_turn_label,
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

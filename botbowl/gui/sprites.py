"""
Image loading and sprite cache for the botbowl pygame GUI.
All images are loaded lazily (on first access) after pygame.init().
"""
import os
import pygame

from botbowl.core.util import get_data_path
from botbowl.gui.fonts import get_body_font

IMG_ROOT = get_data_path("img")

_cache: dict = {}

# Race → role → base filename (ported from feature/python-gui branch)
player_icons = {
    'Chaos': {
        'Beastman': 'cbeastman',
        'Chaos Warrior': 'cwarrior',
        'Minotaur': 'minotaur',
    },
    'Chaos Dwarf': {
        'Hobgoblin': 'cdhobgoblin',
        'Chaos Dwarf Blocker': 'cddwarf',
        'Bull Centaur': 'centaur',
        'Minotaur': 'minotaur',
    },
    'Dark Elf': {
        'Lineman': 'delineman',
        'Blitzer': 'deblitzer',
        'Witch Elf': 'dewitchelf',
        'Runner': 'dethrower',
        'Assassin': 'dehorkon',
    },
    'High Elf': {
        'Lineman': 'helineman',
        'Blitzer': 'heblitzer',
        'Thrower': 'hethrower',
        'Catcher': 'hecatcher',
    },
    'Wood Elf': {
        'Lineman': 'welineman',
        'Wardancer': 'weblitzer',
        'Thrower': 'wethrower',
        'Catcher': 'wecatcher',
        'Treeman': 'treeman',
    },
    'Human': {
        'Lineman': 'hlineman',
        'Blitzer': 'hblitzer',
        'Thrower': 'hthrower',
        'Catcher': 'hcatcher',
        'Ogre': 'ogre',
    },
    'Lizardman': {
        'Kroxigor': 'kroxigor',
        'Saurus': 'lmsaurus',
        'Skink': 'lmskink',
    },
    'Orc': {
        'Lineman': 'olineman',
        'Blitzer': 'oblitzer',
        'Thrower': 'othrower',
        'Black Orc Blocker': 'oblackorc',
        'Troll': 'troll',
        'Goblin': 'goblin',
    },
    'Elven Union': {
        'Lineman': 'eplineman',
        'Blitzer': 'epblitzer',
        'Thrower': 'epthrower',
        'Catcher': 'epcatcher',
    },
    'Skaven': {
        'Lineman': 'sklineman',
        'Blitzer': 'skstorm',
        'Thrower': 'skthrower',
        'Gutter Runner': 'skrunner',
        'Rat Ogre': 'ratogre',
    },
    'Amazon': {
        'Linewoman': 'amlineman',
        'Blitzer': 'amblitzer',
        'Thrower': 'amthrower',
        'Catcher': 'amcatcher',
    },
    'Undead': {
        'Zombie': 'uzombie',
        'Skeleton': 'uskeleton',
        'Ghoul': 'ughoul',
        'Wight': 'uwight',
        'Mummy': 'umummy',
    },
    'Vampire': {
        'Vampire': 'vampire',
        'Thrall': 'vthrall',
    },
    'Dwarf': {
        'Blocker': 'dlongbeard',
        'Runner': 'drunner',
        'Blitzer': 'dblitzer',
        'Troll Slayer': 'dslayer',
        'Deathroller': 'ddeathroller',
    },
    'Halfling': {
        'Halfling': 'halfling',
        'Treeman': 'treeman',
    },
    'Goblin': {
        'Goblin': 'goblin',
        'Bombardier': 'gobomber',
        'Looney': 'gonobbla',
        'Fanatic': 'gofungus',
        'Pogoer': 'goscrappa',
        'Troll': 'troll',
    },
    'Ogre': {
        'Snotling': 'snot',
        'Ogre': 'ogre',
    },
    'Khemri': {
        'Skeleton': 'kmskeleton',
        'Thro-Ra': 'kmthrower',
        'Blitz-Ra': 'kmblitzer',
        'Tomb Guardian': 'kmmummy',
    },
    'Necromantic': {
        'Zombie': 'nzombie',
        'Ghoul': 'nghoul',
        'Wight': 'uwight',
        'Flesh Golem': 'ngolem',
        'Necromantic Werewolf': 'nwerewolf',
    },
    'Norse': {
        'Lineman': 'nlineman',
        'Thrower': 'nthrower',
        'Catcher': 'ncatcher',
        'Blitzer': 'nblitzer',
        'Norse Werewolf': 'nwerewolf',
    },
    'Nurgle': {
        'Rotter': 'rtrotter',
        'Beast of Nurgle': 'rtbeast',
    },
    'Slann': {
        'Lineman': 'lisilibili',
        'Catcher': 'lisilibili',
        'Blitzer': 'lisilibili',
        'Kroxigor': 'kroxigor',
    },
    'Underworld': {
        'Underworld Goblin': 'goblin',
        'Skaven Lineman': 'sklineman',
        'Skaven Thrower': 'skthrower',
        'Skaven Blitzer': 'skstorm',
        'Warpstone Troll': 'troll',
    },
    'Chaos Pact': {
        'Marauder': 'nlineman',
        'Goblin Renegade': 'goblin',
        'Skaven Renegade': 'sklineman',
        'Dark Elf Renegade': 'delineman',
        'Chaos Troll': 'troll',
        'Chaos Ogre': 'ogre',
        'Minotaur': 'minotaur',
    },
}

# WeatherType enum name → pitch image prefix
_WEATHER_TO_PITCH = {
    'SWELTERING_HEAT': 'heat',
    'VERY_SUNNY': 'sunny',
    'NICE': 'nice',
    'POURING_RAIN': 'rain',
    'BLIZZARD': 'blizzard',
}

# WeatherType enum name → weather icon filename
_WEATHER_TO_ICON = {
    'SWELTERING_HEAT': 'sweltering_heat.gif',
    'VERY_SUNNY': 'very_sunny.gif',
    'NICE': 'nice.gif',
    'POURING_RAIN': 'pouring_rain.gif',
    'BLIZZARD': 'blizzard.gif',
}


def _img_path(*parts) -> str:
    return os.path.join(IMG_ROOT, *parts)


def _load(path: str) -> pygame.Surface:
    if path not in _cache:
        _cache[path] = pygame.image.load(path).convert_alpha()
    return _cache[path]


def _load_scaled(path: str, size: tuple) -> pygame.Surface:
    key = (path, size)
    if key not in _cache:
        raw = pygame.image.load(path).convert_alpha()
        _cache[key] = pygame.transform.smoothscale(raw, size)
    return _cache[key]


def _fallback_circle(size: tuple, color: tuple) -> pygame.Surface:
    surf = pygame.Surface(size, pygame.SRCALPHA)
    cx, cy = size[0] // 2, size[1] // 2
    pygame.draw.ellipse(surf, color, (0, 0, size[0], size[1]))
    return surf


def get_pitch_surface(weather_name: str, arena_width: int, arena_height: int,
                      tile_size: int) -> pygame.Surface:
    """Return pitch background image scaled to the inner pitch size (excluding crowd border)."""
    # Inner pitch = arena - 2 (crowd border removed)
    inner_w = arena_width - 2
    inner_h = arena_height - 2
    prefix = _WEATHER_TO_PITCH.get(weather_name, 'nice')
    filename = f"{prefix}-{inner_w}x{inner_h}.jpg"
    path = _img_path("arenas", "pitch", filename)
    px_w = inner_w * tile_size
    px_h = inner_h * tile_size
    if not os.path.exists(path):
        # Fallback: solid green surface
        key = ('pitch_fallback', px_w, px_h)
        if key not in _cache:
            surf = pygame.Surface((px_w, px_h))
            surf.fill((100, 160, 100))
            _cache[key] = surf
        return _cache[key]
    return _load_scaled(path, (px_w, px_h))


def get_dugout_surface(side: str, tile_size: int, height: int) -> pygame.Surface:
    """Return a dugout image (left or right) scaled to the pitch height."""
    filename = f"dugout-{side}.jpg"
    path = _img_path("arenas", "dugouts", filename)
    if not os.path.exists(path):
        key = ('dugout_fallback', side, height)
        if key not in _cache:
            surf = pygame.Surface((tile_size * 2, height))
            surf.fill((80, 80, 80))
            _cache[key] = surf
        return _cache[key]
    raw = pygame.image.load(path).convert()
    aspect = raw.get_width() / raw.get_height()
    w = int(height * aspect)
    return _load_scaled(path, (w, height))


def get_player_surface(player, is_home: bool, is_active: bool) -> pygame.Surface:
    """Return player sprite at natural size, with fallback colored circle."""
    race = player.team.race if hasattr(player.team, 'race') else None
    role_name = player.role.name if player.role else None

    base_icon = None
    if race and role_name and race in player_icons:
        base_icon = player_icons[race].get(role_name)

    if base_icon:
        team_letter = "b" if is_home else ""
        angle = "an" if is_active else ""
        filename = f"{base_icon}1{team_letter}{angle}.gif"
        path = _img_path("iconssmall", filename)
        if os.path.exists(path):
            return _load(path)

    # Fallback: colored circle
    color = (34, 119, 204) if is_home else (204, 119, 34)
    return _fallback_circle((30, 30), color)


def get_ball_surface(is_carried: bool, size: tuple) -> pygame.Surface:
    filename = "holdball3.gif" if is_carried else "sball_30x30.png"
    path = _img_path("ball", filename)
    if os.path.exists(path):
        return _load_scaled(path, size)
    surf = pygame.Surface(size, pygame.SRCALPHA)
    pygame.draw.ellipse(surf, (220, 80, 220), (0, 0, size[0], size[1]))
    return surf


def _die_surface(label: str, size: tuple, bg: tuple, fg: tuple) -> pygame.Surface:
    """Render a small die badge: rounded rect with a centered label."""
    key = ('die', label, size, bg, fg)
    if key in _cache:
        return _cache[key]
    surf = pygame.Surface(size, pygame.SRCALPHA)
    pygame.draw.rect(surf, bg, (0, 0, size[0], size[1]), border_radius=3)
    pygame.draw.rect(surf, fg, (0, 0, size[0], size[1]), 1, border_radius=3)
    font = get_body_font(max(8, size[1] - 5), bold=True)
    txt = font.render(label, True, fg)
    surf.blit(txt, ((size[0] - txt.get_width()) // 2, (size[1] - txt.get_height()) // 2))
    _cache[key] = surf
    return surf


_BLOCK_COLORS = {
    'DEFENDER_DOWN':    ((40, 120, 40),   (120, 255, 120)),
    'ATTACKER_DOWN':    ((120, 40, 40),   (255, 120, 120)),
    'BOTH_DOWN':        ((120, 80, 30),   (255, 190, 80)),
    'PUSH':             ((40, 60, 110),   (130, 170, 255)),
    'DEFENDER_STUMBLES':((60, 110, 60),   (180, 240, 130)),
}
_BLOCK_LABELS = {
    'DEFENDER_DOWN': 'DD',
    'ATTACKER_DOWN': 'AD',
    'BOTH_DOWN': 'BD',
    'PUSH': 'P',
    'DEFENDER_STUMBLES': 'DS',
}
_BLOCK_FILENAMES = {
    'DEFENDER_DOWN': 'defender_down.png',
    'ATTACKER_DOWN': 'attacker_down.png',
    'BOTH_DOWN': 'both_down.png',
    'PUSH': 'push.png',
    'DEFENDER_STUMBLES': 'defender_stumbles.png',
}


def _load_on_white(path: str, size: tuple) -> pygame.Surface:
    """Load a PNG, composite onto a white background, and clip to rounded corners.
    Uses smoothscale to avoid pixelation and BLEND_RGBA_MULT to mask corners."""
    key = ('white_bg', path, size)
    if key in _cache:
        return _cache[key]
    raw = pygame.image.load(path).convert_alpha()
    scaled = pygame.transform.smoothscale(raw, size)
    # White background + image on top
    surf = pygame.Surface(size, pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    pygame.draw.rect(surf, (255, 255, 255), (0, 0, size[0], size[1]), border_radius=3)
    surf.blit(scaled, (0, 0))
    # Clip corners: multiply alpha by a rounded-rect mask
    mask = pygame.Surface(size, pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, size[0], size[1]), border_radius=3)
    surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    # Subtle border
    pygame.draw.rect(surf, (100, 100, 100), (0, 0, size[0], size[1]), 1, border_radius=3)
    _cache[key] = surf
    return surf


def get_d6_surface(value: int, size: tuple) -> pygame.Surface:
    """value: 1-6"""
    path = _img_path("dice", f"{value}.png")
    if os.path.exists(path):
        return _load_on_white(path, size)
    return _die_surface(str(value), size, bg=(50, 50, 65), fg=(230, 210, 100))


def get_d8_surface(value: int, size: tuple) -> pygame.Surface:
    """value: 1-8"""
    path = _img_path("dice", f"d8-{value}.png")
    if os.path.exists(path):
        return _load_on_white(path, size)
    return _die_surface(str(value), size, bg=(40, 60, 80), fg=(130, 200, 255))


def get_block_die_surface(result_name: str, size: tuple) -> pygame.Surface:
    """result_name: ATTACKER_DOWN, DEFENDER_DOWN, BOTH_DOWN, PUSH, DEFENDER_STUMBLES"""
    filename = _BLOCK_FILENAMES.get(result_name)
    if filename:
        path = _img_path("dice", filename)
        if os.path.exists(path):
            return _load_on_white(path, size)
    bg, fg = _BLOCK_COLORS.get(result_name, ((80, 80, 80), (200, 200, 200)))
    label = _BLOCK_LABELS.get(result_name, '?')
    return _die_surface(label, size, bg=bg, fg=fg)


def get_state_surface(state_name: str) -> pygame.Surface:
    """state_name: stunned, prone, bonehead, hypnotized, reallystupid, wildanimal, takenroot"""
    filename_map = {
        'stunned': 'stunned.gif',
        'prone': 'prone.gif',
        'bonehead': 'bonehead.gif',
        'hypnotized': 'hypnotized.gif',
        'reallystupid': 'reallystupid.gif',
        'wildanimal': 'wildanimal.gif',
        'takenroot': 'takenroot.gif',
    }
    filename = filename_map.get(state_name.lower(), 'prone.gif')
    path = _img_path("player_status", filename)
    if os.path.exists(path):
        return _load(path)
    return pygame.Surface((0, 0), pygame.SRCALPHA)


def get_weather_icon(weather_name: str, size: tuple) -> pygame.Surface:
    filename = _WEATHER_TO_ICON.get(weather_name, 'nice.gif')
    path = _img_path("weather", filename)
    if os.path.exists(path):
        return _load_scaled(path, size)
    surf = pygame.Surface(size, pygame.SRCALPHA)
    surf.fill((200, 200, 100))
    return surf


def get_team_logo(race: str, size: tuple) -> pygame.Surface:
    filename = f"{race.lower()}.png"
    path = _img_path("teamlogos", filename)
    if os.path.exists(path):
        return _load_scaled(path, size)
    # Try with spaces replaced by underscores
    filename2 = f"{race.lower().replace(' ', '_')}.png"
    path2 = _img_path("teamlogos", filename2)
    if os.path.exists(path2):
        return _load_scaled(path2, size)
    surf = pygame.Surface(size, pygame.SRCALPHA)
    surf.fill((100, 100, 200))
    return surf


_ACTION_ICON_FILES = {
    'START_MOVE': 'move.gif',
    'START_BLOCK': 'block.gif',
    'START_BLITZ': 'blitz.gif',
    'START_PASS': 'pass.gif',
    'START_HANDOFF': 'handoff.gif',
    'START_FOUL': 'foul.gif',
    'STAB': 'stab.gif',
    'LEAP': 'leap.gif',
    'START_THROW_BOMB': 'bomb.gif',
    'USE_SKILL': 'use.gif',
    'DONT_USE_SKILL': 'dont-use.gif',
    'END_TURN': 'end.gif',
    'STAND_UP': 'standup.gif',
}


def get_action_icon(action_name: str, size: tuple = None) -> pygame.Surface:
    """Load an action icon. If size is None, returns the image at its natural size."""
    filename = _ACTION_ICON_FILES.get(action_name, 'move.gif')
    path = _img_path("icons", "actions", filename)
    if os.path.exists(path):
        if size is None:
            return _load(path)
        return _load_scaled(path, size)
    fallback_size = size or (25, 25)
    surf = pygame.Surface(fallback_size, pygame.SRCALPHA)
    surf.fill((160, 160, 160))
    return surf


def get_resource_icon(name: str, size: tuple) -> pygame.Surface:
    filename_map = {
        'apothecary': 'apothecary_38x38.png',
        'wizard': 'wizard_38x38.png',
        'bribe': 'bribe_38x38.png',
        'babe': 'bloodweiser_babe_38x38.png',
        'chef': 'master_chef_38x38.png',
        'reroll': 're_roll_38x38.png',
    }
    filename = filename_map.get(name, 're_roll_38x38.png')
    path = _img_path("icons", "sidebar", "resources", filename)
    if os.path.exists(path):
        return _load_scaled(path, size)
    surf = pygame.Surface(size, pygame.SRCALPHA)
    surf.fill((200, 200, 200))
    return surf


def clear_cache():
    _cache.clear()

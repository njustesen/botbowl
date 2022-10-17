import os
from pyglet import image
from botbowl.core.model import Player, Ball, Bomb


file_dir = os.path.dirname(__file__)
image_path = os.path.join(file_dir, "../web/static/img/")


player_icons = {
    'Chaos': {
        'Beastman': 'cbeastman',
        'Chaos Warrior': 'cwarrior',
        'Minotaur': 'minotaur'
    },
    'Chaos Dwarf': {
        'Hobgoblin': 'cdhobgoblin',
        'Chaos Dwarf Blocker': 'cddwarf',
        'Bull Centaur': 'centaur',
        'Minotaur': 'minotaur'
    },
    'Dark Elf':{
        'Lineman': 'delineman',
        'Blitzer': 'deblitzer',
        'Witch Elf': 'dewitchelf',
        'Runner': 'dethrower',
        'Assassin': 'dehorkon'
    },
    'High Elf':{
        'Lineman': 'helineman',
        'Blitzer': 'heblitzer',
        'Thrower': 'hethrower',
        'Catcher': 'hecatcher'
    },
    'Wood Elf':{
        'Lineman': 'welineman',
        'Wardancer': 'weblitzer',
        'Thrower': 'wethrower',
        'Catcher': 'wecatcher',
        'Treeman': 'treeman'
    },
    'Human': {
        'Lineman': 'hlineman',
        'Blitzer': 'hblitzer',
        'Thrower': 'hthrower',
        'Catcher': 'hcatcher',
        'Ogre': 'ogre'
    },
    'Lizardman': {
        'Kroxigor': 'kroxigor',
        'Saurus': 'lmsaurus',
        'Skink': 'lmskink'
    },
    'Orc': {
        'Lineman': 'olineman',
        'Blitzer': 'oblitzer',
        'Thrower': 'othrower',
        'Black Orc Blocker': 'oblackorc',
        'Troll': 'troll',
        'Goblin': 'goblin'
    },
    'Elven Union': {
        'Lineman': 'eplineman',
        'Blitzer': 'epblitzer',
        'Thrower': 'epthrower',
        'Catcher': 'epcatcher'
    },
    'Skaven': {
        'Lineman': 'sklineman',
        'Blitzer': 'skstorm',
        'Thrower': 'skthrower',
        'Gutter Runner': 'skrunner',
        'Rat Ogre': 'ratogre'
    },
    'Amazon': {
        'Linewoman': 'amlineman',
        'Blitzer': 'amblitzer',
        'Thrower': 'amthrower',
        'Catcher': 'amcatcher'
    },
    'Undead': {
        'Zombie': 'uzombie',
        'Skeleton': 'uskeleton',
        'Ghoul': 'ughoul',
        'Wight': 'uwight',
        'Mummy': 'umummy'
    },
    'Vampire': {
        'Vampire': 'vampire',
        'Thrall': 'vthrall'
    }
}


sprite_cache = {}


def load_image(local_path):
    return image.load(os.path.join(image_path, local_path))


def get_pitch_sprite(weather, width, height):
    w = weather.split('_')[1] if '_' in weather else weather
    p = f'arenas/pitch/{w}-{width}x{height}.jpg'
    if p not in sprite_cache:
        sprite_cache[p] = load_image(p)
    return sprite_cache[p]


def get_dugout_sprite(left=False, right=False):
    if left:
        d = f'arenas/dugouts/dugout-left.jpg'
    elif right:
        d = f'arenas/dugouts/dugout-right.jpg'
    else:
        raise Exception("Must specify left or right")
    sprite_cache[d] = load_image(d)
    return sprite_cache[d]


def get_ball_sprite(ball: Ball):
    if ball.is_carried:
        return load_image("ball/holdball3.gif")
    return load_image("ball/sball_30x30.png")


def get_player_sprite(player: Player, race, active=False, home=False):
    base_icon = player_icons[race][player.role.name]
    icon_num = "1"
    team_letter = "b" if home else ""
    angle = "an" if active else ""
    p = f"iconssmall/{base_icon}{icon_num}{team_letter}{angle}.gif"
    if p not in sprite_cache:
        sprite_cache[p] = load_image(p)
    return sprite_cache[p]

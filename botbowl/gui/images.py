from pyglet import image
import os
from botbowl.core.game import Game


file_dir = os.path.dirname(__file__)
image_path = os.path.join(file_dir, "../web/static/img/")


def pitch_img(game: Game):
    width = game.arena.width - 2
    height = game.arena.height - 2
    pitch_img = image.load(os.path.join(image_path, f'arenas/pitch/{game.state.weather.name}-{width}x{height}.jpg'))
    pitch_img.anchor_x = pitch_img.width // 2
    pitch_img.anchor_y = pitch_img.height
    return pitch_img

dugout_left = image.load(os.path.join(image_path, 'arenas/dugouts/dugout-left.jpg'))
dugout_left.anchor_x = dugout_left.width
dugout_left.anchor_y = dugout_left.height
dugout_right = image.load(os.path.join(image_path, 'arenas/dugouts/dugout-right.jpg'))
dugout_right.anchor_x = 0
dugout_right.anchor_y = dugout_right.height

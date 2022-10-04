import os
import pyglet
import time
from multiprocessing import Process, Pipe
from pyglet import image
from pyglet.shapes import Line
from botbowl import Game
from botbowl.core.model import Player
import botbowl
from pyglet.gl import *


def worker(remote, parent_remote, config):
    parent_remote.close()
    tile_size = 30
    width = tile_size * 38
    height = tile_size * 25
    window = pyglet.window.Window(width, height)
    main_batch = pyglet.graphics.Batch()

    file_dir = os.path.dirname(__file__)
    image_path = os.path.join(file_dir, "../web/static/img/")

    # text = pyglet.text.Label(text="", x=100, y=100, batch=main_batch)
    pitch_img = image.load(os.path.join(image_path, 'arenas/pitch/nice-26x15.jpg'))
    pitch_img.anchor_x = pitch_img.width // 2
    pitch_img.anchor_y = pitch_img.height
    dugout_left = image.load(os.path.join(image_path, 'arenas/dugouts/dugout-left.jpg'))
    dugout_left.anchor_x = dugout_left.width
    dugout_left.anchor_y = dugout_left.height
    dugout_right = image.load(os.path.join(image_path, 'arenas/dugouts/dugout-right.jpg'))
    dugout_right.anchor_x = 0
    dugout_right.anchor_y = dugout_right.height

    human_lineman1 = image.load(os.path.join(image_path, 'iconssmall/hlineman1.gif'))

    game_data: Game = None
    i = 0

    def draw_text(text, font_size, x, y):
        label = pyglet.text.Label(text,
                                  font_name='Times New Roman',
                                  font_size=font_size,
                                  x=x, y=y,
                                  anchor_x='center', anchor_y='center')
        label.draw()
        return label

    def draw_at(img, tile_x, tile_y, offset_x=0, offset_y=0):
        img.blit(tile_x * tile_size + img.anchor_x + offset_x,
                 height - (tile_y * tile_size + (img.height - img.anchor_y) + offset_y))

    def draw_on_pitch(img, tile_x, tile_y, offset_x=0, offset_y=0):
        draw_at(img, tile_x+5, tile_y+4, offset_x, offset_y)

    def draw_grid(color=(0, 255, 0), opacity=255):
        lines = []
        for i in range(int(width / tile_size) + 1):
            lines.append(Line(i * tile_size, 0, i * tile_size, height, 1, color=color, batch=main_batch))
            lines.append(Line(0, i * tile_size, width, i * tile_size, 1, color=color, batch=main_batch))
        for line in lines:
            line.opacity = opacity
        return lines

    def draw_player(player: Player):
        draw_on_pitch(human_lineman1, player.position.x, player.position.y)

    def draw_game(game: Game):
        for player in game.get_players_on_pitch():
            draw_player(player)

    def refresh(dt):
        nonlocal game_data
        nonlocal i
        if remote.poll():
            cmd, d = remote.recv()
            if cmd == 'update':
                game_data = d
        if game_data is None:
            return

        window.clear()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        labels = []
        label = draw_text(f"Half {game_data.state.half}", font_size=36, x=width//2, y=50)
        labels.append(label)

        draw_at(pitch_img, 6, 5)
        draw_at(dugout_left, 4, 5)
        draw_at(dugout_right, 32, 5)

        draw_game(game_data)

        lines = draw_grid(opacity=100)

        main_batch.draw()

        i += 1

    pyglet.clock.schedule_interval(refresh, 1 / 120.0)
    pyglet.app.run()


class BotBowlGUI:

    def __init__(self, config):
        self.config = config
        self.remote, work_remote = Pipe()
        self.process = Process(target=worker, args=(work_remote, self.remote, config))
        self.process.daemon = True  # If the main process crashes, we should not cause things to hang
        self.process.start()

    def update(self, game):
        self.remote.send(('update', game))

    def close(self):
        self.process.terminate()


def create_game(game_id):
    config = botbowl.load_config("bot-bowl")
    config.competition_mode = False
    config.pathfinding_enabled = False
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)
    config.competition_mode = False
    config.debug_mode = False

    away_agent = botbowl.make_bot("random")
    home_agent = botbowl.make_bot("random")

    game = botbowl.Game(game_id, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
    game.config.fast_mode = False
    return game


if __name__ == '__main__':
    gui1 = BotBowlGUI(None)
    while True:
        game1 = create_game("1")
        game1.init()
        while not game1.state.game_over:
            gui1.update(game1)
            if not game1.state.game_over:
                game1.step(None)

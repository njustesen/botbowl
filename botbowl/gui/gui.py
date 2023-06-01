import os
import pyglet
import time
from multiprocessing import Process, Pipe
import images as img
from botbowl.core.game import Game
from botbowl.core.load import *


header_height = 140
refresh_rate = 1 / 120.0
width = 1200
height = 800
pitch_top = height - header_height
tile_size = 32


def worker(remote, parent_remote, config):
    parent_remote.close()
    window = pyglet.window.Window(width, height)
    main_batch = pyglet.graphics.Batch()
    game: Game = None

    def refresh(dt):
        nonlocal game
        print(f"refreshing game: {game}")
        if remote.poll():
            cmd, d = remote.recv()
            print(f"Received command {cmd} with data {d}")
            if cmd == 'update':
                game = d
        if game is None:
            return
        print("Drawing")
        window.clear()

        pitch_img = img.pitch_img(game)
        pitch_img.blit(width / 2, pitch_top)
        img.dugout_left.blit(width / 2 - pitch_img.width / 2, pitch_top)
        img.dugout_right.blit(width / 2 + pitch_img.width / 2, pitch_top)

        for player in game.get_players_on_pitch():
            img.dugout_left.blit(player.position.x * tile_size, player.position.y * tile_size)

        main_batch.draw()

    pyglet.clock.schedule_interval(refresh, refresh_rate)
    pyglet.app.run()


class BotBowlGUI:

    def __init__(self, config):
        self.config = config
        self.remote, work_remote = Pipe()
        self.process = Process(target=worker, args=(work_remote, self.remote, config))
        self.process.daemon = True  # If the main process crashes, we should not cause things to hang
        self.process.start()

    def update(self, game):
        print("Updating")
        self.remote.send(('update', game))

    def close(self):
        self.process.terminate()


def get_game():
    config = botbowl.load_config("bot-bowl")
    config.competition_mode = False
    config.pathfinding_enabled = True
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)
    config.competition_mode = False
    config.debug_mode = False
    away_agent = botbowl.make_bot("random")
    home_agent = botbowl.make_bot("random")
    game = botbowl.Game("1", home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
    game.config.fast_mode = False
    return game


if __name__ == '__main__':
    gui1 = BotBowlGUI(None)
    gui2 = BotBowlGUI(None)
    game1 = get_game()
    game1.init()
    game2 = get_game()
    game2.init()
    for i in range(1000):
        gui1.update(game1)
        gui2.update(game2)
        time.sleep(0.1)
        game1.step(None)
        game2.step(None)


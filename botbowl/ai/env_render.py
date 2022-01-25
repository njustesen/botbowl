#from .new_env import BotBowlEnv
import tkinter as tk
import math

from botbowl.core.model import Tile, TwoPlayerArena, Skill


class EnvRenderer:
    square_size = 16
    square_size_fl = 4
    top_bar_height = 42
    bot_bar_height = 80
    layer_text_height = 26
    black = '#000000'
    white = '#ffffff'
    crowd = '#113311'
    blue = '#2277cc'
    red = '#cc7722'
    ball = '#ff00cc'
    field = '#77cc77'
    wing = '#55aa55'
    scrimmage = '#338833'

    def __init__(self, env: 'BotBowlEnv', feature_layers=False):
        self.env = env
        self.layers = env.env_conf.layers
        self.root = tk.Tk()
        self.root.title("botbowl Gym")

        game = env.game
        self.game_width = max(500, game.arena.width * EnvRenderer.square_size)
        self.game_height = game.arena.height * EnvRenderer.square_size + EnvRenderer.top_bar_height + EnvRenderer.bot_bar_height

        self.feature_layers = feature_layers

        if feature_layers:
            self.cols = math.floor(math.sqrt(len(self.layers)))
            self.rows = math.ceil(math.sqrt(len(self.layers)))
            self.fl_width = (game.arena.width + 1) * self.cols * EnvRenderer.square_size_fl + EnvRenderer.square_size_fl
            self.fl_height = ((game.arena.height + 1) * EnvRenderer.square_size_fl +
                              EnvRenderer.layer_text_height) * self.rows + EnvRenderer.square_size_fl
            self.cv = tk.Canvas(width=max(self.game_width, self.fl_width), height=self.fl_height + self.game_height,
                                master=self.root)
        else:
            self.cv = tk.Canvas(width=self.game_width, height=self.game_height, master=self.root)

    def render(self):
        self.cv.pack(side='top', fill='both', expand='yes')
        self.cv.delete("all")
        self.root.configure(background='black')
        game = self.env.game

        if game is not None:
            # Squares
            for y in range(game.arena.height):
                for x in range(game.arena.width):
                    if game.arena.board[y][x] == Tile.CROWD:
                        fill = EnvRenderer.crowd
                    elif game.arena.board[y][x] in TwoPlayerArena.home_td_tiles:
                        fill = EnvRenderer.blue
                    elif game.arena.board[y][x] in TwoPlayerArena.away_td_tiles:
                        fill = EnvRenderer.red
                    elif game.arena.board[y][x] in TwoPlayerArena.wing_left_tiles or game.arena.board[y][x] in TwoPlayerArena.wing_right_tiles:
                        fill = EnvRenderer.wing
                    elif game.arena.board[y][x] in TwoPlayerArena.scrimmage_tiles:
                        fill = EnvRenderer.scrimmage
                    else:
                        fill = EnvRenderer.field
                    self.cv.create_rectangle(EnvRenderer.square_size * x, EnvRenderer.square_size * y + EnvRenderer.top_bar_height, EnvRenderer.square_size * x + EnvRenderer.square_size, EnvRenderer.square_size * y + EnvRenderer.square_size + EnvRenderer.top_bar_height, fill=fill, outline=EnvRenderer.black)

            self.cv.create_line(game.arena.width * EnvRenderer.square_size / 2.0 - 1, EnvRenderer.top_bar_height, game.arena.width * EnvRenderer.square_size / 2.0 - 1, game.arena.height * EnvRenderer.square_size + EnvRenderer.top_bar_height, fill=EnvRenderer.black, width=2)

            # Players
            for y in range(game.state.pitch.height):
                for x in range(game.state.pitch.width):
                    player = game.state.pitch.board[y][x]
                    if player is not None:
                        self._draw_player(player, EnvRenderer.square_size * x, EnvRenderer.square_size * y + EnvRenderer.top_bar_height)

            # Dugouts
            x = 4
            y = game.arena.height * EnvRenderer.square_size + EnvRenderer.top_bar_height + 4
            for player in game.get_reserves(game.state.away_team):
                self._draw_player(player, x, y)
                x += EnvRenderer.square_size
            x = 4
            y += EnvRenderer.square_size
            for player in game.get_knocked_out(game.state.away_team):
                self._draw_player(player, x, y)
                x += EnvRenderer.square_size
            x = 4
            y += EnvRenderer.square_size
            for player in game.get_casualties(game.state.away_team):
                self._draw_player(player, x, y)
                x += EnvRenderer.square_size
            x = 4
            y += EnvRenderer.square_size
            for player in game.get_dungeon(game.state.away_team):
                self._draw_player(player, x, y)
                x += EnvRenderer.square_size

            x = game.arena.width * EnvRenderer.square_size - EnvRenderer.square_size
            y = game.arena.height * EnvRenderer.square_size + EnvRenderer.top_bar_height + 4
            for player in game.get_reserves(game.state.home_team):
                self._draw_player(player, x, y)
                x -= EnvRenderer.square_size
            x = game.arena.width * EnvRenderer.square_size - EnvRenderer.square_size
            y += EnvRenderer.square_size
            for player in game.get_knocked_out(game.state.home_team):
                self._draw_player(player, x, y)
                x -= EnvRenderer.square_size
            x = game.arena.width * EnvRenderer.square_size - EnvRenderer.square_size
            y += EnvRenderer.square_size
            for player in game.get_casualties(game.state.home_team):
                self._draw_player(player, x, y)
                x -= EnvRenderer.square_size
            x = game.arena.width * EnvRenderer.square_size - EnvRenderer.square_size
            y += EnvRenderer.square_size
            for player in game.get_dungeon(game.state.home_team):
                self._draw_player(player, x, y)
                x -= EnvRenderer.square_size

            # Ball
            for ball in game.state.pitch.balls:
                self.cv.create_oval(EnvRenderer.square_size * ball.position.x + EnvRenderer.square_size / 4,
                                    EnvRenderer.square_size * ball.position.y + EnvRenderer.square_size / 4 + EnvRenderer.top_bar_height,
                                    EnvRenderer.square_size * ball.position.x + EnvRenderer.square_size - EnvRenderer.square_size / 4,
                                    EnvRenderer.square_size * ball.position.y + EnvRenderer.square_size - EnvRenderer.square_size / 4 + EnvRenderer.top_bar_height,
                                    fill=EnvRenderer.ball, outline=EnvRenderer.black, width=1)

            # Non-spatial
            self.cv.create_text(game.arena.width * EnvRenderer.square_size / 2.0, 10, text='Half: {}, Weather: {}'.format(game.state.half, game.state.weather.name), fill='black')
            self.cv.create_text(game.arena.width * EnvRenderer.square_size / 2.0, 34, text='{}: Score: {}, Turn: {}, RR: {}/{}, Bribes: {}'.format(
                game.state.away_team.name,
                game.state.away_team.state.score,
                game.state.away_team.state.turn,
                game.state.away_team.state.rerolls,
                game.state.away_team.state.rerolls_start,
                game.state.away_team.state.bribes), fill='blue')
            self.cv.create_text(game.arena.width * EnvRenderer.square_size / 2.0, 22,
                                text='{}: Score: {}, Turn: {}, RR: {}/{}, Bribes: {}'.format(
                                    game.state.home_team.name,
                                    game.state.home_team.state.score,
                                    game.state.home_team.state.turn,
                                    game.state.home_team.state.rerolls,
                                    game.state.home_team.state.rerolls_start,
                                    game.state.home_team.state.bribes), fill='red')

        # Feature layers
        if self.feature_layers:
            row = 0
            col = 0
            for layer in self.layers:
                name = layer.name()
                grid = layer.produce(game)
                grid_x = col * (len(grid[0]) + 1) * EnvRenderer.square_size_fl + EnvRenderer.square_size_fl
                grid_y = row * (len(grid) + 1) * EnvRenderer.square_size_fl + self.game_height + EnvRenderer.square_size_fl + ((row + 1) * EnvRenderer.layer_text_height)

                self.cv.create_text(grid_x + (len(grid[0]) * EnvRenderer.square_size_fl) / 2, grid_y - EnvRenderer.layer_text_height / 2, text=name)
                self.cv.create_rectangle(grid_x,
                                         grid_y,
                                         grid_x + len(grid[0]) * EnvRenderer.square_size_fl,
                                         grid_y + len(grid) * EnvRenderer.square_size_fl,
                                         fill='black', outline=EnvRenderer.black, width=2)
                for y in range(len(grid)):
                    for x in range(len(grid[0])):
                        value = 1 - grid[y][x]
                        fill = '#%02x%02x%02x' % (int(value * 255), int(value * 255), int(value * 255))
                        self.cv.create_rectangle(EnvRenderer.square_size_fl * x + grid_x,
                                                 EnvRenderer.square_size_fl * y + grid_y,
                                                 EnvRenderer.square_size_fl * x + grid_x + EnvRenderer.square_size_fl,
                                                 EnvRenderer.square_size_fl * y + grid_y + EnvRenderer.square_size_fl,
                                                 fill=fill, outline=EnvRenderer.black)
                col += 1
                if col >= self.cols:
                    col = 0
                    row += 1

        self.root.update_idletasks()
        self.root.update()

    def _draw_player(self, player, x, y):
        if player.team == self.env.game.state.home_team:
            fill = EnvRenderer.blue
        else:
            fill = EnvRenderer.red
        width = max(1, player.get_st() - 1)
        if player.has_skill(Skill.BLOCK):
            outline = 'red'
        elif player.has_skill(Skill.CATCH):
            outline = 'yellow'
        elif player.has_skill(Skill.PASS):
            outline = 'white'
        elif player.get_st() > 3:
            outline = 'green'
        else:
            outline = 'grey'
        self.cv.create_oval(x, y, x + EnvRenderer.square_size, y + EnvRenderer.square_size, fill=fill, outline=outline,
                            width=width)
        text_fill = 'grey' if player.state.used else 'black'
        self.cv.create_text(x + EnvRenderer.square_size / 2,
                            y + EnvRenderer.square_size / 2,
                            text=str(player.nr), fill=text_fill)

        # State
        if not player.state.up:
            self.cv.create_line(EnvRenderer.square_size * x, EnvRenderer.square_size * y + EnvRenderer.top_bar_height,
                                EnvRenderer.square_size * x + EnvRenderer.square_size,
                                EnvRenderer.square_size * y + EnvRenderer.square_size + EnvRenderer.top_bar_height, fill='white',
                                width=1)
        if player.state.stunned:
            self.cv.create_line(EnvRenderer.square_size * x + EnvRenderer.square_size,
                                EnvRenderer.square_size * y + EnvRenderer.top_bar_height,
                                EnvRenderer.square_size * x,
                                EnvRenderer.square_size * y + EnvRenderer.square_size + EnvRenderer.top_bar_height, fill='white',
                                width=1)
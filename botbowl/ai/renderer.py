import tkinter as tk
import math
from tkinter import *
from botbowl.ai.env import BotBowlEnv


class Renderer:

    square_size = 16
    square_size_fl = 4
    top_bar_height = 42
    bot_bar_height = 80
    layer_text_height = 26
    black = '#000000'
    white = '#ffffff'
    crowd = '#113311'
    blue = '#6666cc'
    red = '#cc6666'
    blue_endzone = '#2222cc'
    red_endzone = '#cc2222'
    ball = '#ff00cc'
    field = '#77cc77'
    wing = '#55aa55'
    scrimmage = '#338833'

    def __init__(self):
        self.roots = {}
        self.cvs = {}

    def _available_players(self, action_type):
        action = None
        for a in self.game.state.available_actions:
            if a.action_type == BotBowlEnv.actions[action_type]:
                action = a
        if action is None:
            return []
        return [player.position for player in action.players]

    def _draw_player(self, obs, idx, x, y, own=False):
        board_x = Renderer.square_size * x
        board_y = Renderer.square_size * y + Renderer.top_bar_height

        if own:
            fill = Renderer.blue
        else:
            fill = Renderer.red
        width = max(1, obs['board']['strength'][y][x] - 1)
        if obs['board']['block'][y][x]:
            outline = 'red'
        elif obs['board']['catch'][y][x]:
            outline = 'yellow'
        elif obs['board']['pass'][y][x]:
            outline = 'white'
        elif obs['board']['strength'][y][x] > 3:
            outline = 'green'
        else:
            outline = 'grey'
        self.cvs[idx].create_oval(board_x, board_y, board_x + Renderer.square_size, board_y + Renderer.square_size, fill=fill, outline=outline,
                            width=width)
        text_fill = 'grey' if not obs['board']['active players'][y][x] else 'black'
        self.cvs[idx].create_text(board_x + Renderer.square_size / 2,
                                  board_y + Renderer.square_size / 2,
                            fill=text_fill)

        # State
        if not obs['board']['standing players'][y][x]:
            self.cvs[idx].create_line(board_x, board_y,
                                      board_x + Renderer.square_size,
                                      board_y + Renderer.square_size, fill='black',
                                      width=2)
        if obs['board']['stunned players'][y][x]:
            self.cvs[idx].create_line(board_x + Renderer.square_size,
                                    board_y,
                                    board_x,
                                    board_y + Renderer.square_size, fill='black',
                                    width=2)

    def render(self, obs, idx, feature_layers=False):
        width = len(obs['board']['occupied'][0])
        height = len(obs['board']['occupied'])
        cv_width = width * Renderer.square_size
        cv_height = height * Renderer.square_size + Renderer.top_bar_height + Renderer.bot_bar_height
        if idx not in self.roots:
            self.roots[idx] = tk.Tk()
            self.roots[idx].title("botbowl Gym")
            if feature_layers:
                cols = math.floor(math.sqrt(len(obs['board'])))
                rows = math.ceil(math.sqrt(len(obs['board'])))
                fl_width = (width+1) * cols * Renderer.square_size_fl + Renderer.square_size_fl
                fl_height = ((height+1) * Renderer.square_size_fl + Renderer.layer_text_height) * rows + Renderer.square_size_fl
                self.cvs[idx] = tk.Canvas(width=max(350, max(cv_width, fl_width)), height=fl_height + cv_height, master=self.roots[idx])
            else:
                self.cvs[idx] = tk.Canvas(width=max(350, cv_width), height=cv_height, master=self.roots[idx])

        self.cvs[idx].pack(side='top', fill='both', expand='yes')
        self.cvs[idx].delete("all")
        self.roots[idx].configure(background='black')

        # Squares
        for y in range(height):
            for x in range(width):
                if y == height-1 or y == 0 or x == 0 or x == width-1:
                    fill = Renderer.crowd
                elif obs['board']['own touchdown'][y][x]:
                    fill = Renderer.blue_endzone
                elif obs['board']['opp touchdown'][y][x]:
                    fill = Renderer.red_endzone
                else:
                    fill = Renderer.field
                self.cvs[idx].create_rectangle(Renderer.square_size*x, Renderer.square_size*y + Renderer.top_bar_height, Renderer.square_size*x + Renderer.square_size, Renderer.square_size*y + Renderer.square_size + Renderer.top_bar_height, fill=fill, outline=Renderer.black)

        self.cvs[idx].create_line(width*Renderer.square_size/2.0-1, Renderer.top_bar_height, width*Renderer.square_size/2.0-1, height*Renderer.square_size + Renderer.top_bar_height, fill=Renderer.black, width=2)

        # Players
        for y in range(height):
            for x in range(width):
                own_player = obs["board"]["own players"][y][x]
                if own_player:
                    self._draw_player(obs, idx, x, y, own=True)
                else:
                    opp_player = obs["board"]["opp players"][y][x]
                    if opp_player:
                        self._draw_player(obs, idx, x, y, own=False)

        '''
        # Dugouts
        x = 4
        y = height*Renderer.square_size + Renderer.top_bar_height + 4
        for player in self.game.get_reserves(self.game.state.away_team):
            self._draw_player(player, x, y)
            x += Renderer.square_size
        x = 4
        y += Renderer.square_size
        for player in self.game.get_kods(self.game.state.away_team):
            self._draw_player(player, x, y)
            x += Renderer.square_size
        x = 4
        y += Renderer.square_size
        for player in self.game.get_casualties(self.game.state.away_team):
            self._draw_player(player, x, y)
            x += Renderer.square_size
        x = 4
        y += Renderer.square_size
        for player in self.game.get_dungeon(self.game.state.away_team):
            self._draw_player(player, x, y)
            x += Renderer.square_size

        x = width*Renderer.square_size - Renderer.square_size
        y = height * Renderer.square_size + Renderer.top_bar_height + 4
        for player in self.game.get_reserves(self.game.state.home_team):
            self._draw_player(player, x, y)
            x -= Renderer.square_size
        x = width * Renderer.square_size - Renderer.square_size
        y += Renderer.square_size
        for player in self.game.get_kods(self.game.state.home_team):
            self._draw_player(player, x, y)
            x -= Renderer.square_size
        x = width * Renderer.square_size - Renderer.square_size
        y += Renderer.square_size
        for player in self.game.get_casualties(self.game.state.home_team):
            self._draw_player(player, x, y)
            x -= Renderer.square_size
        x = width * Renderer.square_size - Renderer.square_size
        y += Renderer.square_size
        for player in self.game.get_dungeon(self.game.state.home_team):
            self._draw_player(player, x, y)
            x -= Renderer.square_size
        '''

        # Ball
        for y in range(height):
            for x in range(width):
                if obs['board']['balls'][y][x]:
                    self.cvs[idx].create_oval(Renderer.square_size * x + Renderer.square_size/4,
                                        Renderer.square_size * y + Renderer.square_size/4 + Renderer.top_bar_height,
                                        Renderer.square_size * x + Renderer.square_size - Renderer.square_size/4,
                                        Renderer.square_size * y + Renderer.square_size - Renderer.square_size/4 + Renderer.top_bar_height,
                                        fill=Renderer.ball, outline=Renderer.black, width=1)

        # Non-spatial
        weather = "Nice"
        if obs['state']['is sweltering heat']:
            weather = "Sweltering heat"
        elif obs['state']['is nice']:
            weather = "Pouring rain"
        elif obs['state']['is very sunny']:
            weather = "Very sunny"
        elif obs['state']['is blizzard']:
            weather = "Blizzard"

        self.cvs[idx].create_text(10, 10, text='Half: {}, Weather: {}'.format(int(obs['state']['half'])+1, weather), fill='black', anchor=W)
        self.cvs[idx].create_text(10, 34, text='TD: {}, Turn: {}, RR: {}/{}, Bribes: {}'.format(
            int(obs['state']['own score'] * 16),
            int(obs['state']['own turns'] * 8) + 1,
            int(obs['state']['own starting rerolls'] * 8),
            int(obs['state']['own rerolls left'] * 8),
            int(obs['state']['own bribes'] * 8)), fill='blue', anchor=W)
        self.cvs[idx].create_text(10, 22,
                            text='TD: {}, Turn: {}, RR: {}/{}, Bribes: {}'.format(
                                int(obs['state']['opp score'] * 16),
                                int(obs['state']['opp turns'] * 8) + 1,
                                int(obs['state']['opp starting rerolls'] * 8),
                                int(obs['state']['opp rerolls left'] * 8),
                                int(obs['state']['opp bribes'] * 8)), fill='red', anchor=W)
        '''
        # Feature layers
        if feature_layers:
            row = 0
            col = 0
            for name, grid in self.last_obs['board'].items():
                grid_x = col * (len(grid[0]) + 1) * Renderer.square_size_fl + Renderer.square_size_fl
                grid_y = row * (len(grid) + 1) * Renderer.square_size_fl + self.cv_height + Renderer.square_size_fl + ((row+1) * Renderer.layer_text_height)

                self.cvs[idx].create_text(grid_x + (len(grid[0]) * Renderer.square_size_fl)/3, grid_y - Renderer.layer_text_height/3, text=name)
                self.cvs[idx].create_rectangle(grid_x,
                                         grid_y,
                                         grid_x + len(grid[0]) * Renderer.square_size_fl,
                                         grid_y + len(grid) * Renderer.square_size_fl,
                                         fill='black', outline=Renderer.black, width=3)
                for y in range(len(grid)):
                    for x in range(len(grid[0])):
                        value = 1 - grid[y][x]
                        fill = '#%02x%02x%02x' % (int(value * 255), int(value * 255), int(value * 255))
                        self.cvs[idx].create_rectangle(Renderer.square_size_fl * x + grid_x,
                                                 Renderer.square_size_fl * y + grid_y,
                                                 Renderer.square_size_fl * x + grid_x + Renderer.square_size_fl,
                                                 Renderer.square_size_fl * y + grid_y + Renderer.square_size_fl,
                                                 fill=fill, outline=Renderer.black)
                col += 1
                if col >= self.cols:
                    col = 0
                    row += 1

        '''
        self.roots[idx].update_idletasks()
        self.roots[idx].update()

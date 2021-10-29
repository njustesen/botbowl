"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains the Host class that is used to manage games in memory. 
A similar host could be implemented that uses a persistent database instead.
"""

from botbowl.core.util import *
from botbowl.core.model import Replay
import pickle
import glob
import uuid


class Save:

    def __init__(self, game, team_id):
        self.game = game
        game.pause_clocks()
        self.team_id = team_id

    def to_json(self):
        return {
            'game': self.game.to_json(),
            'team_id': self.team_id
        }


class InMemoryHost:

    def __init__(self):
        self.games = {}

    def add_game(self, game):
        self.games[game.game_id] = game

    def end_game(self, id):
        del self.games[id]

    def get_game(self, id):
        return self.games[id]

    def get_games(self):
        return list(self.games.values())

    def save_game(self, game_id, name, team_id):
        game = self.get_game(game_id)
        filename = os.path.join(get_data_path("saves/"), name+".botbowl")
        print("Saving game")
        pickle.dump(game, open(filename, "wb"))
        game_clone = pickle.load(open(filename, "rb"))
        game_clone.game_id = str(uuid.uuid1())
        save = Save(game, team_id=team_id)
        pickle.dump(save, open(filename, "wb"))
        print("Game saved")

    # loads Save object from file
    def load_file(self, filename):
        print("Loading game")
        save = pickle.load(open(filename, "rb"))
        print("Game loaded")
        return save

    # creates Game from Save
    def load_game(self, name):
        save = self.load_file(get_data_path("saves/" + name.lower() + ".botbowl"))
        self.games[save.game.game_id] = save.game
        return save

    def get_savenames(self):
        files = glob.glob(get_data_path("saves/*"))
        out = []
        for file in files:
            file = file.lower()
            if ".botbowl" not in file:
                continue
            file = os.path.split(file)[1].split(".")[0]
            out.append(file)
        return out

    def get_saved_games(self):
        games = [self.load_file(filename) for filename in glob.glob(get_data_path("saves/*"))]
        return zip(games, self.get_savenames())

    def get_replay_ids(self):
        replays = [filename.split('/')[-1].split('.rep')[0] for filename in glob.glob(get_data_path("replays") + "/*.rep")]
        return sorted(replays)

    def load_replay(self, replay_id):
        return Replay(replay_id)
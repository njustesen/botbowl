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

    def save_game(self, game_id, name):
        game = self.get_game(game_id)
        data_path = get_data_path("saves/")
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        filename = os.path.join(data_path, name+".bb")
        print("Saving game")
        game.pause_clocks()
        pickle.dump(game, open(filename, "wb"))
        game.resume_clocks()

    def delete_saved_game(self, name):
        data_path = get_data_path("saves/")
        filename = os.path.join(data_path, name + ".bb")
        if os.path.exists(filename):
            os.remove(filename)

    # loads Save object from file
    def load_file(self, filename):
        print("Loading game")
        save = pickle.load(open(filename, "rb"))
        print("Game loaded")
        return save

    def load_game(self, name):
        game = self.load_file(get_data_path("saves/" + name.lower() + ".bb"))
        game.game_id = str(uuid.uuid1())
        self.games[game.game_id] = game
        game.resume_clocks()
        return game

    def get_savenames(self):
        files = glob.glob(get_data_path("saves/*.bb"))
        out = []
        for file in files:
            file = file.lower()
            file = os.path.split(file)[1].split(".")[0]
            out.append(file)
        return out

    def get_saved_games(self):
        return [(os.path.basename(filename).replace(".bb", ""), self.load_file(filename)) for filename in glob.glob(get_data_path("saves/*.bb"))]

    def get_replay_ids(self):
        replays = [filename.split('/')[-1].split('.rep')[0] for filename in glob.glob(get_data_path("replays") + "/*.rep")]
        return sorted(replays)

    def load_replay(self, replay_id):
        return Replay(replay_id)
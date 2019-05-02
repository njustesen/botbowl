#!/usr/bin/env python3

from ffai.web.api import *
import numpy as np
import time


class MyRandomBot(Agent):

    mean_actions_available = []
    steps = []

    def __init__(self, name):
        super().__init__(name)
        self.my_team = None
        self.actions_available = []

    def new_game(self, game, team):
        self.my_team = team
        self.actions_available = []
        self.actions_taken = 0

    def act(self, game):
        # Get time left in seconds to return an action
        seconds_left = game.seconds_left(self.my_team)
        available = 0
        for action_choice in game.state.available_actions:
            if len(action_choice.positions) == 0 and len(action_choice.players) == 0:
                available += 1
            elif len(action_choice.positions) > 0:
                available += len(action_choice.positions)
            else:
                available += len(action_choice.players)
        self.actions_available.append(available)
        
        # Select random action type - but no place player
        while True:
            action_choice = np.random.choice(game.state.available_actions)
            if action_choice.action_type != ActionType.PLACE_PLAYER:
                break

        # Select random position and player
        pos = np.random.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
        player = np.random.choice(action_choice.players) if len(action_choice.players) > 0 else None
        action = Action(action_choice.action_type, pos=pos, player=player)
        self.actions_taken += 1

        return action

    def end_game(self, game):
        print("Num steps:", len(self.actions_available))
        print("Avg. branching factor:", np.mean(self.actions_available))
        MyRandomBot.steps.append(len(self.actions_available))
        MyRandomBot.mean_actions_available.append(np.mean(self.actions_available))
        winner = game.get_winner()
        print("Casualties: ", game.num_casualties())
        if winner is None:
            print("It's a draw")
        elif winner == self:
            print("I ({}) won".format(self.name))
        else:
            print("I ({}) lost".format(self.name))
        print("I took", self.actions_taken, "actions")


if __name__ == "__main__":

    # Load configurations, rules, arena and teams
    config = get_config("ff-11-bot-bowl-i.json")
    ruleset = get_rule_set(config.ruleset, all_rules=False)  # We don't need all the rules
    arena = get_arena(config.arena)
    home = get_team_by_id("human-1", ruleset)
    away = get_team_by_id("human-2", ruleset)
    #config.competition_mode = False

    # Play 100 games
    game_times = []
    for i in range(100):
        away_agent = MyRandomBot("Random Bot 1")
        home_agent = MyRandomBot("Random Bot 2")
        config.debug_mode = False
        game = Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i+1))
        start = time.time()
        game.init()
        end = time.time()
        game_time = (end - start)
        game_times.append(game_time)
        print(game_time)

    print("Avg. num. steps:", np.mean(MyRandomBot.steps))
    print("Avg. branching factor:", np.mean(MyRandomBot.mean_actions_available))
    print("Avg. game time:", np.mean(game_times))

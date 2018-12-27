from bb.web.api import *
import numpy as np
import time


class MyRandomBot(Agent):

    def __init__(self, name):
        super().__init__(name)
        self.my_team = None
        self.actions_taken = 0

    def new_game(self, game, team):
        self.my_team = team
        self.actions_taken = 0

    def act(self, game):
        while True:
            action_choice = np.random.choice(game.state.available_actions)
            if action_choice.action_type != ActionType.PLACE_PLAYER:
                break
        pos = np.random.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
        player = np.random.choice(action_choice.players) if len(action_choice.players) > 0 else None
        action = Action(action_choice.action_type, pos=pos, player=player)
        self.actions_taken += 1
        return action

    def end_game(self, game):
        winner = game.get_winner()
        print("Casualties: ", game.num_casualties())
        if winner is None:
            print("It's a draw")
        elif winner == self.my_team:
            print("I ({}) won".format(self.name))
        else:
            print("I ({}) lost".format(self.name))
        print("I took", self.actions_taken, "actions")


if __name__ == "__main__":

    # Load configurations, rules, arena and teams
    config = get_config("ff-11.json")
    # config = get_config("ff-7.json")
    # config = get_config("ff-5.json")
    # config = get_config("ff-3.json")
    ruleset = get_rule_set(config.ruleset, all_rules=False)  # We don't need all the rules
    arena = get_arena(config.arena)
    home = get_team_by_id("human-1", ruleset)
    away = get_team_by_id("human-2", ruleset)

    # Play 100 games
    for i in range(100):
        away_agent = MyRandomBot("Random Bot 1")
        home_agent = MyRandomBot("Random Bot 2")
        config.debug_mode = False
        game = Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i+1))
        start = time.time()
        game.init()
        game.step()
        end = time.time()
        print(end - start)

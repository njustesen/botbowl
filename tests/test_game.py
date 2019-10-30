import pytest
from multiprocessing import Process, Pipe
from ffai.ai.registry import register_bot, make_bot
from ffai.core.game import *
from ffai.ai.bots.random_bot import RandomBot
from copy import deepcopy
import random

games = {}


class StateCollectorBot(Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.rnd = np.random.RandomState(seed)

    def new_game(self, game, team):
        self.my_team = team

    def act(self, game):

        proc = game.state.stack.peek()
        if type(proc) not in games.keys():
            games[type(proc)] = game.safe_clone()

        # Select a random action type
        while True:
            action_choice = self.rnd.choice(game.state.available_actions)
            # Ignore PLACE_PLAYER actions
            if action_choice.action_type != ActionType.PLACE_PLAYER:
                break

        # Select a random position and/or player
        pos = self.rnd.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
        player = self.rnd.choice(action_choice.players) if len(action_choice.players) > 0 else None

        # Make action object
        action = Action(action_choice.action_type, pos=pos, player=player)

        # Return action to the framework
        return action

    def end_game(self, game):
        pass


procs = [CoinTossFlip, CoinTossKickReceive, Setup, PlaceBall, HighKick, Touchback, Turn, PlayerAction]

while len(games.keys()) < len(procs):
    config = get_config("ff-11")
    ruleset = get_rule_set(config.ruleset)
    home = get_team_by_filename("human", ruleset)
    away = get_team_by_filename("human", ruleset)
    away_agent = StateCollectorBot("bot1", 0)
    home_agent = StateCollectorBot("bot2", 0)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.init()


def get_game(procedure):
    game = games[procedure].safe_clone()
    game.set_home_agent(Agent("human1", human=True))
    game.set_away_agent(Agent("human2", human=True))
    return game


def has_report_of_type(game, outcome_type):
    for report in game.state.reports:
        if report.outcome_type == outcome_type:
            return True
    return False


@pytest.mark.parametrize("action_type", [ActionType.HEADS, ActionType.TAILS])
def test_coin_toss(action_type):
    coverage = set()
    while len(coverage) < 2:
        game = get_game(CoinTossFlip)
        game.set_seed(random.randrange(0, 10000000))
        actor_id = game.actor.agent_id
        game.step(Action(action_type))
        if action_type == ActionType.HEADS:
            if has_report_of_type(game, OutcomeType.HEADS_WON):
                proc = game.state.stack.peek()
                coverage.add(OutcomeType.HEADS_WON)
                assert game.actor.agent_id == actor_id
                assert type(proc) == CoinTossKickReceive
            elif has_report_of_type(game, OutcomeType.TAILS_LOSS):
                proc = game.state.stack.peek()
                coverage.add(OutcomeType.HEADS_LOSS)
                assert game.actor.agent_id != actor_id
                assert type(proc) == CoinTossKickReceive
            else:
                assert False
        elif action_type == ActionType.TAILS:
            if has_report_of_type(game, OutcomeType.TAILS_WON):
                proc = game.state.stack.peek()
                coverage.add(OutcomeType.TAILS_WON)
                assert game.actor.agent_id == actor_id
                assert type(proc) == CoinTossKickReceive
            elif has_report_of_type(game, OutcomeType.HEADS_LOSS):
                proc = game.state.stack.peek()
                coverage.add(OutcomeType.TAILS_LOSS)
                assert game.actor.agent_id != actor_id
                assert type(proc) == CoinTossKickReceive
            else:
                assert False


#if __name__ == "__main__":
#    test_coin_toss(ActionType.HEADS)
#    test_coin_toss(ActionType.TAILS)
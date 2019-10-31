import pytest
from multiprocessing import Process, Pipe
from ffai.ai.registry import register_bot, make_bot
from ffai.core.game import *
from ffai.ai.bots.random_bot import RandomBot
from copy import deepcopy
import random


def get_game(seed=0):
    config = get_config("ff-11")
    ruleset = get_rule_set(config.ruleset)
    home = get_team_by_filename("human", ruleset)
    away = get_team_by_filename("human", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    game.init()
    return game


def has_report_of_type(game, outcome_type):
    for report in game.state.reports:
        if report.outcome_type == outcome_type:
            return True
    return False


@pytest.mark.parametrize("action_type", [ActionType.HEADS, ActionType.TAILS])
def test_coin_toss(action_type):
    coverage = set()
    for i in range(100):
        if len(coverage) == 2:
            return
        game = get_game()
        game.step(Action(ActionType.START_GAME))
        proc = game.state.stack.peek()
        assert type(proc) is CoinTossFlip
        actor_id = game.actor.agent_id
        game.set_seed(i)
        game.step(Action(action_type))
        proc = game.state.stack.peek()
        assert type(proc) is CoinTossKickReceive
        if action_type == ActionType.HEADS:
            if has_report_of_type(game, OutcomeType.HEADS_WON):
                coverage.add(OutcomeType.HEADS_WON)
                assert game.actor.agent_id == actor_id
            elif has_report_of_type(game, OutcomeType.TAILS_LOSS):
                coverage.add(OutcomeType.TAILS_LOSS)
                assert game.actor.agent_id != actor_id
            else:
                assert False
        elif action_type == ActionType.TAILS:
            if has_report_of_type(game, OutcomeType.TAILS_WON):
                coverage.add(OutcomeType.TAILS_WON)
                assert game.actor.agent_id == actor_id
            elif has_report_of_type(game, OutcomeType.HEADS_LOSS):
                coverage.add(OutcomeType.HEADS_LOSS)
                assert game.actor.agent_id != actor_id
            else:
                assert False
        else:
            assert False
    assert False


@pytest.mark.parametrize("action_type", [ActionType.KICK, ActionType.RECEIVE])
def test_kick_receive(action_type):
    actors = set()
    for i in range(100):
        if len(actors) == 2:
            return
        game = get_game()
        game.set_seed(i)
        game.step(Action(ActionType.START_GAME))
        game.step(Action(ActionType.HEADS))
        proc = game.state.stack.peek()
        assert type(proc) is CoinTossKickReceive
        selector = game.actor
        actors.add(game.home_agent == selector)
        game.step(Action(action_type))
        proc = game.state.stack.peek()
        assert type(proc) is Setup
        if action_type == ActionType.KICK:
            assert game.actor == selector
        elif action_type == ActionType.RECEIVE:
            assert game.actor != selector
        else:
            assert False
    assert False


if __name__ == "__main__":
    test_coin_toss(ActionType.HEADS)
    test_coin_toss(ActionType.TAILS)
    test_kick_receive(ActionType.KICK)
    test_kick_receive(ActionType.RECEIVE)

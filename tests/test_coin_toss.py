import pytest
from ffai.core.game import *


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
        acting_team = game.agent_team(game.actor)
        game.set_seed(i)
        game.step(Action(action_type))
        proc = game.state.stack.peek()
        assert type(proc) is CoinTossKickReceive
        if action_type == ActionType.HEADS:
            if game.has_report_of_type(OutcomeType.HEADS_WON):
                coverage.add(OutcomeType.HEADS_WON)
                assert game.actor.agent_id == actor_id
                assert game.state.coin_toss_winner == acting_team
            elif game.has_report_of_type(OutcomeType.TAILS_LOSS):
                coverage.add(OutcomeType.TAILS_LOSS)
                assert game.actor.agent_id != actor_id
                assert game.state.coin_toss_winner != acting_team
            else:
                assert False
        elif action_type == ActionType.TAILS:
            if game.has_report_of_type(OutcomeType.TAILS_WON):
                coverage.add(OutcomeType.TAILS_WON)
                assert game.actor.agent_id == actor_id
                assert game.state.coin_toss_winner == acting_team
            elif game.has_report_of_type(OutcomeType.HEADS_LOSS):
                coverage.add(OutcomeType.HEADS_LOSS)
                assert game.actor.agent_id != actor_id
                assert game.state.coin_toss_winner != acting_team
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
        selecting_team = game.agent_team(game.actor)
        actors.add(game.home_agent == selector)
        game.step(Action(action_type))
        proc = game.state.stack.peek()
        assert type(proc) is Setup
        if action_type == ActionType.KICK:
            assert game.has_report_of_type(
                OutcomeType.AWAY_RECEIVE) if game.home_agent == selector else game.has_report_of_type(
                OutcomeType.HOME_RECEIVE)
            assert game.actor == selector
            assert game.state.kicking_first_half == selecting_team
            assert game.state.kicking_this_drive == selecting_team
        elif action_type == ActionType.RECEIVE:
            assert game.has_report_of_type(
                OutcomeType.HOME_RECEIVE) if game.home_agent == selector else game.has_report_of_type(
                OutcomeType.AWAY_RECEIVE)
            assert game.actor != selector
            assert game.state.kicking_first_half != selecting_team
            assert game.state.kicking_this_drive != selecting_team
        else:
            assert False
    assert False


if __name__ == "__main__":
    test_coin_toss(ActionType.HEADS)
    test_coin_toss(ActionType.TAILS)
    test_kick_receive(ActionType.KICK)
    test_kick_receive(ActionType.RECEIVE)

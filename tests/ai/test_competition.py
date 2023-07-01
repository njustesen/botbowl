import botbowl
from botbowl.core.load import load_config, load_rule_set, load_team_by_filename

config = botbowl.load_config("bot-bowl")
config.competition_mode = False
config.pathfinding_enabled = False  # disabled for speed
ruleset = botbowl.load_rule_set(
    config.ruleset, all_rules=False
)  # We don't need all the rules
arena = botbowl.load_arena(config.arena)
home = botbowl.load_team_by_filename("human", ruleset)
away = botbowl.load_team_by_filename("human", ruleset)


def test_random_comp():
    home_agent = botbowl.make_bot("random")
    away_agent = botbowl.make_bot("random")

    comp = botbowl.Competition(
        home_agent, away_agent, home, away, config, ruleset, arena
    )
    comp.run()


def test_illegal_actions(capsys):
    home_agent = botbowl.make_bot("random")
    away_agent = botbowl.ai.bots.IllegalActionBot("illegal")

    comp = botbowl.Competition(
        home_agent, away_agent, home, away, config, ruleset, arena
    )
    comp.run()

    out, err = capsys.readouterr()
    assert err == ""
    assert out.find("Action not allowed {'action_type': 'USE_APOTHECARY'") >= 0


def test_hide_agent_and_rng():
    game = botbowl.Game(
        1,
        home,
        away,
        home_agent=botbowl.make_bot("random"),
        away_agent=botbowl.make_bot("random"),
        config=load_config("gym-11"),
    )

    rng = game.rng
    home_agent = game.home_agent
    away_agent = game.away_agent

    with game.hide_agents_and_rng():
        assert game.home_agent is not home_agent
        assert game.away_agent is not away_agent
        assert game.rng is not rng

    assert game.home_agent is home_agent
    assert game.away_agent is away_agent
    assert game.rng is rng

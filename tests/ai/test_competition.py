import botbowl

config = botbowl.load_config("bot-bowl-iii")
config.competition_mode = False
config.pathfinding_enabled = False # disabled for speed
ruleset = botbowl.load_rule_set(config.ruleset, all_rules=False)  # We don't need all the rules
arena = botbowl.load_arena(config.arena)
home = botbowl.load_team_by_filename("human", ruleset)
away = botbowl.load_team_by_filename("human", ruleset)


def test_random_comp():
    home_agent = botbowl.make_bot("random")
    away_agent = botbowl.make_bot("random")

    comp = botbowl.Competition(home_agent, away_agent, home, away, config, ruleset, arena)
    comp.run()


def test_illegal_actions(capsys):
    home_agent = botbowl.make_bot("random")
    away_agent = botbowl.ai.bots.IllegalActionBot("illegal")

    comp = botbowl.Competition(home_agent, away_agent, home, away, config, ruleset, arena)
    comp.run()

    out, err = capsys.readouterr()
    assert err == ""
    assert out.find("Action not allowed {'action_type': 'USE_APOTHECARY'") >= 0

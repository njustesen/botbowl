import ffai

config = ffai.load_config("bot-bowl-iii")
config.competition_mode = False
config.pathfinding_enabled = False # disabled for speed
ruleset = ffai.load_rule_set(config.ruleset, all_rules=False)  # We don't need all the rules
arena = ffai.load_arena(config.arena)
home = ffai.load_team_by_filename("human", ruleset)
away = ffai.load_team_by_filename("human", ruleset)


def test_random_comp():
    home_agent = ffai.make_bot("random")
    away_agent = ffai.make_bot("random")

    comp = ffai.Competition(home_agent, away_agent, home, away, config, ruleset, arena)
    comp.run()


def test_illegal_actions(capsys):
    home_agent = ffai.make_bot("random")
    away_agent = ffai.ai.bots.IllegalActionBot("illegal")

    comp = ffai.Competition(home_agent, away_agent, home, away, config, ruleset, arena)
    comp.run()

    out, err = capsys.readouterr()
    assert err == ""
    assert out.find("Action not allowed {'action_type': 'USE_APOTHECARY'") >= 0

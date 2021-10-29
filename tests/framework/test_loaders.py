import pytest
from botbowl.core.load import *


def test_rule_loader():
    rulesetBB2016 = load_rule_set("BB2016")
    rulesetLRB5Experimental = load_rule_set("LRB5-Experimental")
    assert rulesetBB2016.name == "BB2016"
    assert rulesetLRB5Experimental.name == "LRB5-Experimental"
    assert len(rulesetBB2016.races) > 1
    assert len(rulesetLRB5Experimental.races) > 1
    assert len(rulesetBB2016.races[0].roles) > 1
    assert len(rulesetLRB5Experimental.races[0].roles) > 1
    assert rulesetBB2016.races[0].roles[0].ma > 0
    assert rulesetLRB5Experimental.races[0].roles[0].ma > 0


def test_config_loader():
    config_gym11 = load_config("gym-11")
    config_bot_bowl_ii = load_config("bot-bowl-ii")
    assert config_gym11.name == "botbowl"
    assert config_bot_bowl_ii.name == "Bot Bowl II"
    assert config_gym11.roster_size == 16
    assert config_bot_bowl_ii.roster_size == 16


def test_team_loader():
    rulesetBB2016 = load_rule_set("BB2016")
    human_team_2016 = load_team_by_name("Human Team", rulesetBB2016)
    assert len(human_team_2016.players) > 0
    assert human_team_2016.players[0].get_ma() > 0
    rulesetExperimental = load_rule_set("LRB5-Experimental")
    human_team_exp = load_team_by_name("Human Team", rulesetExperimental)
    assert len(human_team_exp.players) > 0
    assert human_team_exp.players[0].get_ma() > 0
    human_team_exp_2 = load_team_by_filename("human", rulesetExperimental)
    assert human_team_exp_2.name == human_team_exp_2.name
    # Teams are assigned new ids when loaded
    assert human_team_exp_2.team_id != human_team_exp.team_id
    assert human_team_2016.team_id != human_team_exp.team_id


def test_arena_loader():
    arena = load_arena("ff-pitch-1")
    arena2 = load_arena("ff-pitch-1.txt")
    assert arena.height == arena2.height
    pitch = load_arena("ff-pitch-11")
    assert pitch.width == 26 + 2
    assert pitch.height == 15 + 2


def test_formation_loader():
    def_spread = load_formation("def_spread")
    def_zone = load_formation("def_zone")
    off_line = load_formation("off_line")
    off_wedge = load_formation("off_wedge")
    assert def_spread.name == "Spread"
    assert def_zone.name == "Zone"
    assert off_line.name == "Line"
    assert off_wedge.name == "Wedge"


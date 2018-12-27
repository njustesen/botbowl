import os
import numpy as np
import untangle
from .model import *
import json
from bb.core.util import *
import glob
import bb
import pkg_resources

arena_char_map = {
    'C': Tile.CROWD,
    'A': Tile.AWAY,
    'h': Tile.HOME,
    'S': Tile.AWAY_SCRIMMAGE,
    's': Tile.HOME_SCRIMMAGE,
    'L': Tile.AWAY_WING_LEFT,
    'R': Tile.AWAY_WING_RIGHT,
    'l': Tile.HOME_WING_LEFT,
    'r': Tile.HOME_WING_RIGHT,
    'E': Tile.AWAY_TOUCHDOWN,
    'e': Tile.HOME_TOUCHDOWN
}


def parse_sc(sc):

    parsed = []
    for cat in sc:
        if cat == "G":
            parsed.append(SkillCategory.General)
        elif cat == "A":
            parsed.append(SkillCategory.Agility)
        elif cat == "S":
            parsed.append(SkillCategory.Strength)
        elif cat == "P":
            parsed.append(SkillCategory.Passing)
        elif cat == "M":
            parsed.append(SkillCategory.Mutation)
        elif cat == "E":
            parsed.append(SkillCategory.Extraordinary)
    return parsed


def get_rule_set(name, debug=False, all_rules=True):

    path = get_data_path('rules/' + name)

    if debug:
        print("Loading rules at " + path)
    obj = untangle.parse(path)

    ruleset = RuleSet(path.split("/")[-1].split(".")[0])

    if debug:
        print("Parsing races")
    for r in obj.rules.rosters.roster:
        if debug:
            print("-- Parsing " + str(r.name.cdata))
        race = Race(r.name.cdata, [], (int)(r.rerollValue.cdata), (bool)(r.apothecary.cdata), (bool)(r.stakes.cdata))
        for p in r.positions.position:
            position = Role(p.title.cdata, [race.name], (int)(p.ma.cdata), (int)(p.st.cdata), (int)(p.ag.cdata), (int)(p.av.cdata), [], (int)(p.cost.cdata), parse_sc(p.normal.cdata), parse_sc(p.double.cdata))
            if len(p.skills) > 0:
                for skill_name in p.skills.skill:
                    position.skills.append(parse_enum(Skill, skill_name.cdata))
            race.roles.append(position)
        ruleset.races.append(race)

    if all_rules:
        if debug:
            print("Parsing star players")
        for star in obj.rules.stars.star:
            if debug:
                print("-- Parsing " + str(star.name.cdata))
            role = Role(star.name.cdata, [], (int)(star.ma.cdata), (int)(star.st.cdata), (int)(star.ag.cdata), (int)(star.av.cdata), [], (int)(star.cost.cdata), (bool)(star.feeder.cdata), [], [], star_player=True)
            if len(star.skills) == 0:
                continue
            for skill_name in star.skills.skill:
                role.skills.append(parse_enum(Skill, skill_name.cdata))
            for race_name in star.races.race:
                role.races.append(race_name.cdata)
            ruleset.star_players.append(role)

        if debug:
            print("Parsing inducements")
        for i in obj.rules.inducements.inducement:
            if debug:
                print("-- Parsing " + str(i["name"]))
            reduced = 0 if not "reduced" in i else i["reduced"]
            inducement = Inducement(i["name"], (int)(i.cdata), (int)(i["max"]), reduced=reduced)
            ruleset.inducements.append(inducement)

        if debug:
            print("Parsing SPP actions")
        for a in obj.rules.spp.action:
            if debug:
                print("-- Parsing " + str(a["name"]))
            ruleset.spp_actions[a["name"]] = (int)(a.cdata)

        if debug:
            print("Parsing SPP levels")
        for l in obj.rules.spp.level:
            if debug:
                print("-- Parsing " + str(l["name"]))
            ruleset.spp_levels[l["name"]] = (int)(l.cdata)

        if debug:
            print("Parsing improvements")
        for imp in obj.rules.improvements.improvement:
            if debug:
                print("-- Parsing " + str(imp["name"]))
            ruleset.improvements[imp["name"]] = (int)(imp.cdata)

        if debug:
            print("Parsing spiralling expenses")
        ruleset.se_start = (int)(obj.rules.spirallingExpenses.start.cdata)
        ruleset.se_interval = (int)(obj.rules.spirallingExpenses.interval.cdata)
        ruleset.se_pace = (int)(obj.rules.spirallingExpenses.pace.cdata)

    if debug:
        print("Done loading rules")

    return ruleset


def get_all_teams(ruleset):
    path = get_data_path('teams/')
    teams = []
    for file in list(glob.glob(path + '/*.json')):
        name = file.split("/")[-1].split(".")[0]
        teams.append(get_team(name, ruleset))
    return teams


def get_team_by_id(team_id, ruleset):
    path = get_data_path('teams/')
    for file in list(glob.glob(path + '/*.json')):
        name = file.split("/")[-1].split(".")[0]
        team = get_team(name, ruleset)
        if team.team_id == team_id:
            return team
    return None


def get_team(name, ruleset):
    path = get_data_path('teams/' + name + '.json')
    f = open(path)
    str = f.read()
    f.close()
    data = json.loads(str)
    coach = Coach(data['coach']['id'], data['coach']['name'])
    team = Team(data['id'], data['name'], data['race'], players=[], coach=coach, treasury=data['treasury'], apothecary=data['apothecary'], rerolls=data['rerolls'], ass_coaches=data['ass_coaches'], cheerleaders=data['cheerleaders'], fan_factor=data['fan_factor'])
    for p in data['players']:
        role = ruleset.get_role(p['position'], team.race)
        player = Player(player_id=p['id'], role=role, name=p['name'], nr=p['nr'], niggling=p['niggling'], extra_ma=p['extra_ma'], extra_st=p['extra_st'], extra_ag=p['extra_ag'], extra_av=p['extra_av'], mng=p['mng'], spp=p['spp'], team=team)
        for s in p['extra_skills']:
            player.extra_skills.append(parse_enum(Skill, s))
        team.players.append(player)
    return team


def get_arena(name):
    path = get_data_path('arenas/' + name)
    # name = 'Unknown arena'
    dungeon = False
    board = []
    file = open(path, 'r')
    while True:
        line = file.readline()
        if not line:
            break
        row = []
        for c in line:
            if c not in arena_char_map.keys():
                if c in ['\n']:
                    continue
                file.close()
                raise Exception("Unknown tile type " + c)
            row.append(arena_char_map[c])
        board.append(np.array(row))
    file.close()
    return TwoPlayerArena(np.array(board))


def get_config(name):
    path = get_data_path('config/' + name)
    f = open(path)
    str = f.read()
    f.close()
    data = json.loads(str)
    config = Configuration()
    config.name = data['name']
    config.arena = data['arena']
    config.ruleset = data['ruleset']
    config.dungeon = data['dungeon']
    config.roster_size = data['roster_size']
    config.pitch_max = data['pitch_max']
    config.pitch_min = data['pitch_min']
    config.scrimmage_min = data['scrimmage_min']
    config.wing_max = data['wing_max']
    config.rounds = data['turns']
    config.kick_off_table = data['kick_off_table']
    config.fast_mode = data['fast_mode']
    config.kick_scatter_dice = data['kick_scatter_dice']
    config.defensive_formations = [get_formation(formation, config.pitch_max) for formation in
                                   data['defensive_formations']]
    config.offensive_formations = [get_formation(formation, config.pitch_max) for formation in
                                   data['offensive_formations']]
    return config


def get_formation(name, size):
    path = get_data_path('formations/' + str(size) + "/" + name)
    board = []
    file = open(path, 'r')
    name = name.replace(".txt", "").replace("off_", "").replace("def_", "")
    while True:
        line = file.readline()
        if not line:
            break
        row = []
        for c in line:
            if c in ['\n']:
                continue
            row.append(c)
        board.append(np.array(row))
    file.close()
    return Formation(name, board)

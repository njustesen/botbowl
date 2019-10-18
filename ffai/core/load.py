"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains the functions used to load data in the /data/ folder.
"""

from ffai.core.model import *
import json
from ffai.core.util import *
import glob
import untangle
import uuid


# Tile set mapping
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
    """
    :param name: The name of the ruleset - this should match the filename of the .xml file to load in data/rules/ (without extension)
    :param debug:
    :param all_rules: If False, only a small set of the rules are loaded.
    :return: A ruleset loaded from .xml.
    """

    path = get_data_path('rules/' + name + '.xml')

    if debug:
        print("Loading rules at " + path)
    obj = untangle.parse(path)

    ruleset = RuleSet(os.path.split(path)[1].split(".")[0])

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


def get_all_teams(ruleset, board_size=11):
    """
    :param ruleset:
    :return: All the teams in data/teams/
    """
    path = get_data_path('teams/')
    teams = []
    for file in list(glob.glob(f'{path}/{board_size}/*json')):
        team_id = os.path.split(file)[1].split(".")[0]
        teams.append(get_team(team_id, ruleset))
    return teams


def get_team(team_id, ruleset, board_size=11):
    """
    :param team_id: identifier for the team / file
    :param ruleset:
    :return: The team with filename team)_id (without file extension).
    """
    # print(team_id)
    path = get_data_path(f'teams/{board_size}/{team_id}.json')
    f = open(path)
    jsonStr = f.read()
    f.close()
    data = json.loads(jsonStr)
    team = Team(team_id, data['name'], data['race'], players=[], treasury=data['treasury'], apothecary=data['apothecary'], rerolls=data['rerolls'], ass_coaches=data['ass_coaches'], cheerleaders=data['cheerleaders'], fan_factor=data['fan_factor'])
    for p in data['players']:
        role = ruleset.get_role(p['position'], team.race)
        player_id = str(uuid.uuid1())
        player = Player(player_id=player_id, role=role, name=p['name'], nr=p['nr'], niggling=p['niggling'], extra_ma=p['extra_ma'], extra_st=p['extra_st'], extra_ag=p['extra_ag'], extra_av=p['extra_av'], mng=p['mng'], spp=p['spp'], team=team)
        for s in p['extra_skills']:
            player.extra_skills.append(parse_enum(Skill, s))
        team.players.append(player)
    return team


def get_arena(name):
    """
    :param name: The filename to load.
    :return: The arena at data/arena/<name>
    """
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
    """
    :param name: the filename to load.
    :return: The configuration in data/config/<name>
    """
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
    config.debug_mode = data['debug_mode']
    config.competition_mode = data['competition_mode']
    config.kick_scatter_dice = data['kick_scatter_dice']
    config.defensive_formations = [get_formation(formation, config.pitch_max) for formation in
                                   data['defensive_formations']]
    config.offensive_formations = [get_formation(formation, config.pitch_max) for formation in
                                   data['offensive_formations']]
    game = None
    disqualification = None
    turn = None
    opp = None
    if data['time_limits'] is not None:
        game = data['time_limits']['game']
        turn = data['time_limits']['turn']
        secondary = data['time_limits']['secondary']
        disqualification = data['time_limits']['disqualification']
        init = data['time_limits']['init']
        end = data['time_limits']['end']
    config.time_limits = TimeLimits(game=game, turn=turn, secondary=secondary, disqualification=disqualification, init=init, end=end)
    return config


def get_formation(name, size):
    """
    :param name: the filename to load.
    :param size: The number of players on the pitch in the used FFAI variant.
    :return: The formation in data/formations/<size>/<name>
    """
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

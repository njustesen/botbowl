"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains the functions used to load data in the /data/ folder.
"""

from botbowl.core.model import *
import json
from botbowl.core.util import *
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


def load_rule_set(name, debug=False, all_rules=True):
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


def load_all_teams(ruleset, board_size=11):
    """
    :param ruleset:
    :return: All the teams in data/teams/
    """
    path = get_data_path('teams/')
    teams = []
    for filepath in list(glob.glob(f'{path}/{board_size}/*json')):
        teams.append(load_team(filepath, ruleset))
    return teams


def load_team_by_filename(name, ruleset, board_size=11):
    path = get_data_path('teams/')
    for filepath in list(glob.glob(f'{path}/{board_size}/*json')):
        if os.path.split(filepath)[1].split(".json")[0] == name:
            return load_team(filepath, ruleset)
    raise Exception("Team file not found.")


def load_team_by_name(name, ruleset, board_size=11):
    for team in load_all_teams(ruleset, board_size):
        if team.name == name:
            return team
    raise Exception(f"Team with name '{name}' not found.")


def load_team(path, ruleset):
    """
    :param path: path to team file name.
    :param ruleset:
    :return: The team with filename team)_id (without file extension).
    """
    #path = get_data_path(path)
    f = open(path)
    jsonStr = f.read()
    f.close()
    data = json.loads(jsonStr)
    team_id = str(uuid.uuid1())
    team = Team(team_id, data['name'], data['race'], players=[], treasury=data['treasury'], apothecaries=data['apothecaries'], rerolls=data['rerolls'], ass_coaches=data['ass_coaches'], cheerleaders=data['cheerleaders'], fan_factor=data['fan_factor'])
    for p in data['players']:
        role = ruleset.get_role(p['position'], team.race)
        player_id = str(uuid.uuid1())
        player = Player(player_id=player_id, role=role, name=p['name'], nr=p['nr'], team=team, niggling_injuries=p['niggling'], extra_ma=p['extra_ma'], extra_st=p['extra_st'], extra_ag=p['extra_ag'], extra_av=p['extra_av'], mng=p['mng'], spp=p['spp'])
        for s in p['extra_skills']:
            player.extra_skills.append(parse_enum(Skill, s))
        team.players.append(player)
    return team


def load_arena(name):
    """
    :param name: The filename to load.
    :return: The arena at data/arena/<name>
    """
    if not name.endswith(".txt"):
        name += ".txt"
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


def load_config(name):
    """
    :param name: the filename to load.
    :return: The configuration in data/config/<name>
    """
    if not name.endswith(".json"):
        name += ".json"
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
    config.defensive_formations = [load_formation(formation, size=config.pitch_max) for formation in
                                   data['defensive_formations']]
    config.offensive_formations = [load_formation(formation, size=config.pitch_max) for formation in
                                   data['offensive_formations']]
    game = None
    disqualification = None
    turn = None
    opp = None
    if data['time_limits'] is not None:
        turn = data['time_limits']['turn']
        secondary = data['time_limits']['secondary']
        init = data['time_limits']['init']
        end = data['time_limits']['end']
    config.time_limits = TimeLimits(turn=turn, secondary=secondary, init=init, end=end)
    config.pathfinding_enabled = False
    config.pathfinding_directly_to_adjacent = False
    if 'pathfinding' in data and data['pathfinding']['enabled']:
        config.pathfinding_enabled = True
        config.pathfinding_directly_to_adjacent = 'directly_to_adjacent' in data['pathfinding'] and data['pathfinding']['directly_to_adjacent']
    return config


def load_formation(name, directory=None, size=11):
    """
    :param name: the filename to load.
    :param path: path to a text file describing the setup formation. If None, the botbowl formation path will be used.
    :param size: The number of players on the pitch in the used botbowl variant.
    :return: The formation in data/formations/<size>/<name>
    """
    if not name.endswith(".txt"):
        name += ".txt"
    if directory is not None:
        path = os.path.join(directory, name)
    else:
        path = get_data_path('formations/' + str(size) + "/" + name)
    board = []
    name = name.replace(".txt", "").replace("off_", "").replace("def_", "").title()
    with open(path, 'r') as file_:
        for line in file_:
            if not line:
                break
            row = list(line.strip())
            board.append(np.array(row))
    return Formation(name, board)

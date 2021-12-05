from typing import List, Optional, Tuple, Union

from botbowl.core.game import *
from botbowl.ai.bots.random_bot import *
from copy import deepcopy

game_turn_empty = {}
game_turn_full = {}


def get_game_turn(seed=0, empty=False):
    D3.FixedRolls = []
    D6.FixedRolls = []
    D8.FixedRolls = []
    BBDie.FixedRolls = []
    if empty:
        if seed in game_turn_empty:
            return deepcopy(game_turn_empty[seed])
    else:
        if seed in game_turn_full:
            return deepcopy(game_turn_full[seed])
    config = load_config("gym-11")
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("orc", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    game.init()
    game.step(Action(ActionType.START_GAME))
    game.step(Action(ActionType.HEADS))
    game.step(Action(ActionType.KICK))
    game.step(Action(ActionType.SETUP_FORMATION_ZONE))
    game.step(Action(ActionType.END_SETUP))
    game.step(Action(ActionType.SETUP_FORMATION_WEDGE))
    game.step(Action(ActionType.END_SETUP))
    random_agent = RandomBot("home")
    while type(game.get_procedure()) is not Turn or game.is_quick_snap() or game.is_blitz():
        action = random_agent.act(game)
        game.step(action)
    if empty:
        game.clear_board()
    if empty:
        game_turn_empty[seed] = deepcopy(game)
    else:
        game_turn_full[seed] = deepcopy(game)
    return game


Position = Union[Square, Tuple[int, int]]


def get_custom_game_turn(player_positions: List[Position], opp_player_positions: Optional[List[Position]] = None,
                         ball_position: Optional[Position] = None, weather: WeatherType = WeatherType.NICE,
                         rerolls: int = 0, forward_model_enabled=False, pathfinding_enabled=False) \
        -> Tuple:
    """
    :param player_positions: places human linemen of active team in these squares
    :param opp_player_positions: places human linemen of not active team in these squares
    :param ball_position: places ball in this square.
    :param weather:
    :param rerolls: number of rerolls
    :param forward_model_enabled:
    :param pathfinding_enabled:
    :return: tuple with created game object followed by all the placed players
    """
    game = get_game_turn(empty=True)
    team = game.get_agent_team(game.actor)
    team_players = [player for player in team.players if player.role.name == "Lineman"]

    game.state.weather = weather
    game.state.teams[0].state.rerolls = rerolls

    def assert_square_type(obj: Position) -> Square:
        if type(obj) == Square:
            return obj
        else:
            return game.get_square(obj[0], obj[1])

    return_list = [game]

    for i, sq in enumerate(player_positions):
        player = team_players[i]
        game.put(player, assert_square_type(sq))
        return_list.append(player)

    if opp_player_positions is not None:
        opp_team_players = [player for player in game.get_opp_team(team).players if player.role.name == "Lineman"]
        for i, sq in enumerate(opp_player_positions):
            player = opp_team_players[i]
            game.put(player, assert_square_type(sq))
            return_list.append(player)

    if ball_position is not None:
        game.get_ball().move_to(assert_square_type(ball_position))
        game.get_ball().is_carried = game.get_player_at(assert_square_type(ball_position)) is not None

    game.config.pathfinding_enabled = pathfinding_enabled
    game.set_available_actions()
    game.state.reports.clear()

    if forward_model_enabled:
        game.enable_forward_model()

    return tuple(return_list)


def get_game_kickoff(seed=0):
    D3.FixedRolls = []
    D6.FixedRolls = []
    D8.FixedRolls = []
    BBDie.FixedRolls = []
    config = load_config("gym-11")
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("orc", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    game.init()
    game.step(Action(ActionType.START_GAME))
    game.step(Action(ActionType.HEADS))
    game.step(Action(ActionType.KICK))
    game.step(Action(ActionType.SETUP_FORMATION_ZONE))
    game.step(Action(ActionType.END_SETUP))
    game.step(Action(ActionType.SETUP_FORMATION_WEDGE))
    game.step(Action(ActionType.END_SETUP))
    return game


def get_game_setup(home_team, seed=0):
    D3.FixedRolls = []
    D6.FixedRolls = []
    D8.FixedRolls = []
    BBDie.FixedRolls = []
    config = load_config("gym-11")
    config.kick_off_table = False
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("human", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    game.init()
    game.step(Action(ActionType.START_GAME))
    game.step(Action(ActionType.HEADS))
    if game.actor == game.home_agent:
        if home_team:
            game.step(Action(ActionType.KICK))
        else:
            game.step(Action(ActionType.RECEIVE))
    if game.actor == game.away_agent:
        if home_team:
            game.step(Action(ActionType.RECEIVE))
        else:
            game.step(Action(ActionType.KICK))
    return game


def get_game_coin_toss(seed=0):
    D3.FixedRolls = []
    D6.FixedRolls = []
    D8.FixedRolls = []
    BBDie.FixedRolls = []
    config = load_config("gym-11")
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("human", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    game.init()
    return game


def get_game_fans(seed=0):
    D3.FixedRolls = []
    D6.FixedRolls = []
    D8.FixedRolls = []
    BBDie.FixedRolls = []
    config = load_config("gym-11")
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("orc", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    return game


def get_game_weather_table(seed=0):
    D3.FixedRolls = []
    D6.FixedRolls = []
    D8.FixedRolls = []
    BBDie.FixedRolls = []
    config = load_config("gym-11")
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("orc", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    #game.init()
    return game


def get_block_players(game, team):
    # get a player
    players = game.get_players_on_pitch(team, False, True)
    attacker = None
    defender = None
    for p in players:
        if p.team != team:
            continue
        attacker = p
        adjacent = game.get_adjacent_opponents(attacker)
        if len(adjacent) > 0:
            defender = adjacent[0]
            break
    return attacker, defender

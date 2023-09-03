from botbowl.core.game import *
from botbowl.ai.bots.random_bot import *
from copy import deepcopy

from typing import List, Optional, Tuple, Union, Iterable, Any
from contextlib import contextmanager

game_turn_empty = {}
game_turn_full = {}


def get_game_turn(seed=0, empty=False, home_team: str = 'human', away_team: str = 'orc', size: int = 11):
    D3.FixedRolls = []
    D6.FixedRolls = []
    D8.FixedRolls = []
    BBDie.FixedRolls = []

    key = f"{seed} {home_team} {away_team} {size}"
    if empty:
        if key in game_turn_empty:
            return deepcopy(game_turn_empty[key])
    else:
        if key in game_turn_full:
            return deepcopy(game_turn_full[key])
    config = load_config(f"gym-{size}")
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename(home_team, ruleset)
    away = load_team_by_filename(away_team, ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    game.init()
    game.step(Action(ActionType.START_GAME))
    game.step(Action(ActionType.HEADS))
    game.step(Action(ActionType.KICK))
    game.step(Action(ActionType.SETUP_FORMATION_SPREAD))
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
        game_turn_empty[key] = deepcopy(game)
    else:
        game_turn_full[key] = deepcopy(game)
    return game


Position = Union[Square, Tuple[int, int]]


def get_custom_game_turn(*,
                         player_positions: List[Position],
                         opp_player_positions: Optional[List[Position]] = None,
                         ball_position: Optional[Position] = None,
                         weather: WeatherType = WeatherType.NICE,
                         rerolls: int = 0,
                         forward_model_enabled=False,
                         pathfinding_enabled=False,
                         size: int = 11,
                         turn: int = 1) \
        -> Tuple[Game, Tuple[Player, ...]]:
    """
    :param player_positions: places human linemen of active team in these squares
    :param opp_player_positions: places human linemen of not active team in these squares
    :param ball_position: places ball in this square.
    :param weather:
    :param rerolls: number of rerolls
    :param forward_model_enabled:
    :param pathfinding_enabled:
    :param size: pitch size
    :param turn: turn of home_team, turn ordering not guaranteed
    :return: tuple with created game object followed by all the placed players
    """
    game = get_game_turn(empty=True, home_team='human', away_team='human', size=size)
    assert game.active_team is game.state.home_team

    home_players = [player for player in game.state.home_team.players if player.role.name == "Lineman"]
    away_players = [player for player in game.state.away_team.players if player.role.name == "Lineman"]
    assert opp_player_positions is None or len(away_players) >= len(opp_player_positions)
    assert player_positions is None or len(home_players) >= len(player_positions)

    game.state.weather = weather

    for team in game.state.teams:
        team.state.rerolls = rerolls

    def assert_square_type(obj: Position) -> Square:
        if type(obj) == Square:
            return game.get_square(obj.x, obj.y)
        else:
            return game.get_square(obj[0], obj[1])

    added_players = []

    for player, sq in zip(home_players, player_positions):
        assert player.team is game.state.home_team and player.position is None
        game.put(player, assert_square_type(sq))
        added_players.append(player)

    if opp_player_positions is not None:
        for player, sq in zip(away_players, opp_player_positions):
            assert player.team is game.state.away_team and player.position is None
            game.put(player, assert_square_type(sq))
            added_players.append(player)

    if ball_position is not None:
        game.get_ball().move_to(assert_square_type(ball_position))
        game.get_ball().is_carried = game.get_player_at(assert_square_type(ball_position)) is not None

    assert game.state.home_team.state.turn <= turn <= 8
    while game.state.home_team.state.turn < turn:
        game.step(Action(ActionType.END_TURN))
    assert game.active_team is game.state.home_team

    game.config.pathfinding_enabled = pathfinding_enabled
    game.set_available_actions()
    game.state.reports.clear()

    if forward_model_enabled:
        game.enable_forward_model()

    return game, tuple(added_players)


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


@contextmanager
def only_fixed_rolls(game: botbowl.Game,
                     assert_no_prev_fixes: bool = True,
                     assert_fixes_consumed: bool = True,
                     d3: Optional[Iterable[int]] = None,
                     d6: Optional[Iterable[int]] = None,
                     d8: Optional[Iterable[int]] = None,
                     block_dice: Optional[Iterable[BBDieResult]] = None):
    """
    Context manager that ensures that
      1) There are no fixes and the fixes rolls according to arguments
      2) No roll other than the fixed rolls are used i.e. no randomness
      3) All fixed rolls are consumed
    Example usage:
    > with only_fixed_rolls(game, block_dice=[BBDieResult.DEFENDER_DOWN], d6=[6, 6]):
    >     game.step(...)
    """
    if assert_no_prev_fixes:
        assert len(botbowl.D3.FixedRolls) == 0, f"There are fixed D3 rolls={botbowl.D3.FixedRolls}"
        assert len(botbowl.D6.FixedRolls) == 0, f"There are fixed D6 rolls={botbowl.D6.FixedRolls}"
        assert len(botbowl.D8.FixedRolls) == 0, f"There are fixed D8 rolls={botbowl.D8.FixedRolls}"
        assert len(botbowl.BBDie.FixedRolls) == 0, f"There are fixed BBDie rolls={botbowl.BBDie.FixedRolls}"

    if d3 is not None:
        for roll in d3:
            assert roll in {1, 2, 3}
            botbowl.D3.fix(roll)
    if d6 is not None:
        for roll in d6:
            assert roll in {1, 2, 3, 4, 5, 6}
            botbowl.D6.fix(roll)
    if d8 is not None:
        for roll in d8:
            assert roll in {1, 2, 3, 4, 5, 6, 7, 8}
            botbowl.D8.fix(roll)
    if block_dice is not None:
        for roll in block_dice:
            assert roll in BBDieResult
            botbowl.BBDie.fix(roll)

    rnd = game.rng
    game.rng = None

    try:
        yield

        if assert_fixes_consumed:
            assert len(botbowl.D3.FixedRolls) == 0, "Not all fixed D3 rolls were consumed"
            assert len(botbowl.D6.FixedRolls) == 0, "Not all fixed D6 rolls were consumed"
            assert len(botbowl.D8.FixedRolls) == 0, "Not all fixed D8 rolls were consumed"
            assert len(botbowl.BBDie.FixedRolls) == 0, "Not all fixed BBDie rolls were consumed"
    finally:
        game.rng = rnd

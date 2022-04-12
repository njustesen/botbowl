import botbowl
from botbowl.core import Action


def position_hash(sq: botbowl.Square) -> str:
    return f"{sq.x * 17 + sq.y:3.0f}"


def playerstate_hash(game: botbowl.Game, player: botbowl.core.model.Player) -> str:
    bool_attr = ('used', 'stunned', 'has_blocked')

    assert player.position is not None

    s = ""
    s += "h" if player.team is game.state.home_team else "a"
    s += position_hash(player.position)
    s += f"{player.state.moves:2.0f}"

    for attr in bool_attr:
        s += f"{1*getattr(player.state, attr)}"

    s += f"{player.role.name}"
    return s


def gamestate_hash(game: botbowl.Game) -> str:
    """
    Based the GameState, provides a str that can be used for fast and approximate game state comparisons
    """
    assert len(game.state.available_actions) > 0 or game.state.game_over, f"len(aa)=0 when game is not over!"

    s = ""
    s += "h" if game.active_team is game.state.home_team else "a"
    s += str(game.state.round)
    s += str(game.state.half)
    s += str(game.state.home_team.state.score)
    s += str(game.state.away_team.state.score)
    s += f"{len(game.state.reports):4.0f}"
    if not game.state.game_over:
        proc = game.get_procedure()
        s += f" {type(proc).__name__} "
        if isinstance(proc, botbowl.core.procedure.Setup):
            s += str(1*proc.reorganize)
        elif isinstance(proc, botbowl.core.procedure.Reroll):
            s += f"({type(proc.context).__name__})"
    else:
        s += "GAME_OVER"

    ball_pos = game.get_ball_position()
    s += f"ball={position_hash(ball_pos)} " if ball_pos is not None else " "

    for ac in game.get_available_actions():
        s += f" {ac.action_type.name}{len(ac.positions)},{hash(tuple(ac.positions))}"

    s += "".join(playerstate_hash(game, p) for p in game.get_players_on_pitch())

    return s

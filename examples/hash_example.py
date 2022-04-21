import botbowl
from botbowl.core import Action


def player_hash(player: botbowl.core.model.Player) -> str:
    if player.position is None:
        pos = None
    else:
        pos = hash(player.position)
    return f"p[{player.name}-{pos}-{hash(player.state)}]"


def team_hash(game: botbowl.Game, team: botbowl.Team):
    s = "h[" if game.state.home_team == team else "a["
    s += f"{hash(team.state)}"
    for player in team.players:
        s += f"-{player_hash(player)}"
    s += "]"
    return s


def gamestate_hash(game: botbowl.Game) -> str:
    """
    Based the GameState, provides a str that can be used for fast and approximate game state comparisons
    """
    s = ""
    s += "h" if game.active_team is game.state.home_team else "a"
    s += str(game.state.round)
    s += str(game.state.half)
    s += str(game.state.home_team.state.score)
    s += str(game.state.away_team.state.score)
    s += f"r{len(game.state.reports)}"
    if not game.state.game_over:
        proc = game.get_procedure()
        s += f"p{type(proc).__name__}"
        if isinstance(proc, botbowl.core.procedure.Setup):
            s += str(1*proc.reorganize)
        elif isinstance(proc, botbowl.core.procedure.Reroll):
            s += f"({type(proc.context).__name__})"
    else:
        s += "GAME_OVER"

    ball_pos = game.get_ball_position()
    s += f"b{hash(ball_pos)}" if ball_pos is not None else "b-"

    for ac in game.get_available_actions():
        s += f"a{hash(ac.action_type)}"

    s += team_hash(game, game.state.home_team)
    s += team_hash(game, game.state.away_team)

    return s

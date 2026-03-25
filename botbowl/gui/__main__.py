"""
CLI entry point: python -m botbowl.gui

Examples:
    python -m botbowl.gui                                   # Lobby
    python -m botbowl.gui --home-agent random --away-agent random
    python -m botbowl.gui --home-agent human --away-agent random
    python -m botbowl.gui --home-agent human --away-agent human
    python -m botbowl.gui --replay <replay-name>
"""
import argparse
import sys
import uuid

import botbowl
from botbowl.core.model import Agent
from botbowl.gui.gui import App
from botbowl.gui.screens.lobby import LobbyScreen


def main():
    parser = argparse.ArgumentParser(description='botbowl pygame GUI')
    parser.add_argument('--home-agent', default=None,
                        help='Home agent: "human" or a bot name (e.g. "random")')
    parser.add_argument('--away-agent', default=None,
                        help='Away agent: "human" or a bot name (e.g. "random")')
    parser.add_argument('--home-team', default='human',
                        help='Home team filename (default: human)')
    parser.add_argument('--away-team', default='human',
                        help='Away team filename (default: human)')
    parser.add_argument('--config', default='bot-bowl',
                        help='Game config name (default: bot-bowl)')
    parser.add_argument('--ai-delay', type=int, default=50,
                        help='Milliseconds to wait between AI steps (default: 50)')
    parser.add_argument('--replay', default=None,
                        help='Load and watch a replay by name')
    args = parser.parse_args()

    app = App(ai_delay_ms=args.ai_delay)

    if args.replay:
        from botbowl.gui.save_load import load_replay
        from botbowl.gui.screens.replay_screen import ReplayScreen
        replay = load_replay(args.replay)
        if replay is None:
            print(f'Replay not found: {args.replay}')
            sys.exit(1)
        app.push_screen(ReplayScreen(app, replay))
    elif args.home_agent is not None or args.away_agent is not None:
        # Start game directly
        _start_game(app, args)
    else:
        # Open lobby
        app.push_screen(LobbyScreen(app))

    app.run()


def _start_game(app, args):
    """Start a game directly from CLI args."""
    config_name = args.config
    config = botbowl.load_config(config_name)
    config.competition_mode = False
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)

    board_size = getattr(config, 'pitch_max', 11)

    home_team = botbowl.load_team_by_filename(args.home_team, ruleset, board_size=board_size)
    away_team = botbowl.load_team_by_filename(args.away_team, ruleset, board_size=board_size)

    home_agent_name = args.home_agent or 'human'
    away_agent_name = args.away_agent or 'human'

    if home_agent_name.lower() == 'human':
        home_agent = Agent('Human', human=True)
    else:
        home_agent = botbowl.make_bot(home_agent_name)

    if away_agent_name.lower() == 'human':
        away_agent = Agent('Human', human=True)
    else:
        away_agent = botbowl.make_bot(away_agent_name)

    game = botbowl.Game(
        str(uuid.uuid4()),
        home_team, away_team,
        home_agent, away_agent,
        config, arena=arena, ruleset=ruleset
    )
    game.config.fast_mode = False
    game.config.pathfinding_enabled = True
    game.config.pathfinding_with_carry_ball = True
    game.init()
    # Temporary: give teams resources for visual testing
    game.state.home_team.state.apothecaries = 1
    game.state.away_team.state.apothecaries = 1
    game.state.home_team.state.bribes = 2
    game.state.away_team.state.bribes = 2

    spectating = (not home_agent.human and not away_agent.human)
    from botbowl.gui.screens.game_screen import GameScreen
    screen = GameScreen(app, game, home_agent, away_agent,
                        spectating=spectating,
                        ai_delay_ms=app.ai_delay_ms)
    app.push_screen(screen)


if __name__ == '__main__':
    main()

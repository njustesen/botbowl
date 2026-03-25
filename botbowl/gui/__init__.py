"""
botbowl pygame GUI package.
"""
from botbowl.gui.gui import App
from botbowl.gui.screens.lobby import LobbyScreen
from botbowl.gui.screens.game_screen import GameScreen


def run_gui(ai_delay_ms: int = 50, **kwargs):
    """Start the GUI at the lobby screen."""
    app = App(ai_delay_ms=ai_delay_ms)
    app.push_screen(LobbyScreen(app))
    app.run()


__all__ = ['App', 'LobbyScreen', 'GameScreen', 'run_gui']

"""
botbowl pygame GUI package.
"""
from botbowl.gui.gui import App
from botbowl.gui.screens.lobby import MainMenuScreen, LobbyScreen
from botbowl.gui.screens.game_screen import GameScreen


def run_gui(ai_delay_ms: int = 50, **kwargs):
    """Start the GUI at the main menu screen."""
    app = App(ai_delay_ms=ai_delay_ms)
    app.push_screen(MainMenuScreen(app))
    app.run()


__all__ = ['App', 'MainMenuScreen', 'LobbyScreen', 'GameScreen', 'run_gui']

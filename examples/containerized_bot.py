from botbowl.ai import make_bot
from botbowl.ai.competition import PythonSocketServer
from botbowl.core.model import Agent
import sys
from examples.scripted_bot_example import MyScriptedBot


def run_bot_server(bot: Agent):
    bot_server = PythonSocketServer(agent=bot)
    bot_server.run()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        bot = MyScriptedBot("scripted")
    elif len(sys.argv) == 2:
        bot = make_bot(sys.argv[1])
    else:
        raise ValueError( f"Usage: python {__file__} takes 0 or 1 arguments, got '{sys.argv[1:]}'")

    run_bot_server(bot)

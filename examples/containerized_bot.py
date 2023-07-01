from botbowl.ai.competition import PythonSocketServer
from botbowl.core.model import Agent
from examples.scripted_bot_example import MyScriptedBot


def run_bot_server(bot: Agent):
    bot_server = PythonSocketServer(agent=bot)
    bot_server.run()


if __name__ == "__main__":
    # If you are submitting to botbowl make sure the your bot is created here instead of the ScriptedBot
    # bot = NuffleBot()
    bot = MyScriptedBot("Scripted bot example")
    run_bot_server(bot)

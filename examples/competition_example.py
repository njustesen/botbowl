#!/usr/bin/env python3

import ffai
import socket
from ffai.ai.competition import PythonSocketClient, PythonSocketServer
from multiprocessing import Process
import time
import secrets
import ffai
#from ffai.ai.bots import IdleBot
#ffai.register_bot('idle', IdleBot)


def run_agent(registration_name, port, token):
    """
    Starts a server that hosts an agent.
    """
    agent = ffai.make_bot(registration_name)
    server = PythonSocketServer(agent, port, token)
    server.run()


# Run servers
port = 50000
token_a = secrets.token_hex(32)
print(f"Token A: {token_a}")
process_a = Process(target=run_agent, args=('random', port, token_a))
process_a.start()
token_b = secrets.token_hex(32)
print(f"Token B: {token_b}")
process_b = Process(target=run_agent, args=('random', port+1, token_b))
process_b.start()

# Specify the host running the agents (localhost)
hostname = socket.gethostname()

# Make sure the agents are running
time.sleep(2)

# Load configurations, rules, arena and teams
config = ffai.load_config("bot-bowl-ii")
'''
config = ffai.load_config("ff-1")
config.competition_mode = True
config.time_limits.turn = 0.5
config.time_limits.secondary = 1
config.time_limits.init = 2
config.time_limits.end = 2
'''

ruleset = ffai.load_rule_set(config.ruleset)
arena = ffai.load_arena(config.arena)
team_a = ffai.load_team_by_filename("human", ruleset)
team_b = ffai.load_team_by_filename("human", ruleset)

# Make proxy agents
client_a = PythonSocketClient("Player A", hostname, port, token=token_a)
client_b = PythonSocketClient("Player B", hostname, port + 1, token=token_b)

# Run competition
competition = ffai.Competition(client_a, client_b, team_a, team_b, config=config, ruleset=ruleset, arena=arena, n=10)
competition.run()
competition.results.print()

# Shut down everything
process_a.terminate()
process_a.join()
process_b.terminate()
process_b.join()

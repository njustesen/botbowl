#!/usr/bin/env python3

import botbowl
import socket
from botbowl.ai.competition import PythonSocketClient, PythonSocketServer
from multiprocessing import Process
import time
import secrets
import botbowl
import socket


def run_agent(registration_name, port, token):
    """
    Starts a server that hosts an agent.
    """
    agent = botbowl.make_bot(registration_name)
    server = PythonSocketServer(agent, port, token)
    server.run()

# Run servers
token_a = secrets.token_hex(32)
print(f"Token A: {token_a}")
process_a = Process(target=run_agent, args=('random', 5100, token_a))
process_a.start()
token_b = secrets.token_hex(32)
print(f"Token B: {token_b}")
process_b = Process(target=run_agent, args=('random', 5200, token_b))
process_b.start()

# Specify the host running the agents (localhost)
hostname = socket.gethostname()

# Make sure the agents are running
time.sleep(2)

# Load configurations, rules, arena and teams
config = botbowl.load_config("bot-bowl-iii")

ruleset = botbowl.load_rule_set(config.ruleset)
arena = botbowl.load_arena(config.arena)
team_a = botbowl.load_team_by_filename("human", ruleset)
team_b = botbowl.load_team_by_filename("human", ruleset)

# Make proxy agents
hostname = socket.gethostname()
client_a = PythonSocketClient("Player A", hostname, 5100, token=token_a)
client_b = PythonSocketClient("Player B", hostname, 5200, token=token_b)

# Run competition
competition = botbowl.Competition(client_a, client_b, team_a, team_b, config=config, ruleset=ruleset, arena=arena, n=2, record=True)
competition.run()
competition.results.print()

# Shut down everything
process_a.terminate()
process_a.join()
process_b.terminate()
process_b.join()

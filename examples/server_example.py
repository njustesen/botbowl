import ffai.web.server as server
import ffai.web.api as api
from ffai.ai.registry import make_bot
from ffai.core.model import Agent

# Import this to register MyScriptedBot with id 'scripted'
import examples.scripted_bot_example
import examples.grodbot

# Create some games
api.new_game(home_team_id="orc-1",
             away_team_id="human-1",
             home_agent=make_bot("random"),
             away_agent=Agent("Player 2", human=True))

api.new_game(home_team_id="orc-1",
             away_team_id="human-1",
             home_agent=make_bot("scripted"),
             away_agent=Agent("Player 2", human=True))

api.new_game(home_team_id="human-1",
             away_team_id="human-2",
             home_agent=Agent("Player 1", human=True),
             away_agent=Agent("Player 2", human=True))

api.new_game(home_team_id="orc-1",
             away_team_id="human-1",
             home_agent=make_bot("GrodBot"),
             away_agent=make_bot("GrodBot"))

# Run server
server.start_server(debug=True, use_reloader=True)
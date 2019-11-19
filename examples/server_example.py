#!/usr/bin/env python3

import ffai.web.server as server

# Create some games
'''
api.new_game(home_team_id="orc-1",
             away_team_id="human-1",
             home_agent=make_bot("random"),
             away_agent=Agent("Player 3", human=True))

api.new_game(home_team_id="orc-1",
             away_team_id="human-1",
             home_agent=make_bot("grodbot"),
             away_agent=Agent("Player 3", human=True))
    
api.new_game(home_team_id="human-1",
             away_team_id="human-3",
             home_agent=Agent("Player 1", human=True),
             away_agent=Agent("Player 3", human=True))

api.new_game(home_team_id="human-1",
             away_team_id="human-3",
             home_agent=make_bot("scripted"),
             away_agent=make_bot("scripted"))

api.new_game(home_team_id="human-1",
             away_team_id="human-3",
             home_agent=make_bot("GrodBot"),
             away_agent=make_bot("GrodBot"))
'''

if __name__ == "__main__":

    # Run server
    server.start_server(debug=True, use_reloader=False)
# Run the Web Application
FFAI comes with a web application that allows you to play the game against a bot or human. You can also spectate a match between other humans or bots.

## Run the App
Make a script (or use the [server_example.py](../examples/server_example.py)) with the following:
```
#!/usr/bin/env python3
import ffai.web.server as server
server.start_server(port=5005)
```

Now run the script and go to [http://localhost:5005/](http://localhost:5005/). You can make a game on the front page.

Alternatively, you can add games like this, e.g. before starting the server:
```
api.new_game(home_team_id="orc",
             away_team_id="human",
             home_agent=Agent("Player 1", human=True)),
             away_agent=make_bot("grodbot")
```

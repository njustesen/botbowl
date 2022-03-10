"""
==========================
Author: Niels Justesen
Year: 2018
==========================
Run this script to start a Flask server locally. The server will start a Host, which will manage games.
"""

from flask import Flask, request, render_template, redirect
from botbowl.web import api
from botbowl.core.load import *
import json
import random
from botbowl.ai.registry import make_bot
import traceback

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/game/create', methods=['PUT'])
def create():
    data = json.loads(request.data)
    bot_list = api.get_bots()
    game_mode = data['mode']
    # make_bot or Agent depending on choice... (unknown name => human)
    homePlayer = data['game']['home_player']
    num_human_players = 0
    if homePlayer in bot_list:
        homeAgent = make_bot(homePlayer)
    else:
        num_human_players += 1
        homeAgent = Agent(f"Player {num_human_players}", human=True)

    awayPlayer = data['game']['away_player']
    if awayPlayer in bot_list:
        awayAgent = make_bot(awayPlayer)
    else:
        num_human_players += 1
        awayAgent = Agent(f"Player {num_human_players}", human=True)

    game = api.new_game(home_team_name=data['game']['home_team_name'],
                        away_team_name=data['game']['away_team_name'],
                        home_agent=homeAgent,
                        away_agent=awayAgent,
                        game_mode=game_mode)

    return json.dumps(game.to_json())


@app.route('/game/save', methods=['POST'])
def save():
    game_id = json.loads(request.data)['game_id']
    name = json.loads(request.data)['name']

    if len(name) > 2 and len(name) < 40 and not api.save_game_exists(name):
        api.save_game(game_id, name)
        return json.dumps("Game was successfully saved")
    else:
        raise Exception("Cannot save this game")


@app.route('/game/<game_id>/delete', methods=['DELETE'])
def delete(game_id):
    api.delete_game(game_id)
    return f"{game_id} deleted"


@app.route('/save/<name>/delete', methods=['DELETE'])
def delete_saved(name):
    api.delete_save(name)
    return f"{name} deleted"


@app.route('/games/', methods=['GET'])
def get_all_games():
    return json.dumps({
        'games': [game.to_json() for game in api.get_games()],
        'saved_games': [{'name': name, 'game': game.to_json()} for (name, game) in api.get_saved_games()]
    })


@app.route('/game-modes/', methods=['GET'])
def get_all_game_modes():
    return json.dumps(list(api.game_modes.keys()))


@app.route('/replays/', methods=['GET'])
def get_all_replays():
    replays = api.get_replay_ids()
    return json.dumps({
        'replays': replays
    })

@app.route('/teams/<game_mode>', methods=['GET'])
def get_all_teams(game_mode='standard'):
    teams = api.get_teams(game_mode)
    team_list = []
    for team in teams:
        team_list.append(team.to_json())
    return json.dumps(team_list)


@app.route('/games/<game_id>/act', methods=['POST'])
def step(game_id):
    try:
        action = json.loads(request.data)['action']
        action_type = parse_enum(ActionType, action['action_type'])
        position = Square(action['position']['x'], action['position']['y']) if 'position' in action and action['position'] is not None else None
        player_id = action['player_id'] if 'player_id' in action else None
        game = api.get_game(game_id)
        player = game.get_player(player_id) if player_id is not None else None
        action = Action(action_type, position=position, player=player)
        game = api.step(game_id, action)
    except Exception as e:
        print(e)
        traceback.print_exc()
        game = api.get_game(game_id)
    return json.dumps(game.to_json())


@app.route('/games/<game_id>', methods=['GET'])
def get_game(game_id):
    return json.dumps(api.get_game(game_id).to_json())

@app.route('/replays/<replay_id>', methods=['GET'])
def get_replay(replay_id):
    replay = api.get_replay(replay_id)
    replay.steps = api.get_replay_steps(replay_id, 0, 100)
    replay_str = json.dumps(replay.to_json())
    return replay_str

@app.route('/steps/<replay_id>/<from_idx>/<num_steps>', methods=['GET'])
def get_steps(replay_id, from_idx, num_steps):
    steps = api.get_replay_steps(replay_id, int(from_idx), int(num_steps))
    steps_str = {idx: step.game for idx, step in steps.items()}
    return steps_str


@app.route('/game/load/<name>/', methods=['GET'])
def load_game(name):
    game = api.load_game(name)
    game.set_seed(random.randint(0, 2**32-1))
    return json.dumps(game.to_json())


@app.route('/bots/', methods=['GET'])
def get_bots():
    return json.dumps(api.get_bots())


def start_server(debug=False, use_reloader=False, port=5000, host="127.0.0.1"):
    
    # Change jinja notation to work with angularjs
    jinja_options = app.jinja_options.copy()
    jinja_options.update(dict(
        block_start_string='<%',
        block_end_string='%>',
        variable_start_string='%%',
        variable_end_string='%%',
        comment_start_string='<#',
        comment_end_string='#>'
    ))
    app.jinja_options = jinja_options

    app.config['TEMPLATES_AUTO_RELOAD']=True
    app.run(host=host, debug=debug, use_reloader=use_reloader, port=port)

from flask import Flask, request, render_template
from bb.web import api
from bb.core.load import *
import json


app = Flask(__name__)

'''
To start server:
$ export FLASK_DEBUG=1
$ flask run
'''


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/game/create', methods=['PUT'])
def create():
    data = json.loads(request.data)
    game = api.new_game(data['game']['home_team_id'], data['game']['away_team_id'])
    return json.dumps(game.to_simple())


@app.route('/game/save', methods=['POST'])
def save():
    game_id = json.loads(request.data)['game_id']
    name = json.loads(request.data)['name']
    if len(name) > 2 and len(name) < 40 and not api.save_game_exists(name):
        api.save_game(game_id, name)
        return json.dumps("Game was successfully saved")
    else:
        raise Exception("Cannot save this game")


@app.route('/games/', methods=['GET'])
def get_all_games():
    games = api.get_games()
    saved_games = api.get_saved_games()
    game_list = [game.to_simple() for game in games]
    saved_game_list = [{'game': save[0].to_simple(), 'name': save[1]} for save in saved_games]
    return json.dumps({
        'games': game_list,
        'saved_games': saved_game_list
    })


@app.route('/teams/', methods=['GET'])
def get_all_teams():
    teams = api.get_teams()
    team_list = []
    for team in teams:
        team_list.append(team.to_simple())
    return json.dumps(team_list)


@app.route('/games/<game_id>/act', methods=['POST'])
def step(game_id):
    try:
        action = json.loads(request.data)['action']
        action_type = parse_enum(ActionType, action['action_type'])
        pos = Square(action['pos']['x'], action['pos']['y']) if 'pos' in action and action['pos'] is not None else None
        player_id = action['player_id'] if 'player_id' in action else None
        idx = action['idx'] if 'idx' in action else -1
        game = api.get_game(game_id)
        player = game.get_player(player_id) if player_id is not None else None
        action = Action(action_type, pos=pos, player=player, idx=idx)
        game = api.step(game_id, action)
    except Exception as e:
        print(e)
        game = api.get_game(game_id)
    return json.dumps(game.to_simple())


@app.route('/games/<game_id>', methods=['GET'])
def get_game(game_id):
    return json.dumps(api.get_game(game_id).to_simple())


@app.route('/game/load/<name>', methods=['GET'])
def load_game(name):
    return json.dumps(api.load_game(name).to_simple())


if __name__ == '__main__':
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
    app.run(debug=True, use_reloader=True)

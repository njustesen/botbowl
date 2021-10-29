"""
==========================
Author: Niels Justesen
Year: 2020
==========================
"""
from botbowl.core.procedure import *
import socket
import pickle
import secrets

HEADERSIZE = 10

# Maintain socket outside of Agent to avoid serializations
sockets = {}


def send_data(data, socket, timeout=None):
    if timeout is not None and timeout < 0:
        raise Exception("Timeout cannot be lower than 0")
    msg = pickle.dumps(data)
    msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8') + msg
    socket.settimeout(timeout)
    socket.sendall(msg)


def receive_data(socket, timeout=None):
    if timeout is not None and timeout < 0:
        raise Exception("Timeout cannot be lower than 0")
    full_msg = b''
    new_msg = True
    socket.settimeout(timeout)
    while True:
        msg = socket.recv(4096)
        if new_msg:
            msglen = int(msg[:HEADERSIZE])
            new_msg = False
        full_msg += msg
        if len(full_msg) - HEADERSIZE == msglen:
            return pickle.loads(full_msg[HEADERSIZE:])


class Request:

    def __init__(self, command, game=None, team=None):
        self.command = command
        self.game = game
        self.team = team
        self.request_id = secrets.token_hex(32)


class Response:

    def __init__(self, object, token, request_id):
        self.object = object
        self.token = token
        self.request_id = request_id


class PythonSocketClient(Agent):

    def __init__(self, name, host, port, token, connection_timeout=1):
        super().__init__(name)
        self.host = host
        self.port = port
        self.token = token
        self.connection_timeout = connection_timeout
        print(f"Connected to {host}:{port}.")

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.connection_timeout)
        s.connect((self.host, self.port))
        sockets[self.agent_id] = s

    def act(self, game):
        seconds = game.get_seconds_left()
        replay = game.replay
        game.replay = None
        try:
            self._connect()
            request = Request('act', game=game)
            send_data(request, sockets[self.agent_id], timeout=seconds)
            seconds = game.get_seconds_left()
            while seconds is None or seconds > 0:
                try:
                    response = receive_data(sockets[self.agent_id], timeout=(seconds if seconds is not None else None))
                    if type(response) != Response:
                        print(f"Invalid response type: {response}")
                    elif response.object is None and type(response.object) == Action:
                        print(f"Invalid action type: {type(response.object)}")
                    elif str(response.token) != str(self.token):
                        print(f"Invalid token: {response.token} != {self.token}")
                    elif str(response.request_id) != str(request.request_id):
                        print(f"Invalid request_id: {response.request_id} != {request.request_id}")
                    else:
                        game.replay = replay
                        return response.object
                except Exception as e:
                    print(f"Error parsing message from {self.name}: ", e)
                    for proc in game.get_procedure_names():
                        print(proc)
                seconds = game.get_seconds_left()
        except Exception as e:
            print(f"{self.name} failed to communicate: ", e)
            print("Returning None action")
        game.replay = replay
        return None

    def new_game(self, game, team):
        seconds = game.get_seconds_left()
        replay = game.replay
        game.replay = None
        try:
            self._connect()
            request = Request('new_game', game=game, team=team)
            send_data(request, sockets[self.agent_id], timeout=seconds)
            if game.config.time_limits is not None:
                seconds = game.config.time_limits.end
            while seconds is None or seconds > 0:
                try:
                    response = receive_data(sockets[self.agent_id], timeout=(seconds if seconds is not None else None))
                    if type(response) != Response:
                        print(f"Invalid response type: {response}")
                    elif str(response.token) != str(self.token):
                        print(f"Invalid token: {response.token} != {self.token}")
                    elif str(response.request_id) != str(request.request_id):
                        print(f"Invalid request_id: {response.request_id} != {request.request_id}")
                    else:
                        game.replay = replay
                        return
                except Exception as e:
                    print(f"Error parsing message: ", e)
                seconds = game.get_seconds_left()
        except Exception as e:
            print(f"{self.name} failed to communicate: ", e)
        game.replay = replay

    def end_game(self, game):
        seconds = game.get_seconds_left()
        replay = game.replay
        game.replay = None
        try:
            self._connect()
            request = Request('end_game', game=game)
            send_data(request, sockets[self.agent_id], timeout=seconds)
            if game.config.time_limits is not None:
                seconds = game.config.time_limits.init
            while seconds is None or seconds > 0:
                try:
                    response = receive_data(sockets[self.agent_id], timeout=(seconds if seconds is not None else None))
                    if type(response) != Response:
                        print(f"Invalid response type: {response}")
                    elif str(response.token) != str(self.token):
                        print(f"Invalid token: {response.token} != {self.token}")
                    elif str(response.request_id) != str(request.request_id):
                        print(f"Invalid request_id: {response.request_id} != {request.request_id}")
                    else:
                        self._close()
                        game.replay = replay
                        return
                except Exception as e:
                    print(f"Error parsing message: ", e)
                seconds = game.get_seconds_left()
        except Exception as e:
            print(f"{self.name} failed to communicate: ", e)
        game.replay = replay

    def _close(self):
        sockets[self.agent_id].close()


class PythonSocketServer(Agent):

    def __init__(self, agent, port, token):
        super().__init__(agent.name)
        self.agent = agent
        self.port = port
        self.token = token
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((socket.gethostname(), port))
        self.socket.listen(1)
        print(f"Agent listening on {socket.gethostname()}:{port} using token {token}")

    def run(self):
        while True:
            try:
                print("socket accept")
                socket, client_address = self.socket.accept()
                print("receive data")
                request = receive_data(socket)
                print("data received")
                if type(request) is not Request:
                    raise Exception(f"Unreadable request {request}")
                if request.command == 'act':
                    if request.game is None:
                        raise Exception(f"No game provided an 'act' request")
                    action = self.act(request.game)
                    send_data(Response(action, self.token, request.request_id), socket, timeout=2)
                elif request.command == 'new_game':
                    if request.game is None or request.team is None:
                        raise Exception(f"No game or team provided an 'new_Game' request")
                    self.new_game(request.game, request.team)
                    send_data(Response(None, self.token, request.request_id), socket, timeout=2)
                elif request.command == 'end_game':
                    if request.game is None:
                        raise Exception(f"No game provided an 'end_game' request")
                    self.end_game(request.game)
                    send_data(Response(None, self.token, request.request_id), socket, timeout=2)
                else:
                    raise Exception(f"Unknown command {request.command}")
            except Exception as e:
                print(e)

    def _close(self):
        if self.socket is not None:
            self.socket.close()
        self.socket = None

    def new_game(self, game, team):
        self.agent.new_game(game, team)

    def act(self, game):
        return self.agent.act(game)

    def end_game(self, game):
        self.agent.end_game(game)

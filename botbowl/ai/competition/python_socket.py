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
import docker
import enum
import time

from typing import Union

HEADERSIZE = 10
DEFAULT_PORT = 5100
DEFAULT_TOKEN = "32"


class AgentCommand(str, enum.Enum):
    ACT = "act"
    END_GAME = "end"
    NEW_GAME = "new_game"
    STATE_NAME = "state_name"


# Maintain socket outside of Agent to avoid serializations
sockets = {}


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


def send_data(data: Union[Request, Response], socket, timeout=None):
    if timeout is not None and timeout < 0:
        raise Exception("Timeout cannot be lower than 0")
    msg = pickle.dumps(data)
    msg = bytes(f"{len(msg):<{HEADERSIZE}}", "utf-8") + msg
    socket.settimeout(timeout)
    socket.sendall(msg)


def receive_data(socket, timeout=None) -> Response:
    if timeout is not None and timeout < 0:
        raise Exception("Timeout cannot be lower than 0")
    full_msg = b""
    new_msg = True
    socket.settimeout(timeout)
    while True:
        msg = socket.recv(4096)
        if new_msg:
            msglen = int(msg[:HEADERSIZE])
            new_msg = False
        full_msg += msg
        if len(full_msg) - HEADERSIZE == msglen:
            # TODO: should use json for this
            return pickle.loads(full_msg[HEADERSIZE:])


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
        # TODO: make sure we are not sending the actual RNG seed.
        self._connect()
        request = Request(AgentCommand.ACT, game=game)
        send_data(request, sockets[self.agent_id], timeout=seconds)
        seconds = game.get_seconds_left()
        while seconds is None or seconds > 0:
            try:
                response = receive_data(
                    sockets[self.agent_id],
                    timeout=(seconds if seconds is not None else None),
                )
                if type(response) != Response:
                    print(f"Invalid response type: {response}")
                elif response.object is None and type(response.object) == Action:
                    # FIX: this is never True... ?
                    print(f"Invalid action type: {type(response.object)}")
                elif str(response.token) != str(self.token):
                    print(f"Invalid token: {response.token} != {self.token}")
                elif str(response.request_id) != str(request.request_id):
                    print(
                        f"Invalid request_id: {response.request_id} != {request.request_id}"
                    )
                else:
                    game.replay = replay
                    return response.object
            except Exception as e:
                print(f"Error parsing message from {self.name}: ", e)
                for proc in game.get_procedure_names():
                    print(proc)
            seconds = game.get_seconds_left()
        game.replay = replay
        return None

    def new_game(self, game, team):
        seconds = game.get_seconds_left()
        replay = game.replay
        game.replay = None
        try:
            self._connect()
            request = Request(AgentCommand.NEW_GAME, game=game, team=team)
            send_data(request, sockets[self.agent_id], timeout=seconds)
            if game.config.time_limits is not None:
                seconds = game.config.time_limits.end
            while seconds is None or seconds > 0:
                try:
                    response = receive_data(
                        sockets[self.agent_id],
                        timeout=(seconds if seconds is not None else None),
                    )
                    if type(response) != Response:
                        print(f"Invalid response type: {response}")
                    elif str(response.token) != str(self.token):
                        print(f"Invalid token: {response.token} != {self.token}")
                    elif str(response.request_id) != str(request.request_id):
                        print(
                            f"Invalid request_id: {response.request_id} != {request.request_id}"
                        )
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
            request = Request(AgentCommand.END_GAME, game=game)
            send_data(request, sockets[self.agent_id], timeout=seconds)
            if game.config.time_limits is not None:
                seconds = game.config.time_limits.init
            while seconds is None or seconds > 0:
                try:
                    response = receive_data(
                        sockets[self.agent_id],
                        timeout=(seconds if seconds is not None else None),
                    )
                    if type(response) != Response:
                        print(f"Invalid response type: {response}")
                    elif str(response.token) != str(self.token):
                        print(f"Invalid token: {response.token} != {self.token}")
                    elif str(response.request_id) != str(request.request_id):
                        print(
                            f"Invalid request_id: {response.request_id} != {request.request_id}"
                        )
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


def docker_image_exists(docker_client, img_name: str) -> bool:
    for image in docker_client.images.list():
        if img_name in image.tags:
            return True
    return False


class DockerAgent(PythonSocketClient):
    def __init__(
        self,
        img_name: str,
        command: Optional[str],
    ):
        self.img_name = img_name
        self.command = command
        # self.container = None
        api = docker.from_env()

        # make sure image is available
        if not docker_image_exists(api, img_name):
            print(f"Pulling image {img_name}")
            api.images.pull(img_name)
        assert docker_image_exists(api, img_name), f"Image {img_name} not found"

        host_port = get_free_port()
        assert host_port > 80, f"Invalid port: {host_port}"
        print(f"Using port {host_port}")
        api.containers.run(
            img_name,
            command,
            detach=True,
            auto_remove=True,
            ports={DEFAULT_PORT: host_port},
        )
        time.sleep(4)

        super().__init__(img_name, host="127.0.0.1", port=host_port, token=DEFAULT_TOKEN, connection_timeout=4)

        # get name of agent
        self._connect()
        send_data(Request(AgentCommand.STATE_NAME), sockets[self.agent_id], timeout=5)
        response = receive_data(sockets[self.agent_id], timeout=5)
        name = response.object
        assert type(name) == str, f"Invalid name type: {type(name)}"


def get_free_port() -> int:
    """Get a port that is not in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

# FIX: not needed, agent is handled with composition not inheritence
class PythonSocketServer(Agent):
    def __init__(self, agent, port: Optional[int] = DEFAULT_PORT, token: Optional[str] = DEFAULT_TOKEN):
        super().__init__(agent.name)  # FIX: not needed
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
                if request.command == AgentCommand.ACT:
                    if request.game is None:
                        raise Exception(f"No game provided an 'act' request")
                    action = self.act(request.game)
                    send_data(
                        Response(action, self.token, request.request_id),
                        socket,
                        timeout=2,
                    )
                elif request.command == AgentCommand.STATE_NAME:
                    name = str(self.agent.name)
                    send_data(
                        Response(name, self.token, request.request_id),
                        socket,
                        timeout=2,
                    )
                elif request.command == AgentCommand.NEW_GAME:
                    if request.game is None or request.team is None:
                        raise Exception(
                            f"No game or team provided an 'new_Game' request"
                        )
                    self.new_game(request.game, request.team)
                    send_data(
                        Response(None, self.token, request.request_id),
                        socket,
                        timeout=2,
                    )
                elif request.command == AgentCommand.END_GAME:
                    if request.game is None:
                        raise Exception(f"No game provided an 'end_game' request")
                    self.end_game(request.game)
                    send_data(
                        Response(None, self.token, request.request_id),
                        socket,
                        timeout=2,
                    )
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

"""
==========================
Author: Niels Justesen
Year: 2020
==========================
"""
from botbowl.core.procedure import *
from botbowl.core.game import Game
import socket
import pickle
import secrets
import docker
import enum
import time

from typing import Union, Type, TypeVar

T = TypeVar("T")

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
    msglen = None
    socket.settimeout(timeout)
    while True:
        msg = socket.recv(4096)
        if msglen is None:
            msglen = int(msg[:HEADERSIZE])
        full_msg += msg
        if len(full_msg) - HEADERSIZE == msglen:
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

    def _send_command(
        self,
        command: AgentCommand,
        *,
        expected_return_type: Type[T] = None,
        game: Game,
        team=None,
    ) -> T:

        self._connect()

        timeout = game.get_seconds_left()
        with game.hide_agents_and_rng():
            request = Request(command, game=game, team=team)
            send_data(request, sockets[self.agent_id], timeout=timeout)

        response = receive_data(sockets[self.agent_id], timeout=timeout)

        assert type(response) == Response, f"'{type(response)}' != '{Response}'"
        assert str(response.token) == str(self.token)
        assert str(response.request_id) == str(request.request_id)
        if expected_return_type is not None: 
            assert type(response.object) == expected_return_type
        return response.object  # type: ignore

    def act(self, game) -> Action:
        return self._send_command(
            AgentCommand.ACT, expected_return_type=Action, game=game
        )

    def new_game(self, game, team) -> None:
        self._send_command(AgentCommand.NEW_GAME, game=game, team=team)

    def end_game(self, game):
        self._send_command(AgentCommand.END_GAME, game=game)
        self._close()

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
        try:
            api = docker.from_env()
        except Exception:
            raise Exception("Failed to initialize docker API")

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

        super().__init__(
            img_name,
            host="127.0.0.1",
            port=host_port,
            token=DEFAULT_TOKEN,
            connection_timeout=4,
        )

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
    socket_: Optional[socket.socket]

    def __init__(
        self,
        agent,
        port: Optional[int] = DEFAULT_PORT,
        token: Optional[str] = DEFAULT_TOKEN,
    ):
        super().__init__(agent.name)  # FIX: not needed
        self.agent = agent
        self.port = port
        self.token = token
        self.socket_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_.bind((socket.gethostname(), port))
        self.socket_.listen(1)
        print(f"Agent listening on {socket.gethostname()}:{port} using token {token}")

    def run(self):
        assert self.socket_ is not None
        while True:
            try:
                print("socket accept")
                socket, _ = self.socket_.accept()
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
        if self.socket_ is not None:
            self.socket_.close()
        self.socket_ = None

    def new_game(self, game, team):
        self.agent.new_game(game, team)

    def act(self, game):
        return self.agent.act(game)

    def end_game(self, game):
        self.agent.end_game(game)

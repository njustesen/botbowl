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
    command: AgentCommand
    game: Optional[Game]
    team: Optional[Team]
    request_id: str

    def __init__(self, command, game=None, team=None):
        self.command = command
        self.game = game
        self.team = team
        self.request_id = secrets.token_hex(32)

    def validate(self):
        if self.command == AgentCommand.ACT:
            assert isinstance(self.game, Game)
            assert self.team is None
        elif self.command == AgentCommand.NEW_GAME:
            assert isinstance(self.game, Game)
            assert isinstance(self.team, Team)
        elif self.command == AgentCommand.END_GAME:
            assert isinstance(self.game, Game)
            assert self.team is None
        elif self.command == AgentCommand.STATE_NAME:
            assert self.game is None
            assert self.team is None
        else:
            raise AssertionError(f"Unknown command: {self.command}")


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


def receive_data(connection, timeout=None) -> Response:
    if timeout is not None and timeout < 0:
        raise Exception("Timeout cannot be lower than 0")
    full_msg = b""
    msglen = None
    connection.settimeout(timeout)
    while True:
        msg = connection.recv(4096)
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
        name: str,
        *,
        image: str,
        command: Optional[str],
    ):
        self.container = None

        try:
            api = docker.from_env()
        except Exception:
            raise Exception("Failed to initialize docker API")

        # make sure image is available
        if not docker_image_exists(api, image):
            print(f"Pulling image {image}")
            api.images.pull(image)
        assert docker_image_exists(api, image), f"Image {image} not found"

        host_port = get_free_port()
        print(f"Using port {host_port}")
        self.container = api.containers.run(
            image,
            command,
            detach=True,
            auto_remove=True,
            ports={DEFAULT_PORT: host_port},
        )

        super().__init__(
            name=name,
            host="127.0.0.1",
            port=host_port,
            token=DEFAULT_TOKEN,
            connection_timeout=4,
        )
        time.sleep(1) # weird connection errors can possibly be solved by increasing this


    def __del__(self):
        if self.container is not None:
            self.container.kill()


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class PythonSocketServer:
    socket_: socket.socket

    def __init__(
        self,
        agent,
        port: Optional[int] = DEFAULT_PORT,
        token: Optional[str] = DEFAULT_TOKEN,
    ):
        self.agent = agent
        self.port = port
        self.token = token

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: 
            s.bind((socket.gethostname(), self.port))
            s.listen(1)
            print(f"Agent listening on {socket.gethostname()}:{self.port} using token {self.token}")

            while True:
                connection, _ = s.accept()
                with connection:
                    request = receive_data(connection)
                    data = self.handle_request(request) # type: ignore
                    response = Response(data, self.token, request.request_id)
                    send_data(response, connection, timeout=2)

    def handle_request(self, request: Request) -> Union[Action, str, None]:
        assert isinstance(request, Request)
        request.validate()
        if request.command == AgentCommand.ACT:
            return self.agent.act(request.game)
        elif request.command == AgentCommand.STATE_NAME:
            return str(self.agent.name)
        elif request.command == AgentCommand.NEW_GAME:
            self.agent.new_game(request.game, request.team)
            return None
        elif request.command == AgentCommand.END_GAME:
            self.agent.end_game(request.game)
            return None
        else:
            raise Exception(f"Unknown command {request.command}")

    def _close(self):
        if self.socket_ is not None:
            self.socket_.close()

    def __del__(self):
        self._close()

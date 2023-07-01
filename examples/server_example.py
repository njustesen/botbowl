#!/usr/bin/env python3
import os
from scripted_bot_example import *

import botbowl.web.server as server

if __name__ == "__main__":
    if os.path.exists("/.dockerenv"):
        # If you're running in a docker container, you want the server to be accessible from outside the container.
        host = "0.0.0.0"
    else:
        # If you're running locally, you want the server to be accessible from localhost only b/c of security.
        host = "127.0.0.1"

    server.start_server(host=host, debug=True, use_reloader=False, port=1234)

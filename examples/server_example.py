#!/usr/bin/env python3
from grodbot import *
from scripted_bot_example import *

import ffai.web.server as server

if __name__ == "__main__":
    server.start_server(debug=True, use_reloader=False)
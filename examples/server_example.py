#!/usr/bin/env python3

import ffai.web.server as server
import examples.grodbot as grodbot

if __name__ == "__main__":
    server.start_server(debug=True, use_reloader=False)
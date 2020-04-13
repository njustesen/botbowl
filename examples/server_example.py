#!/usr/bin/env python3

import ffai.web.server as server

if __name__ == "__main__":
    server.start_server(debug=True, use_reloader=False)
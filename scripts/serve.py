#!/usr/bin/env python3
import functools
import http.server
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

PORT = int(os.environ.get("SKYZERO_WEB_PORT", "8000"))
PUBLIC_DIR = Path(__file__).resolve().parents[1] / "public"
ROUTES = {
    "/poker": "poker/index.html",
    "/2048": "2048/index.html",
    "/channel-dodge": "channel-dodge/index.html",
    "/gomoku": "gomoku/index.html",
    "/mcts-tree": "gomoku/tree/index.html",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        request_path = urlsplit(path).path.rstrip("/") or "/"
        route = ROUTES.get(request_path)
        if route is not None:
            return str(PUBLIC_DIR / route)
        if re.fullmatch(r"/gomoku/lv[1-6]", request_path):
            return str(PUBLIC_DIR / "gomoku/index.html")
        return super().translate_path(path)


handler = functools.partial(Handler, directory=PUBLIC_DIR)
with http.server.ThreadingHTTPServer(("", PORT), handler) as httpd:
    print(f"Serving http://localhost:{PORT}  ->  /gomoku  /2048  /channel-dodge")
    httpd.serve_forever()

#!/usr/bin/env python3
"""Local static preview that mimics Cloudflare Pages clean URLs (/gomoku -> gomoku.html)."""
import http.server
import os

PORT = int(os.environ.get("SKYZERO_WEB_PORT", "8000"))


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        local = super().translate_path(path)
        # Clean URLs: serve gomoku.html for /gomoku and model routes such as
        # /gomoku/lv6, like Cloudflare Pages.
        if path.rstrip("/").startswith("/gomoku/"):
            return os.path.join(os.getcwd(), "gomoku.html")
        if not os.path.splitext(local)[1] and os.path.isfile(local + ".html"):
            return local + ".html"
        return local


with http.server.ThreadingHTTPServer(("", PORT), Handler) as httpd:
    print(f"Serving http://localhost:{PORT}  ->  /gomoku  /2048  /channel-dodge")
    httpd.serve_forever()

#!/usr/bin/env python3
"""Local static preview that mimics Cloudflare Pages clean URLs (/gomoku -> gomoku.html)."""
import http.server
import os
import socketserver

PORT = 8000


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        local = super().translate_path(path)
        # Clean URLs: serve gomoku.html when /gomoku is requested, like Cloudflare Pages.
        if not os.path.splitext(local)[1] and os.path.isfile(local + ".html"):
            return local + ".html"
        return local


with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving http://localhost:{PORT}  ->  /gomoku  /2048  /channel-dodge")
    httpd.serve_forever()

import hmac
import json
import os
import platform
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
from pokerskill_agent import generate_prompt
from pokerskill_agent.schema import ValidationError, validate_game_state

TOKEN = os.environ["POKER_ENGINE_TOKEN"]
if len(TOKEN) < 32:
    raise RuntimeError("POKER_ENGINE_TOKEN must contain at least 32 characters")
if sys.version_info[:2] != (3, 9) or platform.machine() != "x86_64":
    raise RuntimeError("Original engine requires x86_64 CPython 3.9")
REVISION = (ROOT / "REVISION").read_text().strip()
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(20)

    def log_message(self, format: str, *args) -> None:
        pass

    def respond(self, status: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authenticated(self) -> bool:
        actual = self.headers.get("Authorization", "").encode("utf-8")
        if hmac.compare_digest(actual, ("Bearer " + TOKEN).encode("utf-8")):
            return True
        self.respond(401, {"error": "Unauthorized"})
        return False

    def do_GET(self) -> None:
        if not self.authenticated():
            return
        if self.path != "/health":
            self.respond(404, {"error": "Not found"})
            return
        self.respond(200, {"ok": True, "engine": "PokerSkill-author", "python": platform.python_version(), "revision": REVISION})

    def do_POST(self) -> None:
        if not self.authenticated():
            return
        if self.path != "/analyze":
            self.respond(404, {"error": "Not found"})
            return
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            self.respond(415, {"error": "JSON required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                self.respond(413, {"error": "Invalid request size"})
                return
            payload = json.loads(self.rfile.read(length))
            state = validate_game_state(payload["state"])
            with LOCK:
                started = time.perf_counter()
                result = generate_prompt(state)
                elapsed = round((time.perf_counter() - started) * 1000, 3)
            self.respond(200, {"result": result, "analysisMs": elapsed})
        except (ValueError, TypeError, KeyError, ValidationError):
            self.respond(400, {"error": "Invalid poker state"})
        except Exception:
            self.respond(500, {"error": "Author analysis failed"})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("POKER_ENGINE_PORT", "8173"))), Handler)
    server.daemon_threads = True
    print(json.dumps({"ready": True, "port": server.server_port, "python": platform.python_version()}), flush=True)
    server.serve_forever()

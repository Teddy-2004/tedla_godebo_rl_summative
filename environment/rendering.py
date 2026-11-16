"""three.js visualisation for WolaitaCorpusEnv.

A tiny stdlib HTTP server serves a three.js scene (environment/web/) and streams
environment states to it as JSON. No extra Python dependencies and no internet
needed (three.js is vendored).

Endpoints (also the integration path for a web / mobile front-end):
    GET  /                 -> 3D viewer
    GET  /state?since=N    -> {"n": total, "states": [...]} states pushed after index N
    POST /act              -> {"observation": [19 floats]} -> {"action": id, "name": str}
                              (available when a policy is attached with set_policy)
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WEB_DIR = Path(__file__).parent / "web"
MIME = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


class ThreeJSViewer:
    def __init__(self, port: int = 8765, open_browser: bool = True):
        self.states: list[dict] = []
        self.lock = threading.Lock()
        self.policy_fn = None
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):          # keep the terminal clean for agent logs
                pass

            def _send(self, code, body: bytes, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                url = urlparse(self.path)
                if url.path == "/state":
                    since = int(parse_qs(url.query).get("since", ["0"])[0])
                    with viewer.lock:
                        body = {"n": len(viewer.states), "states": viewer.states[since:]}
                    return self._send(200, json.dumps(body).encode())
                name = "index.html" if url.path == "/" else url.path.lstrip("/")
                f = (WEB_DIR / name).resolve()
                if WEB_DIR.resolve() in f.parents and f.is_file():
                    return self._send(200, f.read_bytes(), MIME.get(f.suffix, "text/plain"))
                self._send(404, b"{}")

            def do_POST(self):
                if self.path != "/act" or viewer.policy_fn is None:
                    return self._send(404, b'{"error": "no policy attached"}')
                n = int(self.headers.get("Content-Length", 0))
                try:
                    obs = json.loads(self.rfile.read(n))["observation"]
                    out = viewer.policy_fn(obs)
                    self._send(200, json.dumps(out).encode())
                except Exception as e:  # noqa: BLE001
                    self._send(400, json.dumps({"error": str(e)}).encode())

        self.server = None
        for p in range(port, port + 20):        # pick the first free port
            try:
                self.server = ThreadingHTTPServer(("127.0.0.1", p), Handler)
                break
            except OSError:
                continue
        if self.server is None:
            raise RuntimeError("no free port for the viewer")
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"[viewer] 3D visualisation at {self.url}", flush=True)
        if open_browser:
            webbrowser.open(self.url)

    def set_policy(self, fn):
        """fn(list[float]) -> {"action": int, "name": str}; exposed at POST /act."""
        self.policy_fn = fn

    def push(self, state: dict):
        with self.lock:
            self.states.append(state)

    def close(self):
        self.server.shutdown()
        self.server.server_close()

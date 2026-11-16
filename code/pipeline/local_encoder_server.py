#!/usr/bin/env python3
"""本地 BGE-M3 编码 sidecar：直接 import 线上共享模块，暴露 /embed。

线上后端（无 torch）通过 HTTP 调用本进程；离线脚本直接 import 同一模块。
两条路径执行的是同一份代码（backend/src/services/bge_encoder.py）。
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BACKEND_ROOT = Path("/Volumes/Lenovo/26大创/小鲸OrcaAI/backend")
sys.path.insert(0, str(BACKEND_ROOT))

from src.services import bge_encoder  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 静默
        return

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"status": "ok", **bge_encoder.fingerprint()})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/embed":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send(400, {"error": f"bad json: {exc}"})
            return
        texts = body.get("texts") or ([body["text"]] if body.get("text") else None)
        if not texts:
            self._send(400, {"error": "texts required"})
            return
        try:
            vectors = bge_encoder.encode(list(texts))
            self._send(
                200,
                {
                    "vectors": vectors.tolist(),
                    "dim": int(vectors.shape[1]),
                    "count": int(vectors.shape[0]),
                    "version": bge_encoder.VERSION,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    # 预热：确保首个请求不承担加载成本
    bge_encoder.encode(["预热"])
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"listening on http://{args.host}:{args.port}/embed", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

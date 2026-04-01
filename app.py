"""Unified application entrypoint.

Run with:
    python3 app.py

What it does:
1) Ensures the TypeScript frontend is built into ``dist/``.
2) Starts a Flask server that serves the built web UI.
3) Exposes backend API endpoints from the Python DSP core.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from backend.core import compute_metrics, get_filter_classes, run_pipeline

ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def ensure_frontend_build() -> None:
    """Build the frontend if dist assets are missing."""
    if (DIST_DIR / "index.html").exists():
        return

    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm is not installed. Install Node.js + npm, then run python3 app.py again.")

    print("[startup] dist/ not found. Installing frontend dependencies...")
    _run([npm, "install"])
    print("[startup] Building frontend...")
    _run([npm, "run", "build"])


def create_app() -> Flask:
    ensure_frontend_build()

    app = Flask(__name__, static_folder=str(DIST_DIR), static_url_path="")

    @app.get("/api/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.get("/api/filters")
    def filters() -> Any:
        payload = []
        for cls in get_filter_classes():
            inst = cls()
            payload.append({
                "class": cls.__name__,
                "name": cls.name,
                "params": inst.param_spec(),
            })
        return jsonify(payload)

    @app.post("/api/process")
    def process_audio() -> Any:
        body = request.get_json(force=True)
        sr = int(body.get("sample_rate", 16000))
        raw = np.asarray(body.get("raw", []), dtype=np.float32)
        ref = np.asarray(body.get("reference", []), dtype=np.float32) if body.get("reference") is not None else None

        pipeline_cfg = body.get("pipeline", [])
        class_map = {cls.__name__: cls for cls in get_filter_classes()}
        pipeline = []
        for item in pipeline_cfg:
            class_name = item.get("class")
            params = item.get("params", {})
            cls = class_map.get(class_name)
            if cls is None:
                return jsonify({"error": f"Unknown filter class: {class_name}"}), 400
            pipeline.append(cls(**params))

        processed = run_pipeline(raw, sr, pipeline)
        response: dict[str, Any] = {"processed": processed.tolist()}
        if ref is not None and len(ref) > 0:
            response["metrics"] = compute_metrics(ref, processed, sr)
        return jsonify(response)

    @app.get("/")
    def index() -> Any:
        return send_from_directory(DIST_DIR, "index.html")

    @app.get("/<path:path>")
    def spa(path: str) -> Any:
        target = DIST_DIR / path
        if target.exists() and target.is_file():
            return send_from_directory(DIST_DIR, path)
        return send_from_directory(DIST_DIR, "index.html")

    return app


def main() -> None:
    app = create_app()
    port = 8000
    print(f"[startup] Web app is running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        sys.exit(1)

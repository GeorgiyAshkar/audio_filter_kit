"""Single-command launcher for React application.

Run:
    python3 app.py

This script:
1) Installs frontend dependencies (npm install) if needed.
2) Runs the React/Vite app.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def ensure_node_tools() -> str:
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm not found. Install Node.js + npm and retry.")
    return npm


def ensure_frontend_deps(npm: str) -> None:
    node_modules = ROOT / "node_modules"
    if node_modules.exists():
        return
    print("[startup] Installing frontend dependencies...")
    run([npm, "install"])


def start_react_app(npm: str) -> None:
    print("[startup] Starting React app on http://127.0.0.1:5173")
    run([npm, "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"])


def main() -> None:
    npm = ensure_node_tools()
    ensure_frontend_deps(npm)
    start_react_app(npm)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        sys.exit(1)

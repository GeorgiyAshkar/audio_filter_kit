"""
Top‑level package for the DAS speech filtering application.

This package provides a modular backend with filter implementations,
core utilities and a PyQt frontend. Adding this file allows
`import das_speech_app` and submodules to work when the package is
accessed via `sys.path` modifications within the frontend. The file
is intentionally minimal.
"""

__all__ = ["backend", "frontend"]
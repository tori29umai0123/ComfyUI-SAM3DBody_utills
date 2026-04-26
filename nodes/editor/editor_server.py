# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""HTTP routes for the editor pages.

Two route prefixes:

  /sam3d/editor/...   — page HTML and tmp file delivery (mesh.obj / mask.png)
  /sam3d/api/...      — JSON endpoints used by the editor frontend

The static JS / CSS assets under ``web/editor/static/`` are served
automatically by ComfyUI through ``WEB_DIRECTORY = "./web"`` at
``/extensions/ComfyUI-SAM3DBody_utills/editor/static/...`` — we do not
re-route those.
"""

from __future__ import annotations

import logging
import mimetypes
import os
from pathlib import Path

from aiohttp import web
from server import PromptServer

log = logging.getLogger("SAM3DBody.editor")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WEB_EDITOR_DIR = _REPO_ROOT / "web" / "editor"
_TMP_DIR = _REPO_ROOT / "nodes" / "editor_tmp"
_TMP_DIR.mkdir(parents=True, exist_ok=True)


def _safe_join(base: Path, relative: str) -> Path | None:
    """Join ``relative`` onto ``base`` and refuse traversal escapes."""
    if not relative:
        return None
    target = (base / relative).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError:
        return None
    return target


def tmp_dir() -> Path:
    return _TMP_DIR


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

routes = PromptServer.instance.routes


def _serve_html(name: str) -> web.Response:
    path = _WEB_EDITOR_DIR / name
    if not path.is_file():
        return web.Response(
            status=404,
            text=(
                f"editor page not found: {name}\n"
                f"expected at: {path}\n"
                f"(This is normal if the frontend hasn't been installed yet.)"
            ),
        )
    # Force fresh fetch on every open so the iframe always picks up the
    # latest ``?v=N`` query strings on its <script>/<link> tags. Without
    # this, browsers cache the HTML and keep serving stale references to
    # older JS/CSS bundles — visible to users as "the new buttons aren't
    # showing up after the update".
    return web.FileResponse(
        path,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@routes.get("/sam3d/editor/pose")
async def serve_pose_editor(_request: web.Request) -> web.Response:
    return _serve_html("pose_editor.html")


@routes.get("/sam3d/editor/character")
async def serve_character_editor(_request: web.Request) -> web.Response:
    return _serve_html("character_editor.html")


@routes.get("/sam3d/editor/tmp/{name:.*}")
async def serve_tmp_file(request: web.Request) -> web.Response:
    name = request.match_info.get("name", "")
    target = _safe_join(_TMP_DIR, name)
    if target is None or not target.is_file():
        return web.Response(status=404, text=f"not found: {name}")
    ctype, _ = mimetypes.guess_type(target.name)
    return web.FileResponse(
        target,
        headers={
            "Cache-Control": "no-store",
            "Content-Type": ctype or "application/octet-stream",
        },
    )


# ---------------------------------------------------------------------------
# API routes are defined in editor_api.py (imported below). Keeping the
# heavy SAM3DBody / BiRefNet imports out of this module keeps the page
# routes registered even when the inference stack fails to load — useful
# for diagnosing setup issues from the browser console.
# ---------------------------------------------------------------------------

try:
    from . import editor_api  # noqa: F401  (registers /sam3d/api/* routes)
    log.info("editor API routes registered")
except Exception as exc:  # noqa: BLE001
    log.exception("editor API routes failed to register: %s", exc)


print("[SAM3DBody] Registered editor routes:")
print("[SAM3DBody]   GET  /sam3d/editor/pose")
print("[SAM3DBody]   GET  /sam3d/editor/character")
print("[SAM3DBody]   GET  /sam3d/editor/tmp/{name}")

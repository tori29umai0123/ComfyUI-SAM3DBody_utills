# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""JSON API endpoints for the editor frontend.

Mounts under ``/sam3d/api/...``. Mirrors the FastAPI routes of the
upstream standalone app one-to-one (image→pose pipeline, slider-driven
re-render, preset I/O, slider schema), translated to aiohttp so they
run in-process with ComfyUI's PromptServer.

Save/edit endpoints from the upstream Preset Admin tab are intentionally
**not** ported — pack switching is handled by editing
``config.ini`` and reloading ComfyUI.
"""

from __future__ import annotations

import asyncio
import io
import logging
import traceback
from typing import Any

from aiohttp import web
from PIL import Image
from server import PromptServer

from .editor_app import paths as _paths
from .editor_app import preset_pack
from .editor_app.character_shape import (
    BODY_PARAM_KEYS,
    BONE_LENGTH_KEYS,
    discover_blendshape_names,
)
from .editor_app.pipeline import _normalize_input_image, run_image_to_obj
from .editor_app.renderer import render_from_session

log = logging.getLogger("SAM3DBody.editor.api")
routes = PromptServer.instance.routes


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------

@routes.get("/sam3d/api/ping")
async def api_ping(_request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "sam3d-editor"})


# ---------------------------------------------------------------------------
# Slider schema (used to build the Make-tab UI)
# ---------------------------------------------------------------------------

@routes.get("/sam3d/api/slider_schema")
async def api_slider_schema(_request: web.Request) -> web.Response:
    pack = preset_pack.active_pack_paths()
    bs_names = list(discover_blendshape_names(pack.npz_path))
    return web.json_response({
        "pack": pack.pack_dir.name,
        "body_params": [
            {"key": k, "min": -5.0, "max": 5.0, "step": 0.01, "default": 0.0}
            for k in BODY_PARAM_KEYS
        ],
        "bone_lengths": [
            {"key": "torso", "min": 0.3, "max": 1.8, "step": 0.01, "default": 1.0},
            {"key": "neck",  "min": 0.3, "max": 2.0, "step": 0.01, "default": 1.0},
            {"key": "arm",   "min": 0.3, "max": 2.0, "step": 0.01, "default": 1.0},
            {"key": "leg",   "min": 0.3, "max": 2.0, "step": 0.01, "default": 1.0},
        ],
        "blendshapes": [
            {"key": name, "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0}
            for name in bs_names
        ],
    })


# ---------------------------------------------------------------------------
# Preset I/O (read-only)
# ---------------------------------------------------------------------------

@routes.get("/sam3d/api/presets")
async def api_presets(_request: web.Request) -> web.Response:
    return web.json_response({"presets": preset_pack.list_presets()})


@routes.get("/sam3d/api/preset/{name}")
async def api_preset(request: web.Request) -> web.Response:
    name = request.match_info.get("name", "")
    try:
        return web.json_response(preset_pack.load_preset(name))
    except FileNotFoundError:
        return web.json_response({"error": f"preset {name!r} not found"}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)


# ---------------------------------------------------------------------------
# Image → segmentation → pose
# ---------------------------------------------------------------------------

@routes.post("/sam3d/api/process")
async def api_process(request: web.Request) -> web.Response:
    inference_type = request.query.get("inference_type", "full")
    reader = await request.multipart()
    raw: bytes | None = None
    while True:
        field = await reader.next()
        if field is None:
            break
        if field.name == "image":
            raw = await field.read(decode=False)
            break
    if not raw:
        return web.json_response({"error": "missing 'image' multipart field"}, status=400)

    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
        pil = _normalize_input_image(pil)
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"error": f"could not decode image: {exc}"}, status=400)

    try:
        result = await asyncio.to_thread(
            run_image_to_obj,
            pil,
            inference_type=inference_type,
            use_segmentation=True,
            segmentation_backend="birefnet_lite",
            confidence_threshold=0.5,
            min_width_pixels=0,
            min_height_pixels=0,
            device_mode=None,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline failed")
        return web.json_response({
            "error": str(exc),
            "trace": traceback.format_exc(limit=8),
        }, status=500)

    return web.json_response({
        "job_id": result.job_id,
        "obj_url": result.obj_url,
        "mask_url": result.mask_url,
        "bbox_xyxy": result.bbox_xyxy,
        "width": result.width,
        "height": result.height,
        "elapsed_sec": round(result.elapsed_sec, 3),
        "best_score": result.best_score,
        "num_candidates": result.num_detections,
        "pose_json": result.pose_json,
    })


# ---------------------------------------------------------------------------
# Re-render (slider drag)
# ---------------------------------------------------------------------------

@routes.post("/sam3d/api/render")
async def api_render(request: web.Request) -> web.Response:
    try:
        payload: dict[str, Any] = await request.json()
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"error": f"invalid json body: {exc}"}, status=400)

    job_id = str(payload.get("job_id") or "")
    if not job_id:
        return web.json_response({"error": "job_id is required"}, status=400)
    settings = payload.get("settings") or {}

    try:
        result = await asyncio.to_thread(render_from_session, job_id, settings)
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except Exception as exc:  # noqa: BLE001
        log.exception("render failed")
        return web.json_response({
            "error": str(exc),
            "trace": traceback.format_exc(limit=8),
        }, status=500)

    return web.json_response({
        "job_id": result.job_id,
        "obj_url": result.obj_url,
        "elapsed_sec": round(result.elapsed_sec, 3),
        "settings": result.settings,
        "humanoid_skeleton": result.humanoid_skeleton,
    })


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@routes.get("/sam3d/api/active_pack")
async def api_active_pack(_request: web.Request) -> web.Response:
    pack = preset_pack.active_pack_paths()
    return web.json_response({
        "name": pack.pack_dir.name,
        "pack_dir": str(pack.pack_dir),
        "npz_path": str(pack.npz_path),
        "npz_exists": pack.npz_path.is_file(),
        "chara_settings_dir": str(pack.chara_settings_dir),
        "models_dir": str(_paths.MODELS_DIR),
        "tmp_dir": str(_paths.TMP_DIR),
    })


print("[SAM3DBody] Registered editor API routes:")
for line in (
    "  GET  /sam3d/api/ping",
    "  GET  /sam3d/api/slider_schema",
    "  GET  /sam3d/api/presets",
    "  GET  /sam3d/api/preset/{name}",
    "  POST /sam3d/api/process",
    "  POST /sam3d/api/render",
    "  GET  /sam3d/api/active_pack",
):
    print(f"[SAM3DBody]{line}")

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
import json
import logging
import os
import traceback
from typing import Any

from aiohttp import web
from PIL import Image
from server import PromptServer

from .editor_app import plus_session
from .editor_app import paths as _paths
from .editor_app import preset_pack
from .editor_app.body_preset_shape import (
    BODY_PARAM_KEYS,
    BONE_LENGTH_KEYS,
    discover_blendshape_names,
)
from .editor_app import sam2_segmenter
from .editor_app.plus_pipeline import PersonInput, run_plus_image_to_obj
from .editor_app.plus_renderer import render_plus_from_session
from .editor_app import plus_export
from .editor_app.pipeline import _normalize_input_image, run_image_to_obj
from .editor_app.renderer import render_from_session
from .editor_app.units import read_unit_config

# preset_pack module — used for the [units] read/write API. Shadowing the
# editor-local ``preset_pack`` import would be confusing, so import the
# project-root one under a distinct alias.
from ..preset_pack import (
    DEFAULT_ADULT_HEIGHT_M as _DEFAULT_ADULT_HEIGHT_M,
    VALID_DISPLAY_UNITS as _VALID_DISPLAY_UNITS,
    clean_blender_exe_path as _clean_blender_exe_path,
    get_blender_exe_path as _get_blender_exe_path,
    set_display_unit as _set_display_unit,
)

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
    raw_lhand: bytes | None = None
    raw_rhand: bytes | None = None
    while True:
        field = await reader.next()
        if field is None:
            break
        if field.name == "image":
            raw = await field.read(decode=False)
        elif field.name == "left_hand_image":
            raw_lhand = await field.read(decode=False)
        elif field.name == "right_hand_image":
            raw_rhand = await field.read(decode=False)
    if not raw:
        return web.json_response({"error": "missing 'image' multipart field"}, status=400)

    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
        pil = _normalize_input_image(pil)
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"error": f"could not decode image: {exc}"}, status=400)

    def _decode_optional(b: bytes | None) -> Image.Image | None:
        if not b:
            return None
        try:
            im = Image.open(io.BytesIO(b))
            im.load()
            return _normalize_input_image(im)
        except Exception as exc:  # noqa: BLE001
            log.warning("hand image decode failed: %s", exc)
            return None

    pil_lhand = _decode_optional(raw_lhand)
    pil_rhand = _decode_optional(raw_rhand)

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
            left_hand_image=pil_lhand,
            right_hand_image=pil_rhand,
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
# Pose Editor + — image → per-person segmentation → pose
# ---------------------------------------------------------------------------

def _decode_optional_pil(b: bytes | None) -> Image.Image | None:
    if not b:
        return None
    try:
        im = Image.open(io.BytesIO(b))
        im.load()
        return _normalize_input_image(im)
    except Exception as exc:  # noqa: BLE001
        log.warning("optional image decode failed: %s", exc)
        return None


@routes.post("/sam3d/api/plus/process")
async def api_multi_process(request: web.Request) -> web.Response:
    """Run SAM 2 + SAM 3D Body on a single image with N user-drawn boxes.

    Multipart fields:
        image      — main PNG/JPEG bytes (required)
        payload    — JSON: {"persons": [...], "inference_type": "full"|"body"|"hand"}
        lhand_<id> — optional left-hand crop bytes (one per person id)
        rhand_<id> — optional right-hand crop bytes (one per person id)
    """
    inference_type_default = request.query.get("inference_type", "full")

    main_bytes: bytes | None = None
    payload_text: str | None = None
    lhands: dict[str, bytes] = {}
    rhands: dict[str, bytes] = {}

    reader = await request.multipart()
    while True:
        field = await reader.next()
        if field is None:
            break
        name = field.name or ""
        if name == "image":
            main_bytes = await field.read(decode=False)
        elif name == "payload":
            payload_text = (await field.read(decode=True)).decode("utf-8", "replace")
        elif name.startswith("lhand_"):
            lhands[name[len("lhand_"):]] = await field.read(decode=False)
        elif name.startswith("rhand_"):
            rhands[name[len("rhand_"):]] = await field.read(decode=False)

    if not main_bytes:
        return web.json_response({"error": "missing 'image' multipart field"}, status=400)
    if not payload_text:
        return web.json_response({"error": "missing 'payload' multipart field"}, status=400)

    try:
        payload = json.loads(payload_text)
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"error": f"invalid payload JSON: {exc}"}, status=400)

    persons_raw = payload.get("persons") if isinstance(payload, dict) else None
    if not isinstance(persons_raw, list) or not persons_raw:
        return web.json_response({"error": "payload.persons must be a non-empty list"}, status=400)
    inference_type = str(payload.get("inference_type") or inference_type_default)

    persons: list[PersonInput] = []
    for entry in persons_raw:
        if not isinstance(entry, dict):
            return web.json_response({"error": "each persons[] entry must be a dict"}, status=400)
        try:
            pid = str(entry["id"])
            box = [float(v) for v in entry["bbox_xyxy"]]
        except (KeyError, TypeError, ValueError) as exc:
            return web.json_response(
                {"error": f"persons[] requires 'id' and 'bbox_xyxy': {exc}"},
                status=400,
            )
        if len(box) != 4:
            return web.json_response({"error": f"bbox_xyxy must have 4 values, got {len(box)}"}, status=400)
        try:
            height_m = float(entry.get("height_m") or 0.0)
        except (TypeError, ValueError):
            height_m = 0.0
        if height_m <= 0:
            height_m = float(_DEFAULT_ADULT_HEIGHT_M)
        def _coerce(raw_list) -> list[tuple[float, float, float, float]]:
            out: list[tuple[float, float, float, float]] = []
            if not isinstance(raw_list, list):
                return out
            for nb in raw_list:
                try:
                    vec = [float(v) for v in nb]
                except (TypeError, ValueError):
                    continue
                if len(vec) == 4:
                    out.append((vec[0], vec[1], vec[2], vec[3]))
            return out

        persons.append(PersonInput(
            person_id=pid,
            bbox_xyxy=(box[0], box[1], box[2], box[3]),
            height_m=height_m,
            lhand_image=_decode_optional_pil(lhands.get(pid)),
            rhand_image=_decode_optional_pil(rhands.get(pid)),
            additional_bboxes=_coerce(entry.get("additional_bboxes")),
            negative_bboxes=_coerce(entry.get("negative_bboxes")),
        ))

    try:
        pil = Image.open(io.BytesIO(main_bytes))
        pil.load()
        pil = _normalize_input_image(pil)
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"error": f"could not decode image: {exc}"}, status=400)

    try:
        result = await asyncio.to_thread(
            run_plus_image_to_obj,
            pil,
            persons,
            inference_type=inference_type,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("plus pipeline failed")
        return web.json_response({
            "error": str(exc),
            "trace": traceback.format_exc(limit=8),
        }, status=500)

    return web.json_response({
        "plus_job_id": result.plus_job_id,
        "obj_url": result.obj_url,
        "width": result.width,
        "height": result.height,
        "elapsed_sec": round(result.elapsed_sec, 3),
        "per_person": result.per_person,
    })


@routes.post("/sam3d/api/plus/render")
async def api_multi_render(request: web.Request) -> web.Response:
    """Re-render a cached multi session with new per-person settings.

    JSON body:
        {
          "plus_job_id": "abc123",
          "per_person_settings": {
              "p0": {"height_m": 1.70},
              "p1": {"height_m": 1.55}
          }
        }
    """
    try:
        payload: dict[str, Any] = await request.json()
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"error": f"invalid json body: {exc}"}, status=400)

    plus_job_id = str(payload.get("plus_job_id") or "")
    if not plus_job_id:
        return web.json_response({"error": "plus_job_id is required"}, status=400)
    settings = {
        "per_person_settings": payload.get("per_person_settings") or {},
        "active_person_id": payload.get("active_person_id"),
    }

    try:
        result = await asyncio.to_thread(render_plus_from_session, plus_job_id, settings)
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except Exception as exc:  # noqa: BLE001
        log.exception("multi render failed")
        return web.json_response({
            "error": str(exc),
            "trace": traceback.format_exc(limit=8),
        }, status=500)

    sess = plus_session.get(result.plus_job_id)
    person_ids = [s.person_id for s in sess.persons] if sess else []
    return web.json_response({
        "plus_job_id": result.plus_job_id,
        "obj_url": result.obj_url,
        "elapsed_sec": round(result.elapsed_sec, 3),
        "per_person": [
            {
                "id": pid,
                "vertex_range": list(vr),
                "face_range": list(fr),
                "height_m": h,
                "humanoid_skeleton": sk,
                "settings": st,
                "hip_world": hw,
            }
            for pid, vr, fr, h, sk, st, hw in zip(
                person_ids,
                result.per_person_vertex_ranges,
                result.per_person_face_ranges,
                result.per_person_height_m,
                result.per_person_skeletons,
                result.per_person_settings,
                result.per_person_hip_world,
            )
        ],
    })


@routes.post("/sam3d/api/sam2_preview")
async def api_sam2_preview(request: web.Request) -> web.Response:
    """Run SAM 2 against the user-drawn bboxes and return per-person
    masked-and-cropped PNG URLs. Called by the editor every time a bbox
    is added / removed so the sidebar thumbnail reflects what SAM 2 will
    actually feed the SAM 3D Body inference path.

    Multipart fields:
        image    — main PNG/JPEG bytes (required)
        payload  — JSON: {"persons": [{"id": str, "bbox_xyxy": [x1,y1,x2,y2]}]}
    """
    main_bytes: bytes | None = None
    payload_text: str | None = None

    reader = await request.multipart()
    while True:
        field = await reader.next()
        if field is None:
            break
        name = field.name or ""
        if name == "image":
            main_bytes = await field.read(decode=False)
        elif name == "payload":
            payload_text = (await field.read(decode=True)).decode("utf-8", "replace")

    if not main_bytes:
        return web.json_response({"error": "missing 'image' multipart field"}, status=400)
    if not payload_text:
        return web.json_response({"error": "missing 'payload' multipart field"}, status=400)

    try:
        payload = json.loads(payload_text)
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"error": f"invalid payload JSON: {exc}"}, status=400)
    persons_raw = payload.get("persons") if isinstance(payload, dict) else None
    if not isinstance(persons_raw, list) or not persons_raw:
        return web.json_response({"error": "payload.persons must be a non-empty list"}, status=400)

    def _coerce_box_list(raw) -> list[list[float]]:
        out: list[list[float]] = []
        if not isinstance(raw, list):
            return out
        for nb in raw:
            try:
                vec = [float(v) for v in nb]
            except (TypeError, ValueError):
                continue
            if len(vec) == 4:
                out.append(vec)
        return out

    person_ids: list[str] = []
    boxes_list: list[list[float]] = []
    add_boxes_list: list[list[list[float]]] = []
    neg_boxes_list: list[list[list[float]]] = []
    for entry in persons_raw:
        if not isinstance(entry, dict):
            return web.json_response({"error": "each persons[] entry must be a dict"}, status=400)
        try:
            pid = str(entry["id"])
            box = [float(v) for v in entry["bbox_xyxy"]]
        except (KeyError, TypeError, ValueError) as exc:
            return web.json_response(
                {"error": f"persons[] requires 'id' and 'bbox_xyxy': {exc}"}, status=400,
            )
        if len(box) != 4:
            return web.json_response({"error": "bbox_xyxy must have 4 values"}, status=400)
        person_ids.append(pid)
        boxes_list.append(box)
        add_boxes_list.append(_coerce_box_list(entry.get("additional_bboxes")))
        neg_boxes_list.append(_coerce_box_list(entry.get("negative_bboxes")))

    try:
        pil = Image.open(io.BytesIO(main_bytes))
        pil.load()
        pil = _normalize_input_image(pil)
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"error": f"could not decode image: {exc}"}, status=400)

    import numpy as np

    rgb = np.asarray(pil, dtype=np.uint8)
    bboxes = np.asarray(boxes_list, dtype=np.float32).reshape(-1, 4)
    neg_arrs = [
        np.asarray(nb, dtype=np.float32).reshape(-1, 4) if nb else None
        for nb in neg_boxes_list
    ]
    add_arrs = [
        np.asarray(ab, dtype=np.float32).reshape(-1, 4) if ab else None
        for ab in add_boxes_list
    ]

    try:
        mask_result = await asyncio.to_thread(
            sam2_segmenter.predict_masks, rgb, bboxes, neg_arrs, add_arrs,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("sam2_preview failed")
        return web.json_response({
            "error": str(exc),
            "trace": traceback.format_exc(limit=8),
        }, status=500)

    h, w = rgb.shape[:2]
    out_persons: list[dict[str, Any]] = []
    import uuid as _uuid
    version = _uuid.uuid4().hex[:8]
    for i, (pid, box, mask) in enumerate(
        zip(person_ids, bboxes, mask_result.masks)
    ):
        x1 = max(0, int(round(float(box[0]))))
        y1 = max(0, int(round(float(box[1]))))
        x2 = min(w, int(round(float(box[2]))))
        y2 = min(h, int(round(float(box[3]))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            out_persons.append({
                "id": pid, "trimmed_url": None, "mask_score": 0.0,
            })
            continue
        rgb_crop = rgb[y1:y2, x1:x2]
        mask_crop = (mask[y1:y2, x1:x2] > 0).astype(np.uint8)
        rgba = np.dstack([rgb_crop, mask_crop * 255]).astype(np.uint8)
        fname = f"sam2_preview_{pid}.png"
        out_path = _paths.tmp_dir() / fname
        Image.fromarray(rgba, mode="RGBA").save(out_path)
        score = float(mask_result.scores[i]) if i < mask_result.scores.size else 0.0
        out_persons.append({
            "id": pid,
            "trimmed_url": _paths.tmp_url(fname, version=version),
            "mask_score": score,
        })

    return web.json_response({"per_person": out_persons})


@routes.post("/sam3d/api/object_crop")
async def api_object_crop(request: web.Request) -> web.Response:
    """Crop a single bbox from an image using SAM 2 and return a
    transparent-PNG URL for the masked region.

    Used by the Pose Editor +'s "object" overlay feature: the user
    draws a bbox over an item in the input image (e.g., something the
    person is holding), the backend runs SAM 2 with that box as a
    positive prompt, and the resulting RGBA crop is composited on top
    of the rendered scene at confirm time.
    """
    main_bytes: bytes | None = None
    payload_text: str | None = None

    reader = await request.multipart()
    while True:
        field = await reader.next()
        if field is None:
            break
        name = field.name or ""
        if name == "image":
            main_bytes = await field.read(decode=False)
        elif name == "payload":
            payload_text = (await field.read(decode=True)).decode("utf-8", "replace")

    if not main_bytes:
        return web.json_response({"error": "missing 'image' multipart field"}, status=400)
    if not payload_text:
        return web.json_response({"error": "missing 'payload' multipart field"}, status=400)

    try:
        payload = json.loads(payload_text)
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"error": f"invalid payload JSON: {exc}"}, status=400)
    try:
        bbox = [float(v) for v in payload.get("bbox_xyxy", [])]
    except (TypeError, ValueError) as exc:
        return web.json_response({"error": f"bbox_xyxy must be 4 floats: {exc}"}, status=400)
    if len(bbox) != 4:
        return web.json_response({"error": "bbox_xyxy must have 4 values"}, status=400)
    obj_name = str(payload.get("name") or "obj")

    def _coerce_box_list(raw) -> list[list[float]]:
        out: list[list[float]] = []
        if not isinstance(raw, list):
            return out
        for nb in raw:
            try:
                vec = [float(v) for v in nb]
            except (TypeError, ValueError):
                continue
            if len(vec) == 4:
                out.append(vec)
        return out

    add_bboxes = _coerce_box_list(payload.get("additional_bboxes"))
    neg_bboxes = _coerce_box_list(payload.get("negative_bboxes"))

    try:
        pil = Image.open(io.BytesIO(main_bytes))
        pil.load()
        pil = _normalize_input_image(pil)
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"error": f"could not decode image: {exc}"}, status=400)

    import numpy as np
    rgb = np.asarray(pil, dtype=np.uint8)
    bboxes = np.asarray([bbox], dtype=np.float32)
    add_arr = np.asarray(add_bboxes, dtype=np.float32).reshape(-1, 4) if add_bboxes else None
    neg_arr = np.asarray(neg_bboxes, dtype=np.float32).reshape(-1, 4) if neg_bboxes else None

    try:
        mask_result = await asyncio.to_thread(
            sam2_segmenter.predict_masks,
            rgb, bboxes,
            [neg_arr], [add_arr],
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("object_crop failed")
        return web.json_response({
            "error": str(exc),
            "trace": traceback.format_exc(limit=8),
        }, status=500)

    h, w = rgb.shape[:2]
    # Crop to the mask's actual bbox (not just the primary box) so any
    # additional positive bbox extending past the primary still ends up
    # in the cropped PNG.
    full_mask = (mask_result.masks[0] > 0).astype(np.uint8)
    if not full_mask.any():
        return web.json_response({"error": "mask is empty"}, status=400)
    ys, xs = np.where(full_mask > 0)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    if x2 - x1 < 2 or y2 - y1 < 2:
        return web.json_response({"error": "mask too small"}, status=400)

    rgb_crop = rgb[y1:y2, x1:x2]
    mask_crop = full_mask[y1:y2, x1:x2]
    rgba = np.dstack([rgb_crop, mask_crop * 255]).astype(np.uint8)

    import uuid as _uuid
    fname = f"object_{obj_name}_{_uuid.uuid4().hex[:6]}.png"
    out_path = _paths.tmp_dir() / fname
    Image.fromarray(rgba, mode="RGBA").save(out_path)

    return web.json_response({
        "obj_url": _paths.tmp_url(fname),
        "natural_width": int(x2 - x1),
        "natural_height": int(y2 - y1),
        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
    })


@routes.post("/sam3d/api/plus/drop")
async def api_multi_drop(request: web.Request) -> web.Response:
    try:
        payload: dict[str, Any] = await request.json()
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"error": f"invalid json body: {exc}"}, status=400)
    plus_job_id = str(payload.get("plus_job_id") or "")
    if not plus_job_id:
        return web.json_response({"error": "plus_job_id is required"}, status=400)
    plus_session.drop(plus_job_id)
    return web.json_response({"ok": True, "dropped": plus_job_id})


@routes.post("/sam3d/api/plus/reconcile_persons")
async def api_plus_reconcile_persons(request: web.Request) -> web.Response:
    """Drop / re-order person slots inside an existing plus session so the
    backend matches the frontend after a person is removed.

    Body: ``{ plus_job_id, keep_old_ids: [old_id, old_id, ...] }`` — the
    old IDs of the slots the frontend wants to keep, in the order they
    will be renumbered to ``p0, p1, …`` on the client. Slots not in the
    list are dropped; matching slots are renamed to the same sequential
    IDs the frontend uses, preserving the cached pose / settings so the
    next ``/plus/render`` reuses the existing inference output.
    """
    try:
        payload: dict[str, Any] = await request.json()
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"error": f"invalid json body: {exc}"}, status=400)
    plus_job_id = str(payload.get("plus_job_id") or "")
    keep_raw = payload.get("keep_old_ids")
    if not plus_job_id:
        return web.json_response({"error": "plus_job_id is required"}, status=400)
    if not isinstance(keep_raw, list):
        return web.json_response(
            {"error": "keep_old_ids must be a list of strings"}, status=400,
        )
    keep_ids: list[str] = [str(x) for x in keep_raw if isinstance(x, (str, int))]

    sess = plus_session.get(plus_job_id)
    if sess is None:
        return web.json_response({"error": f"session {plus_job_id!r} not found"}, status=404)

    by_id = {slot.person_id: slot for slot in sess.persons}
    new_slots = []
    for new_idx, old_id in enumerate(keep_ids):
        slot = by_id.get(old_id)
        if slot is None:
            continue
        slot.person_id = f"p{new_idx}"
        new_slots.append(slot)
    sess.persons = new_slots

    # Re-key any per-person overrides cached on the session so the next
    # render call picks up the same height/transform tweaks.
    if sess.overrides:
        new_overrides: dict[str, Any] = {}
        for new_idx, old_id in enumerate(keep_ids):
            ov = sess.overrides.get(old_id)
            if ov is not None:
                new_overrides[f"p{new_idx}"] = ov
        sess.overrides = new_overrides

    # The renderer's geometry cache keys off the full per-person settings
    # blob; drop it so the next render rebuilds against the new slot list.
    try:
        from .editor_app.plus_renderer import invalidate_cache
        invalidate_cache()
    except Exception as exc:  # noqa: BLE001
        log.warning("plus renderer cache invalidate failed: %s", exc)

    return web.json_response({
        "ok": True,
        "remaining": [s.person_id for s in sess.persons],
    })


# ---------------------------------------------------------------------------
# Per-person FBX / BVH export (Pose Editor +)
# ---------------------------------------------------------------------------

async def _api_plus_export(request: web.Request, *, fmt: str) -> web.Response:
    """Shared handler for /plus/export_fbx and /plus/export_bvh. ``fmt`` is
    either ``"fbx"`` or ``"bvh"``."""
    try:
        payload: dict[str, Any] = await request.json()
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"error": f"invalid json body: {exc}"}, status=400)
    plus_job_id = str(payload.get("plus_job_id") or "")
    person_id = str(payload.get("person_id") or "")
    blender_exe = str(payload.get("blender_exe") or "")
    output_filename = str(payload.get("output_filename") or "")
    if not plus_job_id or not person_id:
        return web.json_response(
            {"error": "plus_job_id and person_id are required"}, status=400,
        )
    if not _clean_blender_exe_path(blender_exe):
        return web.json_response(
            {"error": "blender_exe is required (set it on the Pose Editor + node)"},
            status=400,
        )

    runner = (
        plus_export.export_person_fbx if fmt == "fbx"
        else plus_export.export_person_bvh
    )
    try:
        output_path = await asyncio.to_thread(
            runner, plus_job_id, person_id, blender_exe, output_filename,
        )
    except LookupError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except (ValueError, RuntimeError) as exc:
        log.exception("plus %s export failed", fmt)
        return web.json_response({
            "error": str(exc),
            "trace": traceback.format_exc(limit=8),
        }, status=500)
    except Exception as exc:  # noqa: BLE001
        log.exception("plus %s export crashed", fmt)
        return web.json_response({
            "error": str(exc),
            "trace": traceback.format_exc(limit=8),
        }, status=500)

    return web.json_response({
        "ok": True,
        "person_id": person_id,
        "format": fmt,
        "output_path": output_path,
        "output_filename": os.path.basename(output_path),
    })


@routes.post("/sam3d/api/plus/export_fbx")
async def api_plus_export_fbx(request: web.Request) -> web.Response:
    return await _api_plus_export(request, fmt="fbx")


@routes.post("/sam3d/api/plus/export_bvh")
async def api_plus_export_bvh(request: web.Request) -> web.Response:
    return await _api_plus_export(request, fmt="bvh")


# ---------------------------------------------------------------------------
# Blender path (read-only — writes happen on node execute)
# ---------------------------------------------------------------------------

@routes.get("/sam3d/api/blender_path")
async def api_blender_path(_request: web.Request) -> web.Response:
    """Return the currently-saved Blender executable path. Used by the
    Pose Editor + frontend so the export buttons can show the user which
    Blender install will run when they click."""
    return web.json_response({"blender_exe": _get_blender_exe_path()})


# ---------------------------------------------------------------------------
# Display unit (cm / inch)
# ---------------------------------------------------------------------------

@routes.get("/sam3d/api/units")
async def api_units_get(_request: web.Request) -> web.Response:
    cfg = read_unit_config()
    return web.json_response({
        "display_unit": cfg.display_unit,
        "default_adult_height_m": cfg.default_adult_height_m,
        "valid_units": list(_VALID_DISPLAY_UNITS),
    })


@routes.post("/sam3d/api/units")
async def api_units_set(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"error": f"invalid json body: {exc}"}, status=400)
    unit = str((payload or {}).get("display_unit") or "").strip().lower()
    if unit not in _VALID_DISPLAY_UNITS:
        return web.json_response(
            {"error": f"display_unit must be one of {list(_VALID_DISPLAY_UNITS)}"},
            status=400,
        )
    _set_display_unit(unit)
    cfg = read_unit_config()
    return web.json_response({
        "display_unit": cfg.display_unit,
        "default_adult_height_m": cfg.default_adult_height_m,
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
        "body_preset_settings_dir": str(pack.body_preset_settings_dir),
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
    "  POST /sam3d/api/plus/process",
    "  POST /sam3d/api/plus/render",
    "  POST /sam3d/api/plus/drop",
    "  POST /sam3d/api/plus/reconcile_persons",
    "  POST /sam3d/api/plus/export_fbx",
    "  POST /sam3d/api/plus/export_bvh",
    "  POST /sam3d/api/sam2_preview",
    "  POST /sam3d/api/object_crop",
    "  GET  /sam3d/api/units",
    "  POST /sam3d/api/units",
    "  GET  /sam3d/api/blender_path",
    "  GET  /sam3d/api/active_pack",
):
    print(f"[SAM3DBody]{line}")

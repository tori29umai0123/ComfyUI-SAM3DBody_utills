# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""Multi-person renderer.

Builds a single merged OBJ from a cached ``PlusPoseSession`` plus a
per-person settings dict, mirroring the deform stack of the single
renderer (lean correction, bone-length scaling, blendshapes, per-bone
rotation overrides, body-shape sliders) for each person individually.

Geometry math:
    For each person ``i`` we run the shared per-session deform stack
    (``renderer.compute_session_mesh``), then re-scale the result by
    ``s_i = height_m_i / measured_height_m_i`` and translate by
    ``pred_cam_t_i * s_i``. Multiplying both verts and cam_t by the same
    ``s_i`` keeps the 2D image projection invariant — image-space bbox
    sizes and positions remain consistent with the input photo.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import plus_session
from . import paths as _paths
from .obj_export import write_obj
from .renderer import compute_session_mesh, normalise_settings
from .sam3dbody_loader import load_bundle

log = logging.getLogger(__name__)


_RENDER_CACHE_MAX = 32
# value: (all_verts, all_faces, vranges, franges, per_person_skeletons,
#         per_person_hip_world)
_render_cache: "OrderedDict[str, tuple]" = OrderedDict()


def _cache_key(plus_job_id: str, settings_norm: dict[str, Any]) -> str:
    return json.dumps(
        {"j": plus_job_id, "s": settings_norm}, sort_keys=True, separators=(",", ":")
    )


def _cache_get(key: str):
    if key not in _render_cache:
        return None
    _render_cache.move_to_end(key)
    return _render_cache[key]


def _cache_put(key: str, value) -> None:
    _render_cache[key] = value
    _render_cache.move_to_end(key)
    while len(_render_cache) > _RENDER_CACHE_MAX:
        _render_cache.popitem(last=False)


def invalidate_cache() -> None:
    _render_cache.clear()


@dataclass
class PlusRenderResult:
    plus_job_id: str
    obj_url: str
    obj_path: str
    elapsed_sec: float
    per_person_vertex_ranges: list[tuple[int, int]] = field(default_factory=list)
    per_person_face_ranges: list[tuple[int, int]] = field(default_factory=list)
    per_person_height_m: list[float] = field(default_factory=list)
    per_person_skeletons: list[dict[str, Any] | None] = field(default_factory=list)
    per_person_settings: list[dict[str, Any]] = field(default_factory=list)
    # World-space hip position per person — the (cam_t × height_scale)
    # offset baked into the OBJ. Frontend uses this as the pivot anchor
    # for the gizmo so user-applied rotations rotate around the body's
    # hip rather than the scene origin.
    per_person_hip_world: list[list[float]] = field(default_factory=list)


def _normalise_multi_settings(
    sess: plus_session.PlusPoseSession,
    raw: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve per-person settings to a stable, JSON-serialisable dict.

    Each person's full slider state passes through ``renderer.normalise_settings``
    so the cache key is consistent with the single renderer's view of the
    inputs. Height defaults to the session's stored target if not overridden.
    """
    out: dict[str, Any] = {"persons": {}}
    raw_persons: dict[str, Any] = {}
    if raw and isinstance(raw, dict):
        raw_persons = raw.get("per_person_settings") or raw.get("persons") or {}
        if not isinstance(raw_persons, dict):
            raw_persons = {}

    for slot in sess.persons:
        pid = slot.person_id
        person_raw = raw_persons.get(pid) or {}

        # Height — separate from the slider settings dict.
        try:
            override_h = float(person_raw.get("height_m"))
        except (TypeError, ValueError):
            override_h = None
        if override_h is None or not (0.3 <= override_h <= 3.0):
            override_h = float(slot.target_height_m)

        # Slider settings — accepts the same payload shape the single
        # editor sends (body_params / bone_lengths / blendshapes /
        # pose_adjust). Defaults filled in by ``normalise_settings``.
        slider_raw = {
            "body_params":  person_raw.get("body_params"),
            "bone_lengths": person_raw.get("bone_lengths"),
            "blendshapes":  person_raw.get("blendshapes"),
            "pose_adjust":  person_raw.get("pose_adjust"),
        }
        normalised = normalise_settings(slider_raw)

        out["persons"][pid] = {
            "height_m": round(override_h, 4),
            "settings": normalised,
        }
    return out


def _world_offset_from_cam_t(cam_t: np.ndarray | None, scale: float) -> np.ndarray:
    """Convert a SAM 3D Body ``pred_cam_t`` value into a Y-up world offset
    consistent with MHR vertex output. SAM 3D Body emits OpenCV-style
    translations (Y down, Z forward into scene); the editor's world is
    OpenGL-style (Y up, Z toward viewer), so we flip Y and Z."""
    if cam_t is None:
        return np.zeros(3, dtype=np.float32)
    o = np.asarray(cam_t, dtype=np.float32).reshape(3) * float(scale)
    o[1] = -o[1]
    o[2] = -o[2]
    return o


def render_plus_from_session(
    plus_job_id: str,
    settings: dict[str, Any] | None,
) -> PlusRenderResult:
    """Render every person in ``plus_job_id`` into a single merged OBJ."""
    t0 = time.monotonic()
    sess = plus_session.get(plus_job_id)
    if sess is None:
        raise KeyError(f"no multi-pose session for plus_job_id {plus_job_id!r}")

    norm = _normalise_multi_settings(sess, settings)
    bundle = load_bundle()
    device = torch.device(bundle.device)
    mhr_head = bundle.model.head_pose
    faces_local = mhr_head.faces.detach().cpu().numpy().astype(np.int64)

    # Persist per-person settings on the slot for downstream consumers.
    for slot in sess.persons:
        person_norm = norm["persons"][slot.person_id]
        slot.settings = person_norm["settings"]
        slot.target_height_m = float(person_norm["height_m"])

    cache_key = _cache_key(plus_job_id, norm)
    cached = _cache_get(cache_key)
    if cached is not None:
        all_verts, all_faces, vranges, franges, skeletons, hip_worlds = cached
        render_id = uuid.uuid4().hex[:8]
        obj_path = _paths.tmp_dir() / "plus_mesh.obj"
        write_obj(
            obj_path, all_verts, all_faces,
            header=f"# sam3dbody multi_job {plus_job_id} render {render_id} (cached)",
        )
        obj_url = _paths.tmp_url("plus_mesh.obj", version=render_id)
        elapsed = time.monotonic() - t0
        log.info(
            "plus-render job=%s render=%s CACHED elapsed=%.3fs",
            plus_job_id, render_id, elapsed,
        )
        return PlusRenderResult(
            plus_job_id=plus_job_id,
            obj_url=obj_url,
            obj_path=str(obj_path),
            elapsed_sec=elapsed,
            per_person_vertex_ranges=vranges,
            per_person_face_ranges=franges,
            per_person_height_m=[norm["persons"][s.person_id]["height_m"] for s in sess.persons],
            per_person_skeletons=skeletons,
            per_person_settings=[norm["persons"][s.person_id]["settings"] for s in sess.persons],
            per_person_hip_world=hip_worlds,
        )

    verts_chunks: list[np.ndarray] = []
    faces_chunks: list[np.ndarray] = []
    vranges: list[tuple[int, int]] = []
    franges: list[tuple[int, int]] = []
    height_per_person: list[float] = []
    skeletons: list[dict[str, Any] | None] = []
    v_offset = 0
    f_offset = 0

    hip_worlds: list[list[float]] = []
    for slot in sess.persons:
        person_norm = norm["persons"][slot.person_id]
        height_m = float(person_norm["height_m"])
        slider_settings = person_norm["settings"]

        # 1. Run the shared deform stack (lean / bone-length / blendshapes
        # / shape sliders / rotation overrides). Returns world-frame Y-up
        # vertices centred at the body's hip.
        verts, skeleton = compute_session_mesh(
            slot.pose_session, slider_settings, bundle=bundle, device=device,
        )
        verts = verts.astype(np.float32)

        # 2. Real-world height correction. ``measured_height_m`` is the
        # neutral T-pose height the plus pipeline cached at inference time.
        s = float(height_m) / max(float(slot.measured_height_m), 1e-3)
        s = float(np.clip(s, 0.2, 5.0))
        slot.height_scale = s
        verts = verts * s

        # The skeleton was built before the height-scale was applied so its
        # world_position values are still in raw MHR units. Multiply by
        # ``s`` so frontend bone handles / skeleton lines land at the same
        # scale as the rendered mesh — otherwise they'd cluster near the
        # origin while the body sits where cam_t × s placed it.
        if skeleton and isinstance(skeleton.get("bones"), list):
            scaled_bones = []
            for b in skeleton["bones"]:
                wp = b.get("world_position")
                if isinstance(wp, list) and len(wp) == 3:
                    nb = dict(b)
                    nb["world_position"] = [
                        float(wp[0]) * s, float(wp[1]) * s, float(wp[2]) * s,
                    ]
                    scaled_bones.append(nb)
                else:
                    scaled_bones.append(b)
            skeleton = {**skeleton, "bones": scaled_bones}

        # 3. Place in scene world by translating with cam_t * s. Same scale
        # on both sides preserves the 2D projection. User-supplied whole-
        # body rotation / translation is NOT applied here — the frontend
        # owns that adjustment via per-person Three.js anchors so the
        # gizmo and the sidebar sliders feel instantaneous (no backend
        # round-trip per drag frame).
        offset = _world_offset_from_cam_t(slot.pose_session.orig_cam_t, s)
        verts = verts + offset
        hip_worlds.append([float(offset[0]), float(offset[1]), float(offset[2])])

        f_remapped = faces_local + v_offset
        v_start = v_offset
        v_end = v_offset + verts.shape[0]
        f_start = f_offset
        f_end = f_offset + f_remapped.shape[0]

        verts_chunks.append(verts)
        faces_chunks.append(f_remapped)
        vranges.append((v_start, v_end))
        franges.append((f_start, f_end))
        height_per_person.append(height_m)
        skeletons.append(skeleton)

        v_offset = v_end
        f_offset = f_end

    if not verts_chunks:
        raise RuntimeError("plus-renderer received an empty session")

    all_verts = np.concatenate(verts_chunks, axis=0)
    all_faces = np.concatenate(faces_chunks, axis=0)

    render_id = uuid.uuid4().hex[:8]
    obj_path = _paths.tmp_dir() / "plus_mesh.obj"
    person_summary = " | ".join(
        f"{s.person_id}={vr[1] - vr[0]}v"
        for s, vr in zip(sess.persons, vranges)
    )
    write_obj(
        obj_path, all_verts, all_faces,
        header=(
            f"# sam3dbody multi_job {plus_job_id} render {render_id} "
            f"persons={len(sess.persons)} ({person_summary})"
        ),
    )
    obj_url = _paths.tmp_url("plus_mesh.obj", version=render_id)

    _cache_put(cache_key, (all_verts, all_faces, vranges, franges, skeletons, hip_worlds))

    elapsed = time.monotonic() - t0
    log.info(
        "plus-render job=%s render=%s persons=%d verts=%d elapsed=%.2fs",
        plus_job_id, render_id, len(sess.persons), len(all_verts), elapsed,
    )
    return PlusRenderResult(
        plus_job_id=plus_job_id,
        obj_url=obj_url,
        obj_path=str(obj_path),
        elapsed_sec=elapsed,
        per_person_vertex_ranges=vranges,
        per_person_face_ranges=franges,
        per_person_height_m=height_per_person,
        per_person_skeletons=skeletons,
        per_person_settings=[norm["persons"][s.person_id]["settings"] for s in sess.persons],
        per_person_hip_world=hip_worlds,
    )


__all__ = [
    "PlusRenderResult",
    "invalidate_cache",
    "render_plus_from_session",
]

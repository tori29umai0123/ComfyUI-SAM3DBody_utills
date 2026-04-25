"""Body-shape renderer: takes a cached pose + slider values, produces a new
OBJ mesh. This is the fast path used by every slider drag — the SAM3DBody
neutral-body mesh is regenerated via ``mhr_forward`` using user-provided
``shape_params``, then bone-length + blend-shape deltas are layered on top.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import character_shape as cs
from . import paths as _paths
from . import pose_session
from .obj_export import write_obj
from .preset_pack import active_pack_paths
from .sam3dbody_loader import load_bundle

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# In-memory LRU cache of the computed vertex array keyed on
# (job_id, normalized settings). Slider drags routinely bounce between a
# handful of nearby values — skipping the ~100–200 ms MHR forward + bone /
# blendshape passes on cache hit makes the UI feel instantaneous.
#
# - Cache stores the final vertex numpy array (post bone-length / blend-shape).
# - Faces come straight from the MHR model and are re-read on every call
#   (cheap: one tensor → numpy conversion).
# - Pack admin changes (FBX rebuild, pack switch) can invalidate keys —
#   call `invalidate_cache()` from those endpoints.
# --------------------------------------------------------------------------

_RENDER_CACHE_MAX = 128
# Cache entries are (vertices, humanoid_skeleton_dict_or_None) — the skeleton
# is snapshotted post-lean / pre-overrides so slider drags get the same
# base frame without a fresh MHR forward each time.
_render_cache: "OrderedDict[str, tuple[np.ndarray, dict[str, Any] | None]]" = OrderedDict()


def _render_cache_key(job_id: str, settings_norm: dict[str, Any]) -> str:
    return json.dumps(
        {"j": job_id, "s": settings_norm}, sort_keys=True, separators=(",", ":")
    )


def _render_cache_get(key: str) -> tuple[np.ndarray, dict[str, Any] | None] | None:
    if key not in _render_cache:
        return None
    _render_cache.move_to_end(key)
    return _render_cache[key]


def _render_cache_put(
    key: str, entry: tuple[np.ndarray, dict[str, Any] | None],
) -> None:
    _render_cache[key] = entry
    _render_cache.move_to_end(key)
    while len(_render_cache) > _RENDER_CACHE_MAX:
        _render_cache.popitem(last=False)


def invalidate_cache() -> None:
    """Drop all cached meshes. Pack admin / FBX rebuild flows call this
    when the blendshape / bone-length schema changes."""
    _render_cache.clear()


@dataclass
class RenderResult:
    job_id: str
    obj_url: str
    obj_path: str
    elapsed_sec: float
    settings: dict[str, Any]
    humanoid_skeleton: dict[str, Any] | None = None


def _empty_settings() -> dict[str, Any]:
    base: dict[str, Any] = {
        "body_params": {k: 0.0 for k in cs.BODY_PARAM_KEYS},
        "bone_lengths": {k: 1.0 for k in cs.BONE_LENGTH_KEYS},
        "blendshapes": {},
        "pose_adjust": {
            k: float(cs.POSE_ADJUST_DEFAULTS.get(k, 0.0))
            for k in cs.POSE_ADJUST_KEYS
        },
    }
    base["pose_adjust"]["rotation_overrides"] = {}
    return base


def _normalise_rotation_overrides_payload(raw: Any) -> dict[str, list[float]]:
    """Normalise the frontend ``rotation_overrides`` payload into a plain
    ``{ "joint_id_str": [rx, ry, rz] }`` dict. Drops non-humanoid / malformed
    entries so downstream consumers don't have to re-validate."""
    if not raw:
        return {}
    items = raw.items() if isinstance(raw, dict) else raw
    out: dict[str, list[float]] = {}
    for entry in items:
        if isinstance(entry, tuple) and len(entry) == 2:
            jid, euler = entry
        else:
            continue
        try:
            j = int(jid)
        except (TypeError, ValueError):
            continue
        if j not in cs.HUMANOID_JOINT_IDS:
            continue
        try:
            rx = float(euler[0]); ry = float(euler[1]); rz = float(euler[2])
        except (TypeError, ValueError, IndexError):
            continue
        import math as _m
        if not (_m.isfinite(rx) and _m.isfinite(ry) and _m.isfinite(rz)):
            continue
        if abs(rx) < 1e-8 and abs(ry) < 1e-8 and abs(rz) < 1e-8:
            continue
        out[str(j)] = [rx, ry, rz]
    return out


def _normalise_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    """Fill missing keys with their neutral defaults."""
    base = _empty_settings()
    if not settings:
        return base
    bp = settings.get("body_params") or {}
    bl = settings.get("bone_lengths") or {}
    bs = settings.get("blendshapes") or {}
    pa = settings.get("pose_adjust") or {}
    for k in cs.BODY_PARAM_KEYS:
        base["body_params"][k] = float(bp.get(k, 0.0))
    for k in cs.BONE_LENGTH_KEYS:
        base["bone_lengths"][k] = float(bl.get(k, 1.0))
    base["blendshapes"] = {str(k): float(v) for k, v in bs.items()}
    for k in cs.POSE_ADJUST_KEYS:
        default_v = float(cs.POSE_ADJUST_DEFAULTS.get(k, 0.0))
        try:
            base["pose_adjust"][k] = float(pa.get(k, default_v))
        except (TypeError, ValueError):
            base["pose_adjust"][k] = default_v
    base["pose_adjust"]["rotation_overrides"] = _normalise_rotation_overrides_payload(
        pa.get("rotation_overrides")
    )
    return base


def _mat3_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix → quaternion ``(qx, qy, qz, qw)`` (three.js order)."""
    m00, m01, m02 = float(R[0, 0]), float(R[0, 1]), float(R[0, 2])
    m10, m11, m12 = float(R[1, 0]), float(R[1, 1]), float(R[1, 2])
    m20, m21, m22 = float(R[2, 0]), float(R[2, 1]), float(R[2, 2])
    trace = m00 + m11 + m22
    import math as _m
    if trace > 0.0:
        s = 0.5 / _m.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m21 - m12) * s
        qy = (m02 - m20) * s
        qz = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * _m.sqrt(max(1e-12, 1.0 + m00 - m11 - m22))
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * _m.sqrt(max(1e-12, 1.0 + m11 - m00 - m22))
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = 2.0 * _m.sqrt(max(1e-12, 1.0 + m22 - m00 - m11))
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


def _build_humanoid_skeleton(
    rots: np.ndarray | None,
    coords: np.ndarray | None,
) -> dict[str, Any] | None:
    """Package the humanoid bone subset (post-lean, pre-overrides) for the
    image-tab rotation editor — see ``cs.HUMANOID_BONES`` for the current
    set. Returns ``None`` when the forward pass didn't yield per-joint
    rotations/coords."""
    if rots is None or coords is None:
        return None
    num_joints = int(rots.shape[0])
    bones: list[dict[str, Any]] = []
    for name, joint_id, parent_name in cs.HUMANOID_BONES:
        if joint_id < 0 or joint_id >= num_joints:
            continue
        R = rots[joint_id]
        p = coords[joint_id]
        qx, qy, qz, qw = _mat3_to_quat(R)
        bones.append({
            "name": name,
            "joint_id": int(joint_id),
            "parent_name": parent_name,
            "world_position": [float(p[0]), float(p[1]), float(p[2])],
            "world_quaternion": [qx, qy, qz, qw],
        })
    return {"bones": bones}


def render_from_session(
    job_id: str,
    settings: dict[str, Any] | None,
) -> RenderResult:
    """Build a mesh for ``job_id`` with the given slider values and write an
    OBJ under ``output/``. Reuses the cached SAM3DBody model + pose session.

    The reserved ``MAKE_JOB_ID`` renders the MHR neutral body in T-pose and
    is used by the Character Make tab — we auto-create it on first request
    so it stays available even if LRU eviction ever drops it."""
    t0 = time.monotonic()
    if job_id == pose_session.MAKE_JOB_ID:
        sess = pose_session.ensure_make_session()
    else:
        sess = pose_session.get(job_id)
        if sess is None:
            raise KeyError(f"no pose session for job_id {job_id!r}")

    s = _normalise_settings(settings)
    bundle = load_bundle()
    model = bundle.model
    device = torch.device(bundle.device)
    mhr_head = model.head_pose

    # Short-circuit on cache hit: skip MHR forward + all post-processing,
    # just re-write tmp/mesh.obj from the memoized vertices. This is the
    # hot path for slider drags that jitter between a few close values.
    faces = mhr_head.faces.detach().cpu().numpy()
    cache_key = _render_cache_key(job_id, s)
    cached = _render_cache_get(cache_key)
    if cached is not None:
        cached_verts, cached_skeleton = cached
        render_id = uuid.uuid4().hex[:8]
        obj_path = _paths.tmp_dir() / "mesh.obj"
        write_obj(
            obj_path, cached_verts, faces,
            header=f"# sam3dbody job {job_id} render {render_id} (cached)",
        )
        obj_url = _paths.tmp_url("mesh.obj", version=render_id)
        elapsed = time.monotonic() - t0
        log.info("render job=%s render=%s CACHED elapsed=%.3fs", job_id, render_id, elapsed)
        return RenderResult(
            job_id=job_id,
            obj_url=obj_url,
            obj_path=str(obj_path),
            elapsed_sec=elapsed,
            settings=s,
            humanoid_skeleton=cached_skeleton,
        )

    # Pose adjust (lean correction) is applied AFTER mhr_forward by
    # rotating the spine→neck chain; nothing to do to global_rot here.
    if job_id == pose_session.MAKE_JOB_ID:
        lean_strength = 0.0
    else:
        lean_strength = float(s["pose_adjust"].get("lean_correction", 0.0))
    global_rot = cs.to_tensor_1xN(sess.global_rot, device, width=3)
    body_pose = cs.to_tensor_1xN(sess.body_pose_params, device, width=133)
    hand_pose = cs.to_tensor_1xN(sess.hand_pose_params, device, width=108)
    expr = torch.zeros((1, mhr_head.num_face_comps), dtype=torch.float32, device=device)
    global_trans = torch.zeros((1, 3), dtype=torch.float32, device=device)
    shape_params = cs.build_shape_params(s["body_params"], mhr_head.num_shape_comps, device)
    scale_params = torch.zeros((1, mhr_head.num_scale_comps), dtype=torch.float32, device=device)

    with torch.no_grad():
        out = mhr_head.mhr_forward(
            global_trans=global_trans,
            global_rot=global_rot,
            body_pose_params=body_pose,
            hand_pose_params=hand_pose,
            scale_params=scale_params,
            shape_params=shape_params,
            expr_params=expr,
            return_joint_rotations=True,
            return_joint_coords=True,
        )

    verts = out[0]
    joint_rots = None
    joint_coords = None
    for t in out[1:]:
        if t.ndim == 4 and t.shape[-1] == 3 and t.shape[-2] == 3:
            joint_rots = t
        elif t.ndim == 3 and t.shape[-1] == 3 and t.shape[-2] != 3:
            joint_coords = t

    vertices = verts.detach().cpu().numpy()
    if vertices.ndim == 3:
        vertices = vertices[0]
    rots_np = joint_rots.detach().cpu().numpy() if joint_rots is not None else None
    if rots_np is not None and rots_np.ndim == 4:
        rots_np = rots_np[0]
    coords_np = joint_coords.detach().cpu().numpy() if joint_coords is not None else None
    if coords_np is not None and coords_np.ndim == 3:
        coords_np = coords_np[0]
    # `faces` already loaded above for cache short-circuit.

    # Cache MHR rest data (idempotent per mhr_head).
    cs.ensure_mhr_rest_cache(mhr_head, device)

    if coords_np is not None:
        vertices = cs.normalize_bone_lengths(vertices, coords_np)

    pack = active_pack_paths()
    bs_sliders = s["blendshapes"]
    if any(v != 0.0 for v in bs_sliders.values()) and rots_np is not None:
        rest_verts = cs.ensure_mhr_rest_cache(mhr_head, device)
        vertices = cs.apply_face_blendshapes(
            vertices, rest_verts, bs_sliders, rots_np,
            presets_dir=str(pack.pack_dir),
            npz_path=str(pack.npz_path),
        )

    bl = s["bone_lengths"]
    if rots_np is not None and any(abs(bl[k] - 1.0) > 1e-9 for k in cs.BONE_LENGTH_KEYS):
        vertices = cs.apply_bone_length_scales(
            vertices,
            arm_scale=bl["arm"], leg_scale=bl["leg"],
            torso_scale=bl["torso"], neck_scale=bl["neck"],
            joint_rots_posed=rots_np,
        )
        # Mirror the per-chain scaling onto the joint positions so the
        # humanoid skeleton overlay sits on the deformed mesh's joints (and
        # so the lean / rotation overrides below pivot at the right place).
        if coords_np is not None:
            coords_np = cs.scale_joint_coords_by_bone_length(
                coords_np,
                arm_scale=bl["arm"], leg_scale=bl["leg"],
                torso_scale=bl["torso"], neck_scale=bl["neck"],
            )

    # Lean correction — straighten a forward-leaning upper body by
    # rotating the spine→neck chain backward (applied last so it composes
    # on top of bone-length / blendshape adjustments).
    parents_np = cs._FACE_BS_CACHE.get("joint_parents")
    if coords_np is not None and lean_strength > 1e-6:
        vertices = cs.apply_pose_lean_correction_mesh(
            vertices, coords_np, lean_strength,
        )
        # Carry the lean correction forward into the per-joint state so the
        # humanoid skeleton returned to the frontend (and any subsequent
        # rotation overrides applied below) sees the post-lean pose.
        if rots_np is not None and parents_np is not None:
            rots_np, coords_np = cs.apply_pose_lean_correction_rig(
                rots_np, coords_np, parents_np, lean_strength,
            )

    # Snapshot humanoid skeleton AFTER lean but BEFORE rotation overrides —
    # this is the "base" frame the frontend rotation editor uses to
    # interpret stored local-frame Euler overrides.
    humanoid_skeleton = _build_humanoid_skeleton(rots_np, coords_np)

    # Per-bone rotation overrides (image-tab rotation editor). Applied last
    # so lean / bone-length / blendshape layers are fully baked first.
    rotation_overrides = s["pose_adjust"].get("rotation_overrides") or {}
    if (coords_np is not None and rots_np is not None
            and parents_np is not None and rotation_overrides):
        vertices = cs.apply_pose_rotation_overrides_mesh(
            vertices, rots_np, coords_np, parents_np, rotation_overrides,
        )

    # mhr_forward already produces vertices in an OpenGL-convention world
    # frame (Y up, Z toward the viewer); dropping them straight into Three.js
    # is correct.
    # Single overwriting file in tmp/; the `?v=...` query string below
    # defeats the browser/Three.js cache so each slider drag pulls the
    # fresh mesh rather than the stale one.
    render_id = uuid.uuid4().hex[:8]
    obj_path = _paths.tmp_dir() / "mesh.obj"
    write_obj(obj_path, vertices, faces, header=f"# sam3dbody job {job_id} render {render_id}")
    obj_url = _paths.tmp_url("mesh.obj", version=render_id)

    _render_cache_put(cache_key, (vertices, humanoid_skeleton))

    elapsed = time.monotonic() - t0
    log.info("render job=%s render=%s verts=%d elapsed=%.2fs", job_id, render_id, len(vertices), elapsed)
    return RenderResult(
        job_id=job_id,
        obj_url=obj_url,
        obj_path=str(obj_path),
        elapsed_sec=elapsed,
        settings=s,
        humanoid_skeleton=humanoid_skeleton,
    )

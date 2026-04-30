# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""Per-person FBX / BVH export for the Pose Editor +.

The single Pose / FBX / BVH export nodes in ``nodes/processing/`` parse a
``body_preset_json`` + ``pose_json`` string pair, then run the same
SAM3DBody → MHR forward stack we use for inference. This module reuses
those private helpers but feeds them the per-person pose params + body
preset settings already cached by the Pose Editor +'s plus session, so
the user can hit "FBX / BVH" on a person card and get a standalone file
without round-tripping through ComfyUI export nodes.

Public surface:
    export_person_fbx(plus_job_id, person_id, blender_exe, output_filename)
    export_person_bvh(plus_job_id, person_id, blender_exe, output_filename)

Both return the absolute path of the file Blender wrote.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import torch

from . import plus_session
from . import sam3dbody_loader

# Re-use the heavy SAM3DBody + MHR helpers from the existing processing
# pipeline. Importing process / export_rigged / export_bvh modules also
# pulls in their internal caches so subsequent invocations are cheap.
from ...processing import process as _process
from ...processing.process import (
    _FACE_BS_CACHE,
    _apply_face_blendshapes,
    _apply_bone_length_scales,
    _get_mhr_rest_verts,
    _to_batched_tensor,
)
from ...processing.export_rigged import (
    _BUILD_SCRIPT,
    _KNOWN_JOINT_NAMES,
    _SHAPE_SLIDER_NORM,
    _SHAPE_SLIDER_SIGN,
    _scale_skeleton_rest,
    _unpack_batched,
)
from ...processing.export_bvh import (
    _BUILD_RIGGED_BVH_SCRIPT,
    _run_blender_export,
    _subset_humanoid,
)
from ...preset_pack import clean_blender_exe_path, set_blender_exe_path

import folder_paths  # type: ignore[import-not-found]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model bridge — share the SAM3DBody bundle between the editor's loader and
# the processing pipeline's cache so we don't load weights twice.
# ---------------------------------------------------------------------------

def _ensure_model_in_process_cache() -> dict:
    """Make sure ``process._MODEL_CACHE`` knows about the bundle the editor
    already loaded; return a ``model_config``-shaped dict the existing
    helpers expect."""
    bundle = sam3dbody_loader.load_bundle()
    model_path = os.path.join(folder_paths.models_dir, "sam3dbody")
    ckpt_path = os.path.join(model_path, "model.ckpt")
    mhr_path = os.path.join(model_path, "assets", "mhr_model.pt")
    if ckpt_path not in _process._MODEL_CACHE:
        _process._MODEL_CACHE[ckpt_path] = {
            "model": bundle.model,
            "model_cfg": bundle.cfg,
            "device": bundle.device,
            "mhr_path": mhr_path,
        }
    return {
        "model_path": model_path,
        "ckpt_path": ckpt_path,
        "mhr_path": mhr_path,
        "device": bundle.device,
    }


# ---------------------------------------------------------------------------
# Per-person data extraction
# ---------------------------------------------------------------------------

def _extract_person_payload(
    plus_job_id: str, person_id: str,
) -> tuple[dict, dict]:
    """Pull a ``(preset, pose)`` pair out of the cached plus session for
    the given person. Returns exactly the structure the existing export
    helpers expect — ``preset`` has body_params/bone_lengths/blendshapes,
    ``pose`` has body_pose_params/hand_pose_params/global_rot."""
    sess = plus_session.get(plus_job_id)
    if sess is None:
        raise LookupError(f"plus session {plus_job_id!r} not found")
    slot = next((s for s in sess.persons if s.person_id == person_id), None)
    if slot is None:
        raise LookupError(
            f"person {person_id!r} not present in session {plus_job_id!r}",
        )

    # Pull the live editor settings (body_params / bone_lengths / blendshapes)
    # straight from the slot. Pose-adjust + rotation overrides intentionally
    # stay out of the rigged FBX — the export path bakes the inference pose
    # only (lean correction is rendered as a mesh rotation, not a rig delta).
    raw = slot.settings or {}
    preset = {
        "body_params":  raw.get("body_params", {}) or {},
        "bone_lengths": raw.get("bone_lengths", {}) or {},
        "blendshapes":  raw.get("blendshapes", {}) or {},
    }

    pose = slot.pose_session
    pose_payload = {
        "body_pose_params": np.asarray(pose.body_pose_params).tolist(),
        "hand_pose_params": np.asarray(pose.hand_pose_params).tolist(),
        "global_rot":       np.asarray(pose.global_rot).tolist(),
    }
    return preset, pose_payload


# ---------------------------------------------------------------------------
# Output file naming
# ---------------------------------------------------------------------------

def _resolve_output_path(output_filename: str, default_stem: str, ext: str) -> str:
    """Mirror the auto-naming behaviour of the existing export nodes. ``ext``
    is included with the leading dot, e.g. ``".fbx"``."""
    name = (output_filename or "").strip()
    if not name:
        name = f"{default_stem}_{int(time.time())}{ext}"
    elif not name.lower().endswith(ext):
        name = name + ext
    output_dir = folder_paths.get_output_directory()
    return os.path.abspath(os.path.join(output_dir, name))


# ---------------------------------------------------------------------------
# Shared SAM3DBody → MHR forward step (rest + posed)
# ---------------------------------------------------------------------------

def _build_rest_and_pose(preset: dict, pose: dict):
    """Run the MHR rest / posed forwards for a single person. Returns a
    bag of numpy arrays the per-format packagers below consume."""
    model_config = _ensure_model_in_process_cache()
    loaded = _process._load_sam3d_model(model_config)
    sam_3d_model = loaded["model"]
    device = torch.device(loaded["device"])
    mhr_head = sam_3d_model.head_pose

    # Prime caches (rest_verts, lbs_weights, joint_parents, joint_chain_cats…).
    _get_mhr_rest_verts(mhr_head, device)
    parents = _FACE_BS_CACHE["joint_parents"].astype(np.int32)
    lbs_weights = _FACE_BS_CACHE["lbs_weights"].astype(np.float32)
    num_joints = parents.shape[0]
    faces = mhr_head.faces.detach().cpu().numpy().astype(np.int32)

    bp = preset.get("body_params", {}) or {}
    bl = preset.get("bone_lengths", {}) or {}
    bs = preset.get("blendshapes", {}) or {}

    body_shape_ui = [
        float(bp.get("fat", 0.0)),
        float(bp.get("muscle", 0.0)),
        float(bp.get("fat_muscle", 0.0)),
        float(bp.get("limb_girth", 0.0)),
        float(bp.get("limb_muscle", 0.0)),
        float(bp.get("limb_fat", 0.0)),
        float(bp.get("chest_shoulder", 0.0)),
        float(bp.get("waist_hip", 0.0)),
        float(bp.get("thigh_calf", 0.0)),
    ]
    shape_params = torch.zeros(
        (1, mhr_head.num_shape_comps), dtype=torch.float32, device=device,
    )
    for i in range(min(9, mhr_head.num_shape_comps)):
        shape_params[0, i] = (
            body_shape_ui[i] * _SHAPE_SLIDER_NORM[i] * _SHAPE_SLIDER_SIGN[i]
        )
    scale_params = torch.zeros(
        (1, mhr_head.num_scale_comps), dtype=torch.float32, device=device,
    )
    expr_params = torch.zeros(
        (1, mhr_head.num_face_comps), dtype=torch.float32, device=device,
    )

    # ---- REST pose (body_pose = 0) — used as the FBX/BVH bind pose ----
    zeros3 = torch.zeros((1, 3), dtype=torch.float32, device=device)
    body_zero = torch.zeros((1, 133), dtype=torch.float32, device=device)
    hand_zero = torch.zeros((1, 108), dtype=torch.float32, device=device)
    global_trans = torch.zeros((1, 3), dtype=torch.float32, device=device)
    with torch.no_grad():
        rest_out = mhr_head.mhr_forward(
            global_trans=global_trans, global_rot=zeros3,
            body_pose_params=body_zero, hand_pose_params=hand_zero,
            scale_params=scale_params, shape_params=shape_params,
            expr_params=expr_params,
            return_joint_rotations=True, return_joint_coords=True,
        )
    rest_verts = rest_out[0].detach().cpu().numpy().astype(np.float32)
    if rest_verts.ndim == 3:
        rest_verts = rest_verts[0]
    rest_rots, rest_coords = _unpack_batched(rest_out[1:])
    rest_rots = rest_rots.astype(np.float32)
    rest_coords = rest_coords.astype(np.float32)

    # Apply blend shapes on top of the rest mesh (identity rotation here).
    bs_sliders = {str(k): float(v) for k, v in bs.items()}
    if any(v != 0.0 for v in bs_sliders.values()):
        from .preset_pack import active_pack_paths
        pack = active_pack_paths()
        rest_verts = _apply_face_blendshapes(
            rest_verts, _FACE_BS_CACHE["rest_verts"], bs_sliders,
            rest_rots, str(pack.pack_dir), str(pack.npz_path),
        )

    bone_scales = {
        "torso": float(bl.get("torso", 1.0)),
        "neck":  float(bl.get("neck", 1.0)),
        "arm":   float(bl.get("arm", 1.0)),
        "leg":   float(bl.get("leg", 1.0)),
    }
    any_bone_scaled = any(v != 1.0 for v in bone_scales.values())
    if any_bone_scaled:
        rest_verts = _apply_bone_length_scales(
            rest_verts,
            arm_scale=bone_scales["arm"], leg_scale=bone_scales["leg"],
            torso_scale=bone_scales["torso"], neck_scale=bone_scales["neck"],
            joint_rots_posed=rest_rots,
        )
        rest_coords = _scale_skeleton_rest(
            rest_coords, parents,
            _FACE_BS_CACHE["joint_chain_cats"], bone_scales,
        )

    # ---- POSED pose (body_pose from session) ----
    global_rot_t = _to_batched_tensor(pose.get("global_rot"), device, width=3)
    body_pose_t  = _to_batched_tensor(pose.get("body_pose_params"), device, width=133)
    hand_pose_t  = _to_batched_tensor(pose.get("hand_pose_params"), device, width=108)
    with torch.no_grad():
        posed_out = mhr_head.mhr_forward(
            global_trans=global_trans, global_rot=global_rot_t,
            body_pose_params=body_pose_t, hand_pose_params=hand_pose_t,
            scale_params=scale_params, shape_params=shape_params,
            expr_params=expr_params,
            return_joint_rotations=True, return_joint_coords=True,
        )
    posed_rots, posed_coords = _unpack_batched(posed_out[1:])
    posed_rots = posed_rots.astype(np.float32)
    posed_coords = posed_coords.astype(np.float32)
    if any_bone_scaled:
        posed_coords = _scale_skeleton_rest(
            posed_coords, parents,
            _FACE_BS_CACHE["joint_chain_cats"], bone_scales,
        )

    names_full = [_KNOWN_JOINT_NAMES.get(i, f"joint_{i:03d}") for i in range(num_joints)]
    return {
        "rest_verts": rest_verts,
        "rest_rots": rest_rots,
        "rest_coords": rest_coords,
        "posed_rots": posed_rots,
        "posed_coords": posed_coords,
        "parents": parents,
        "names_full": names_full,
        "lbs_weights": lbs_weights,
        "faces": faces,
        "num_joints": num_joints,
    }


# ---------------------------------------------------------------------------
# Public: per-format export
# ---------------------------------------------------------------------------

def export_person_fbx(
    plus_job_id: str,
    person_id: str,
    blender_exe: str,
    output_filename: str,
) -> str:
    """Run the rigged-FBX export pipeline for one person and return the
    absolute output path."""
    cleaned_exe = clean_blender_exe_path(blender_exe)
    if not cleaned_exe:
        raise ValueError("blender_exe is empty")
    try:
        set_blender_exe_path(cleaned_exe)
    except Exception as exc:  # noqa: BLE001
        log.warning("config.ini blender path save failed: %s", exc)

    preset, pose_payload = _extract_person_payload(plus_job_id, person_id)
    state = _build_rest_and_pose(preset, pose_payload)

    parents = state["parents"]
    lbs_weights = state["lbs_weights"]
    num_joints = state["num_joints"]

    # Same weightless-leaf prune as export_rigged.py — keeps the FBX free
    # of disconnected stub bones.
    keep = lbs_weights.sum(axis=0) > 1e-6
    for j in range(num_joints - 1, -1, -1):
        if keep[j]:
            p = int(parents[j])
            if p >= 0:
                keep[p] = True
    kept_idx = np.where(keep)[0]
    j_remap = np.full(num_joints, -1, dtype=np.int32)
    for new, old in enumerate(kept_idx):
        j_remap[int(old)] = new
    new_parents = [
        int(j_remap[int(parents[o])]) if int(parents[o]) >= 0 else -1
        for o in kept_idx
    ]
    names = [state["names_full"][o] for o in kept_idx]
    rest_coords_out  = state["rest_coords"][kept_idx]
    rest_rots_out    = state["rest_rots"][kept_idx]
    posed_coords_out = state["posed_coords"][kept_idx]
    posed_rots_out   = state["posed_rots"][kept_idx]

    nonzero = lbs_weights > 1e-5
    v_idx, j_idx = np.where(nonzero)
    w_val = lbs_weights[nonzero].astype(np.float32)
    j_idx_new = j_remap[j_idx]
    valid = j_idx_new >= 0
    v_idx = v_idx[valid]
    j_idx_new = j_idx_new[valid]
    w_val = w_val[valid]

    output_path = _resolve_output_path(
        output_filename,
        default_stem=f"sam3d_pose_plus_{person_id}",
        ext=".fbx",
    )

    package = {
        "output_path": output_path,
        "rest_verts":         state["rest_verts"].tolist(),
        "faces":              state["faces"].tolist(),
        "joint_parents":      new_parents,
        "joint_names":        names,
        "rest_joint_coords":  rest_coords_out.tolist(),
        "rest_joint_rots":    rest_rots_out.tolist(),
        "posed_joint_coords": posed_coords_out.tolist(),
        "posed_joint_rots":   posed_rots_out.tolist(),
        "lbs_v_idx":  v_idx.astype(np.int32).tolist(),
        "lbs_j_idx":  j_idx_new.astype(np.int32).tolist(),
        "lbs_weight": w_val.tolist(),
    }

    log.info("plus FBX export: %s persons=%s -> %s",
             plus_job_id, person_id, output_path)
    _run_blender_export(package, _BUILD_SCRIPT, cleaned_exe, timeout_sec=600)
    if not os.path.exists(output_path):
        raise RuntimeError(
            f"Blender reported success but {output_path} was not created."
        )
    return output_path


def export_person_bvh(
    plus_job_id: str,
    person_id: str,
    blender_exe: str,
    output_filename: str,
) -> str:
    """Run the rigged-BVH export pipeline for one person and return the
    absolute output path."""
    cleaned_exe = clean_blender_exe_path(blender_exe)
    if not cleaned_exe:
        raise ValueError("blender_exe is empty")
    try:
        set_blender_exe_path(cleaned_exe)
    except Exception as exc:  # noqa: BLE001
        log.warning("config.ini blender path save failed: %s", exc)

    preset, pose_payload = _extract_person_payload(plus_job_id, person_id)
    state = _build_rest_and_pose(preset, pose_payload)

    output_path = _resolve_output_path(
        output_filename,
        default_stem=f"sam3d_pose_plus_{person_id}",
        ext=".bvh",
    )

    package = _subset_humanoid({
        "output_path": output_path,
        "joint_parents":      state["parents"].tolist(),
        "joint_names":        state["names_full"],
        "rest_joint_coords":  state["rest_coords"].tolist(),
        "rest_joint_rots":    state["rest_rots"].tolist(),
        "posed_joint_coords": state["posed_coords"].tolist(),
        "posed_joint_rots":   state["posed_rots"].tolist(),
    })
    package["output_path"] = output_path

    log.info("plus BVH export: %s persons=%s -> %s",
             plus_job_id, person_id, output_path)
    _run_blender_export(package, _BUILD_RIGGED_BVH_SCRIPT, cleaned_exe, timeout_sec=600)
    if not os.path.exists(output_path):
        raise RuntimeError(
            f"Blender reported success but {output_path} was not created."
        )
    return output_path

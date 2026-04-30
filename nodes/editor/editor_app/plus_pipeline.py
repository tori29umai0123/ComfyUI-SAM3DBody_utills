# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""Multi-person image -> per-instance segmentation -> pose estimation pipeline.

Pipeline:
    1. Receive a single image plus N user-drawn boxes (each with a target
       height and optional left/right hand crops).
    2. Hand the boxes to SAM 2 to get one mask per person.
    3. Hand both arrays to SAM 3D Body's ``process_one_image`` — that path
       already supports N >= 2 because the underlying ``prepare_batch``
       loops over rows.
    4. For each person, splice in any hand-decoder override and compute a
       height correction factor ``s = target_height_m / mesh_height_m``.
       Apply ``s`` to vertices and ``pred_cam_t`` at render time so the 2D
       projection is preserved (image-space bbox stays the same).
    5. Persist the lot as a ``PlusPoseSession`` and run the initial render.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from . import plus_session
from . import paths as _paths
from . import sam2_segmenter
from .hand_inference import (
    hand_rgb_to_uint8,
    run_hand_only_inference,
    splice_hand_into_params,
)
from .pipeline import _normalize_input_image
from .pose_session import PoseSession
from .sam3dbody_loader import load_bundle

log = logging.getLogger(__name__)


@dataclass
class PersonInput:
    person_id: str
    bbox_xyxy: tuple[float, float, float, float]
    height_m: float
    lhand_image: Image.Image | None = None
    rhand_image: Image.Image | None = None
    # Optional list of additional positive bboxes (Shift+left-drag in the
    # editor). Each is segmented independently by SAM 2 and OR-ed onto
    # the primary person mask so missed regions can be reclaimed.
    additional_bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    # Optional list of "exclude this region" rectangles (right-click drag
    # in the editor). Each is segmented independently by SAM 2 and the
    # union of those masks is subtracted from the person mask.
    negative_bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class PlusPipelineResult:
    plus_job_id: str
    obj_url: str
    obj_path: str
    width: int
    height: int
    elapsed_sec: float
    per_person: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mesh height bookkeeping
# ---------------------------------------------------------------------------

def _measure_neutral_tpose_height_m(bundle, device: torch.device) -> float:
    """Y-extent of the MHR neutral mesh (T-pose, zero shape / scale).

    The renderer always draws meshes with ``shape_params == 0`` and
    ``scale_params == 0`` — so this is the implicit "1.0x" height every
    person renders at before height correction. Cached on the bundle the
    first time it is measured.
    """
    cached = getattr(bundle, "_neutral_tpose_height_m", None)
    if cached is not None:
        return float(cached)

    mhr_head = bundle.model.head_pose
    nf = mhr_head.num_face_comps
    ns = mhr_head.num_shape_comps
    ksc = mhr_head.num_scale_comps

    z3 = torch.zeros((1, 3), dtype=torch.float32, device=device)
    zb = torch.zeros((1, 133), dtype=torch.float32, device=device)
    zh = torch.zeros((1, 108), dtype=torch.float32, device=device)
    zs = torch.zeros((1, ns), dtype=torch.float32, device=device)
    zsc = torch.zeros((1, ksc), dtype=torch.float32, device=device)
    zf = torch.zeros((1, nf), dtype=torch.float32, device=device)

    with torch.no_grad():
        out = mhr_head.mhr_forward(
            global_trans=z3, global_rot=z3,
            body_pose_params=zb, hand_pose_params=zh,
            scale_params=zsc, shape_params=zs, expr_params=zf,
        )
    verts = out[0].detach().cpu().numpy()
    if verts.ndim == 3:
        verts = verts[0]
    h = float(verts[:, 1].max() - verts[:, 1].min())
    bundle._neutral_tpose_height_m = h
    log.info("MHR neutral T-pose height = %.4f m", h)
    return h


def compute_height_scale(target_height_m: float, neutral_height_m: float) -> float:
    """Real-world-aware uniform scale factor.

    Returns ``target / neutral`` clamped to a sane range so a typo in the
    height field can't blow the scene up or collapse it to a dot. The
    renderer uses this on both vertices AND ``pred_cam_t`` so the 2D
    projection is invariant under it (foreshortening preserved).
    """
    if neutral_height_m <= 1e-3:
        return 1.0
    s = float(target_height_m) / float(neutral_height_m)
    return float(np.clip(s, 0.2, 5.0))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _save_combined_mask_png(masks: np.ndarray, path) -> None:
    """Compose all per-person masks into a single grey-coded PNG (one
    intensity per person) so the editor can show a debug overlay if needed.
    Persons numbered from 1; index 0 is background.
    """
    if masks.ndim != 3:
        raise ValueError(f"expected (N,H,W) masks, got {masks.shape}")
    h, w = masks.shape[1], masks.shape[2]
    out = np.zeros((h, w), dtype=np.uint8)
    for i, m in enumerate(masks):
        # Cap at 255 to avoid wrap-around if someone really stacks 256+ persons.
        out[m > 0] = min(255, (i + 1) * max(1, 255 // max(1, masks.shape[0])))
    Image.fromarray(out, mode="L").save(path)


def run_plus_image_to_obj(
    pil_image: Image.Image,
    persons: list[PersonInput],
    *,
    inference_type: str = "full",
    device_mode: str | None = None,
) -> PlusPipelineResult:
    """Run SAM 2 + SAM 3D Body on a single image with N user-drawn boxes."""
    if not persons:
        raise ValueError("run_plus_image_to_obj requires at least one PersonInput")

    t0 = time.monotonic()
    bundle = load_bundle(device_mode)
    device = torch.device(bundle.device)

    pil_image = _normalize_input_image(pil_image)
    rgb = np.asarray(pil_image, dtype=np.uint8)
    h, w = rgb.shape[:2]

    bboxes = np.stack(
        [np.asarray(p.bbox_xyxy, dtype=np.float32).reshape(4) for p in persons], axis=0
    )
    add_bboxes = [
        np.asarray(p.additional_bboxes, dtype=np.float32).reshape(-1, 4)
        if p.additional_bboxes else None
        for p in persons
    ]
    neg_bboxes = [
        np.asarray(p.negative_bboxes, dtype=np.float32).reshape(-1, 4)
        if p.negative_bboxes else None
        for p in persons
    ]

    # 1. Per-person masks via SAM 2: primary box + additional positive
    # boxes are unioned, then negative-bbox masks are subtracted.
    mask_result = sam2_segmenter.predict_masks(
        rgb, bboxes, neg_bboxes, add_bboxes,
    )
    masks = mask_result.masks  # (N, H, W) uint8
    masks_for_model = masks.reshape(-1, h, w, 1).astype(np.uint8)

    # 2. Run SAM 3D Body on all N people in one shot. The estimator keys
    # on its own internal cache and resets it on each ``process_one_image``,
    # so we hand it raw RGB and let it manage the rest.
    results = bundle.estimator.process_one_image(
        rgb,
        bboxes=bboxes,
        masks=masks_for_model,
        inference_type=inference_type,
    )
    if not results or len(results) != len(persons):
        raise RuntimeError(
            f"SAM 3D Body returned {len(results) if results else 0} detections "
            f"for {len(persons)} requested persons"
        )

    # 3. Per-person hand override + session build.
    neutral_h = _measure_neutral_tpose_height_m(bundle, device)

    plus_job_id = uuid.uuid4().hex[:12]
    slots: list[plus_session.PlusPersonSlot] = []
    pred_cam_ts: list[np.ndarray] = []
    focal_lengths: list[float] = []

    for idx, (req, raw) in enumerate(zip(persons, results)):
        hand_pose = np.asarray(raw.get("hand_pose_params"), dtype=np.float32)

        lhand_arr = hand_rgb_to_uint8(req.lhand_image)
        rhand_arr = hand_rgb_to_uint8(req.rhand_image)
        lhand_params = (
            run_hand_only_inference(bundle.estimator, lhand_arr, is_left=True)
            if lhand_arr is not None else None
        )
        rhand_params = (
            run_hand_only_inference(bundle.estimator, rhand_arr, is_left=False)
            if rhand_arr is not None else None
        )
        if lhand_params is not None or rhand_params is not None:
            hand_pose = splice_hand_into_params(
                hand_pose, lhand_params=lhand_params, rhand_params=rhand_params,
            )

        # 4. Height correction factor — applied at render time.
        target_h = float(req.height_m) if req.height_m and req.height_m > 0 else neutral_h
        s = compute_height_scale(target_h, neutral_h)

        cam_t = np.asarray(raw.get("pred_cam_t"), dtype=np.float32).reshape(-1)
        focal = float(np.asarray(raw.get("focal_length")).reshape(-1)[0])
        pred_cam_ts.append(cam_t)
        focal_lengths.append(focal)

        body_pose = np.asarray(raw.get("body_pose_params"), dtype=np.float32)
        global_rot = np.asarray(raw.get("global_rot"), dtype=np.float32)
        keypoints_3d = (
            np.asarray(raw["pred_keypoints_3d"], dtype=np.float32)
            if raw.get("pred_keypoints_3d") is not None else None
        )

        person_job_id = f"{plus_job_id}_p{idx}"
        sess = PoseSession(
            job_id=person_job_id,
            pose_json={},  # multi node returns no JSON in this revision
            global_rot=global_rot,
            body_pose_params=body_pose,
            hand_pose_params=hand_pose,
            image_width=w,
            image_height=h,
            orig_focal_length=focal,
            orig_cam_t=cam_t,
            orig_keypoints_3d=keypoints_3d,
            bbox_xyxy=bboxes[idx].astype(np.float32),
            height_scale=s,
        )

        slots.append(plus_session.PlusPersonSlot(
            person_id=req.person_id,
            bbox_xyxy=bboxes[idx].astype(np.float32),
            target_height_m=target_h,
            measured_height_m=neutral_h,
            height_scale=s,
            pose_session=sess,
            has_lhand_override=lhand_params is not None,
            has_rhand_override=rhand_params is not None,
            mask_score=float(mask_result.scores[idx])
                if idx < mask_result.scores.size else 0.0,
        ))

    # Shared focal — warn (but don't fail) if the per-person values disagree
    # by more than ~5%. SAM 3D Body emits per-person estimates that should be
    # essentially identical for a single image; large divergence usually
    # means a bbox is bad and the model fell back to a degenerate solution.
    if focal_lengths:
        f_arr = np.asarray(focal_lengths, dtype=np.float32)
        f_med = float(np.median(f_arr))
        f_max_dev = float(np.max(np.abs(f_arr - f_med)) / max(f_med, 1e-6))
        if f_max_dev > 0.05:
            log.warning(
                "per-person focal_length disagreement %.1f%% (med=%.2f, all=%s) — "
                "using median; check bbox accuracy",
                f_max_dev * 100.0, f_med, f_arr.tolist(),
            )
    else:
        f_med = 1.0

    sess_obj = plus_session.PlusPoseSession(
        plus_job_id=plus_job_id,
        persons=slots,
        image_width=w,
        image_height=h,
        focal_length=f_med,
        pred_cam_ts=pred_cam_ts,
    )
    plus_session.put(sess_obj)

    # 5. Initial render. plus_renderer is imported lazily so the import
    # graph stays acyclic and the SAM 2 / SAM 3D Body imports above don't
    # tug renderer-side dependencies into early-load failure paths.
    from .plus_renderer import render_plus_from_session

    render = render_plus_from_session(plus_job_id, None)

    # 6. Optional debug mask PNG.
    try:
        _save_combined_mask_png(masks, _paths.tmp_dir() / "plus_mask.png")
    except Exception as exc:  # noqa: BLE001
        log.warning("plus-mask PNG save failed: %s", exc)

    elapsed = time.monotonic() - t0
    log.info(
        "plus pipeline %s: image=%dx%d persons=%d elapsed=%.2fs",
        plus_job_id, w, h, len(persons), elapsed,
    )

    per_person_payload: list[dict[str, Any]] = []
    for slot, vrange, frange, skel, settings, hip_world in zip(
        slots,
        render.per_person_vertex_ranges,
        render.per_person_face_ranges,
        render.per_person_skeletons,
        render.per_person_settings,
        render.per_person_hip_world,
    ):
        per_person_payload.append({
            "id": slot.person_id,
            "bbox_xyxy": slot.bbox_xyxy.astype(float).tolist(),
            "target_height_m": float(slot.target_height_m),
            "measured_height_m": float(slot.measured_height_m),
            "height_scale": float(slot.height_scale),
            "has_lhand_override": bool(slot.has_lhand_override),
            "has_rhand_override": bool(slot.has_rhand_override),
            "mask_score": float(slot.mask_score),
            "vertex_range": list(vrange),
            "face_range": list(frange),
            "humanoid_skeleton": skel,
            "settings": settings,
            "hip_world": hip_world,
        })

    return PlusPipelineResult(
        plus_job_id=plus_job_id,
        obj_url=render.obj_url,
        obj_path=render.obj_path,
        width=w,
        height=h,
        elapsed_sec=elapsed,
        per_person=per_person_payload,
    )


__all__ = [
    "PlusPipelineResult",
    "PersonInput",
    "compute_height_scale",
    "run_plus_image_to_obj",
]

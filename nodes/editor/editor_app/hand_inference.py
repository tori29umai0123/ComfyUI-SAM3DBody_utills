# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""Hand-only inference helper.

The full SAM3D Body inference path runs both the body and hand decoders.
This module exposes a thin wrapper around the hand decoder that takes a
single cropped hand image and returns a 54-dim hand pose params vector,
which the caller can splice into a pose JSON to override the body's
hand prediction.

Used by both the editor pipeline (when the user uploads a hand image
beside the main image) and the SAM3DBodyProcessToJson node (when the
``Left_hand_image`` / ``Right_hand_image`` optional inputs are wired).
"""
from __future__ import annotations

import logging

import numpy as np
import torch

log = logging.getLogger(__name__)


def hand_rgb_to_uint8(image_like) -> np.ndarray | None:
    """Convert various image inputs (PIL, numpy, torch tensor) into a
    contiguous H×W×3 uint8 RGB array. Returns ``None`` for empty / 1×1
    placeholders and for inputs the caller didn't actually wire up.
    """
    if image_like is None:
        return None
    arr = None
    try:
        if isinstance(image_like, torch.Tensor):
            t = image_like
            if t.dim() == 4:
                t = t[0]
            arr = t.detach().cpu().numpy()
        elif isinstance(image_like, np.ndarray):
            arr = image_like[0] if image_like.ndim == 4 else image_like
        else:
            from PIL import Image as _PILImage
            if isinstance(image_like, _PILImage.Image):
                arr = np.asarray(image_like.convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        log.warning("hand image normalize failed: %s", exc)
        return None
    if arr is None:
        return None
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        return None
    if arr.shape[0] < 4 or arr.shape[1] < 4:
        return None
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def run_hand_only_inference(estimator, hand_rgb_uint8: np.ndarray, *, is_left: bool) -> np.ndarray:
    """Run the hand decoder on a cropped hand image.

    Mirrors what the full pipeline (``sam3d_body.py:1238-1272``) does for
    a left hand: flip the image first so the model sees a "right" hand,
    then read the [:, 54:] slot of its output, which corresponds to the
    actual hand we care about. For a right hand we feed the image as-is.

    Returns a 54-dim numpy float32 vector.
    """
    from ...sam_3d_body.data.utils.prepare_batch import prepare_batch
    from ...sam_3d_body.utils import recursive_to

    img = hand_rgb_uint8
    if is_left:
        img = np.ascontiguousarray(img[:, ::-1])
    h, w = img.shape[:2]
    bbox = np.array([[0, 0, w, h]], dtype=np.float32)
    with torch.no_grad():
        batch = prepare_batch(img, estimator.transform_hand, bbox)
        batch = recursive_to(batch, estimator.device)
        estimator.model._initialize_batch(batch)
        pose_output = estimator.model.forward_step(batch, decoder_type="hand")
    hand = pose_output["mhr_hand"]["hand"]  # (B, 108) — 54 left + 54 right
    return hand[0, 54:].detach().cpu().numpy().astype(np.float32)


def splice_hand_into_params(
    hand_pose_params,
    *,
    lhand_params: np.ndarray | None = None,
    rhand_params: np.ndarray | None = None,
) -> np.ndarray:
    """Splice user-provided hand params into a (108,) hand_pose_params
    vector. Accepts ``None`` / lists / arrays for the source vector.
    Returns a fresh (108,) float32 array — does not mutate the input.
    """
    if hand_pose_params is None:
        out = np.zeros((108,), dtype=np.float32)
    else:
        out = np.asarray(hand_pose_params, dtype=np.float32).reshape(-1).copy()
        if out.size != 108:
            fixed = np.zeros((108,), dtype=np.float32)
            fixed[: min(108, out.size)] = out[: min(108, out.size)]
            out = fixed
    if lhand_params is not None:
        out[:54] = np.asarray(lhand_params, dtype=np.float32).reshape(-1)[:54]
    if rhand_params is not None:
        out[54:] = np.asarray(rhand_params, dtype=np.float32).reshape(-1)[:54]
    return out

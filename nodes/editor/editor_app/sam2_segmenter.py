# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""SAM 2 (hiera-tiny) backend for per-instance segmentation.

The Pose Editor + lets the user draw one bounding box per person. Those
boxes are forwarded to SAM 2 as box prompts; we keep the largest connected
component of the resulting mask so a person silhouette doesn't fragment into
extra blobs (loose hair, shadows, the floor under their feet, etc.).

Weights live under ``<ComfyUI>/models/sam2/tiny/`` and are auto-downloaded
from ``facebook/sam2-hiera-tiny`` on first call.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

from . import paths

log = logging.getLogger(__name__)

# Candidate (yaml_filename, ckpt_filename) pairs in <ComfyUI>/models/sam2/tiny/.
# We try v2.1 first (newer schema, recommended); fall back to v1 if only
# the older files are present locally.
_SAM2_LOCAL_CANDIDATES = (
    ("sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    ("sam2_hiera_t.yaml",   "sam2_hiera_tiny.pt"),
)
# If neither pair is present we fall back to downloading the v2.1 set
# from this HF repo into the local dir.
_SAM2_FALLBACK_REPO = "facebook/sam2.1-hiera-tiny"
_SAM2_FALLBACK_FILES = ("sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt")

_lock = threading.Lock()
_predictor = None
_device: str | None = None
# Last image fingerprint we ran ``predictor.set_image`` against. SAM 2's
# image encoder is the expensive part of the pipeline (~hundreds of ms on
# the tiny model); caching the embeddings lets the multi-bbox preview
# iterate at decoder-only speed (~ms per box).
_last_image_fingerprint: str | None = None


@dataclass
class MultiMaskResult:
    masks: np.ndarray        # (N, H, W) uint8 — 0/1
    scores: np.ndarray       # (N,) float32 — model-reported confidence


# ---------------------------------------------------------------------------
# Weight management
# ---------------------------------------------------------------------------

def _resolve_local_files() -> tuple[Path, Path]:
    """Locate a (yaml, .pt) pair under ``<ComfyUI>/models/sam2/tiny/``.

    Returns the first candidate whose files both exist. If neither pair
    is present, downloads the v2.1 set from HF and returns that.
    """
    target = paths.sam2_dir()
    target.mkdir(parents=True, exist_ok=True)

    for yaml_name, ckpt_name in _SAM2_LOCAL_CANDIDATES:
        y = target / yaml_name
        c = target / ckpt_name
        if y.is_file() and c.is_file():
            return y, c

    log.info("No local SAM 2 files under %s; downloading %s",
             target, _SAM2_FALLBACK_REPO)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download SAM 2 weights"
        ) from exc
    for fname in _SAM2_FALLBACK_FILES:
        hf_hub_download(
            repo_id=_SAM2_FALLBACK_REPO,
            filename=fname,
            local_dir=str(target),
        )
    yaml_name, ckpt_name = _SAM2_FALLBACK_FILES
    y = target / yaml_name
    c = target / ckpt_name
    if not (y.is_file() and c.is_file()):
        raise RuntimeError(
            f"Download finished but expected files are missing under {target}"
        )
    return y, c


def _build_model_from_local(yaml_path: Path, ckpt_path: Path, device: str):
    """Instantiate the SAM 2 model directly from the local YAML config and
    .pt checkpoint, bypassing the ``sam2`` package's bundled-config
    Hydra registry. Mirrors what ``build_sam2`` does internally:

        cfg = OmegaConf.load(yaml_path)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        model.load_state_dict(torch.load(ckpt_path)["model"])
        model.to(device).eval()

    The post-processing overrides ``build_sam2`` applies for inference
    are appended to the config before instantiation so the loaded model
    behaves identically.
    """
    import torch
    import torch.nn as nn
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    log.info("Loading SAM 2 from %s + %s on %s", yaml_path.name, ckpt_path.name, device)
    cfg = OmegaConf.load(str(yaml_path))

    # Match the inference-time overrides the image-predictor ``build_sam2``
    # applies via Hydra. The video predictor adds two more
    # (``binarize_mask_from_pts_for_mem_enc``, ``fill_hole_area``) but
    # those aren't accepted by ``SAM2Base.__init__`` and only appear for
    # the video flow — applying them here breaks ``instantiate``.
    overrides = OmegaConf.from_dotlist([
        "model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
        "model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
        "model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
    ])
    cfg = OmegaConf.merge(cfg, overrides)
    OmegaConf.resolve(cfg)

    model = instantiate(cfg.model, _recursive_=True)
    if not isinstance(model, nn.Module):
        raise RuntimeError(
            f"hydra.instantiate returned a {type(model).__name__} "
            f"instead of an nn.Module — check the YAML's model._target_"
        )

    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        log.warning("SAM 2 state_dict missing keys (%d): %s%s",
                    len(missing), missing[:5], " ..." if len(missing) > 5 else "")
    if unexpected:
        log.warning("SAM 2 state_dict unexpected keys (%d): %s%s",
                    len(unexpected), unexpected[:5], " ..." if len(unexpected) > 5 else "")

    model = model.to(device).eval()
    return model


def _load_predictor():
    global _predictor, _device
    with _lock:
        if _predictor is not None:
            return _predictor, _device or "cpu"

        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "SAM 2 backend requires the 'sam2' package. Install it with "
                "`pip install sam2` (also pulls in `hydra-core`)."
            ) from exc

        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        yaml_path, ckpt_path = _resolve_local_files()
        sam2_model = _build_model_from_local(yaml_path, ckpt_path, device)

        _predictor = SAM2ImagePredictor(sam2_model)
        _device = device
        return _predictor, device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _largest_component(mask_2d: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_2d.astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return mask_2d.astype(np.uint8)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8)


def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _mask_bbox_xyxy(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32
    )


def _segment_one_box(
    predictor,
    box_xyxy: np.ndarray,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    """Run a single positive-box SAM 2 query and return a clean uint8 mask.

    Picks the multimask candidate whose mask-derived bbox best matches
    the prompt (more reliable than raw confidence under heavy overlap),
    then keeps only the largest connected component.
    """
    box = np.array(
        [
            max(0.0, float(box_xyxy[0])),
            max(0.0, float(box_xyxy[1])),
            min(float(image_w), float(box_xyxy[2])),
            min(float(image_h), float(box_xyxy[3])),
        ],
        dtype=np.float32,
    )
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return np.zeros((image_h, image_w), dtype=np.uint8)
    masks, ious, _ = predictor.predict(
        box=box[None, :],
        multimask_output=True,
    )
    best_idx = -1
    best_iou = -1.0
    for k in range(masks.shape[0]):
        m = (masks[k] > 0).astype(np.uint8)
        if not m.any():
            continue
        m_bbox = _mask_bbox_xyxy(m)
        iou = _bbox_iou_xyxy(box, m_bbox)
        if iou > best_iou:
            best_iou = iou
            best_idx = k
    if best_idx < 0:
        best_idx = int(np.argmax(ious))
    picked = (masks[best_idx] > 0).astype(np.uint8)
    return _largest_component(picked)


def _segment_many_and_union(
    predictor,
    bboxes: np.ndarray | None,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    """Segment each bbox independently and OR the results into a single
    mask. Used to assemble the "exclude this region" mask for a person
    by running SAM 2 over each user-drawn negative bbox."""
    out = np.zeros((image_h, image_w), dtype=bool)
    if bboxes is None:
        return out.astype(np.uint8)
    arr = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    for nb in arr:
        m = _segment_one_box(predictor, nb, image_h, image_w)
        if m.any():
            out |= m.astype(bool)
    return out.astype(np.uint8)


def predict_masks(
    image_rgb: np.ndarray,
    bboxes_xyxy: np.ndarray,
    negative_bboxes_per_person: list[np.ndarray] | None = None,
    additional_bboxes_per_person: list[np.ndarray] | None = None,
) -> MultiMaskResult:
    """Run SAM 2 per bbox, then union/subtract user-drawn extras.

    Per-person mask = (primary_mask ∪ additional_masks) − negative_masks
    where each component mask comes from an independent SAM 2 box prompt
    and the unions / subtractions happen at pixel granularity.

    ``additional_bboxes_per_person`` and ``negative_bboxes_per_person``
    are optional length-N lists of ``(K_i, 4)`` arrays. Person order
    carries no semantic weight — each person's mask is resolved purely
    from their own boxes, with no inter-person subtraction. If the user
    wants two persons to be disjoint where they overlap, they draw a
    negative bbox on the appropriate side themselves.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError(f"image_rgb must be (H, W, 3) uint8 RGB, got {image_rgb.shape}")
    boxes = np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
    if boxes.shape[0] == 0:
        raise ValueError("predict_masks requires at least one bbox")
    if negative_bboxes_per_person is None:
        negative_bboxes_per_person = [None] * boxes.shape[0]
    elif len(negative_bboxes_per_person) != boxes.shape[0]:
        raise ValueError(
            "negative_bboxes_per_person length must match bboxes_xyxy length"
        )
    if additional_bboxes_per_person is None:
        additional_bboxes_per_person = [None] * boxes.shape[0]
    elif len(additional_bboxes_per_person) != boxes.shape[0]:
        raise ValueError(
            "additional_bboxes_per_person length must match bboxes_xyxy length"
        )

    predictor, _device = _load_predictor()
    h, w = image_rgb.shape[:2]

    import hashlib
    import torch

    # Avoid re-encoding the image when the same bitmap is segmented again
    # with different prompts (typical on the bbox-preview path: each new
    # person triggers a fresh ``predict_masks`` call but the input image
    # hasn't changed).
    fp_hash = hashlib.md5()
    fp_hash.update(np.array(image_rgb.shape, dtype=np.int32).tobytes())
    fp_hash.update(np.ascontiguousarray(image_rgb[::32, ::32]).tobytes())
    fp = fp_hash.hexdigest()

    with torch.inference_mode():
        global _last_image_fingerprint
        if fp != _last_image_fingerprint:
            predictor.set_image(image_rgb)
            _last_image_fingerprint = fp

        masks_out: list[np.ndarray] = []
        scores_out: list[float] = []
        for i, box in enumerate(boxes):
            # 1. Primary bbox → person mask via standard SAM 2 prompt.
            person_mask = _segment_one_box(predictor, box, h, w)

            # 2. Additional positive bboxes for the same person — each
            # segmented independently and OR-ed onto the person mask so
            # parts SAM 2 missed (e.g., a hand sticking out past the main
            # box) can be added by drawing another rectangle around them.
            pos_extra_union = _segment_many_and_union(
                predictor, additional_bboxes_per_person[i], h, w,
            )
            if pos_extra_union.any():
                person_mask = (
                    person_mask.astype(bool) | pos_extra_union.astype(bool)
                ).astype(np.uint8)
                # Don't filter to a single component here: legitimate
                # additional regions (a separated hand, a stray hat) may
                # be intentionally disconnected from the primary body.

            # 3. Negative bboxes are likewise segmented as positive boxes
            # on their own; the union of their masks is SUBTRACTED from
            # the person mask. More controllable than SAM 2's native
            # negative-point prompt (which can shrink the target mask far
            # past what the user wanted).
            neg_union = _segment_many_and_union(
                predictor, negative_bboxes_per_person[i], h, w,
            )
            if neg_union.any():
                kept = person_mask.astype(bool) & ~neg_union.astype(bool)
                person_mask = _largest_component(kept.astype(np.uint8))

            masks_out.append(person_mask)
            # Score = pixel coverage of the final mask within the box; a
            # cheap stand-in for the model's IoU now that we may have
            # subtracted bits.
            x1 = max(0, int(round(float(box[0]))))
            y1 = max(0, int(round(float(box[1]))))
            x2 = min(w, int(round(float(box[2]))))
            y2 = min(h, int(round(float(box[3]))))
            if x2 > x1 and y2 > y1:
                area = float(((x2 - x1) * (y2 - y1)))
                covered = float(person_mask[y1:y2, x1:x2].sum())
                scores_out.append(covered / max(area, 1.0))
            else:
                scores_out.append(0.0)

    masks_np = np.stack(masks_out, axis=0).astype(np.uint8)
    scores_np = np.asarray(scores_out, dtype=np.float32)
    return MultiMaskResult(masks=masks_np, scores=scores_np)


def predict_masks_pil(
    pil_image: Image.Image,
    bboxes_xyxy: np.ndarray,
) -> MultiMaskResult:
    rgb = np.asarray(pil_image.convert("RGB"))
    return predict_masks(rgb, bboxes_xyxy)


__all__ = [
    "MultiMaskResult",
    "predict_masks",
    "predict_masks_pil",
]

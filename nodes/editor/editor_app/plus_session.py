# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""In-memory session cache for the Pose Editor +.

A ``PlusPoseSession`` packages every person detected in one input image
(plus the shared image metadata and camera intrinsics) so the renderer can
re-render the whole scene from cached pose params without re-running SAM 2
or SAM 3D Body. Lifecycle is per ``plus_job_id``; the underlying single-
person ``PoseSession`` store is left alone so existing nodes are unaffected.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any  # noqa: F401  (used in dataclass field annotation)

import numpy as np

from .pose_session import PoseSession


@dataclass
class PlusPersonSlot:
    """One person's worth of state inside a multi-person session."""
    person_id: str
    bbox_xyxy: np.ndarray            # (4,) float32 — user-drawn box, image space
    target_height_m: float           # the value the user typed (always meters internally)
    measured_height_m: float         # the mesh height before correction (used for diagnostics)
    height_scale: float              # ``target / measured`` — applied to verts and cam_t
    pose_session: PoseSession        # body / hand pose params + originals
    # Per-person editor state. Populated/updated from the frontend on each
    # /multi_render call. Layout matches what ``renderer._normalise_settings``
    # expects (body_params / bone_lengths / blendshapes / pose_adjust) so the
    # multi renderer can hand it straight to ``compute_session_mesh``.
    settings: dict[str, Any] = field(default_factory=dict)
    has_lhand_override: bool = False
    has_rhand_override: bool = False
    mask_score: float = 0.0


@dataclass
class PlusPoseSession:
    plus_job_id: str
    persons: list[PlusPersonSlot]   # ordered as added by the user
    image_width: int
    image_height: int
    focal_length: float              # shared across persons (single camera)
    pred_cam_ts: list[np.ndarray] = field(default_factory=list)
    # Settings overrides keyed by ``person_id``. Populated by the multi
    # renderer when the frontend tweaks a person's height — keeps recent
    # values around for the cache key without mutating the original
    # ``target_height_m`` (so a "reset" can fall back to the inference-time
    # height).
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


class _MultiSessionStore:
    def __init__(self, max_size: int = 8) -> None:
        self._data: "OrderedDict[str, PlusPoseSession]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_size

    def put(self, session: PlusPoseSession) -> None:
        with self._lock:
            self._data[session.plus_job_id] = session
            self._data.move_to_end(session.plus_job_id)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def get(self, plus_job_id: str) -> PlusPoseSession | None:
        with self._lock:
            sess = self._data.get(plus_job_id)
            if sess is not None:
                self._data.move_to_end(plus_job_id)
            return sess

    def drop(self, plus_job_id: str) -> None:
        with self._lock:
            self._data.pop(plus_job_id, None)


_store = _MultiSessionStore()


def put(session: PlusPoseSession) -> None:
    _store.put(session)


def get(plus_job_id: str) -> PlusPoseSession | None:
    return _store.get(plus_job_id)


def drop(plus_job_id: str) -> None:
    _store.drop(plus_job_id)

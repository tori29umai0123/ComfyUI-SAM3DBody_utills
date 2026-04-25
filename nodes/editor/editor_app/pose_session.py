"""In-memory session cache for pose results.

After `/api/process` runs segmentation + pose estimation we keep the pose tensors here so
`/api/render` can re-apply body-shape sliders without re-running the heavy
detection path. One entry per job_id; evicted by LRU size limit (default 16).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PoseSession:
    job_id: str
    pose_json: dict[str, Any]             # serialisable pose data (for rerun / debug)
    # Raw numpy arrays used by the renderer (kept as np to avoid CUDA reserve
    # on idle sessions):
    global_rot: np.ndarray                # (3,) or (1, 3)
    body_pose_params: np.ndarray          # (133,)
    hand_pose_params: np.ndarray          # (108,)
    image_width: int
    image_height: int
    orig_focal_length: float | None
    orig_cam_t: np.ndarray | None         # (3,)
    orig_keypoints_3d: np.ndarray | None  # (K, 3)
    bbox_xyxy: np.ndarray | None          # (4,)


class _SessionStore:
    def __init__(self, max_size: int = 16) -> None:
        self._data: "OrderedDict[str, PoseSession]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_size

    def put(self, session: PoseSession) -> None:
        with self._lock:
            self._data[session.job_id] = session
            self._data.move_to_end(session.job_id)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def get(self, job_id: str) -> PoseSession | None:
        with self._lock:
            sess = self._data.get(job_id)
            if sess is not None:
                self._data.move_to_end(job_id)
            return sess

    def drop(self, job_id: str) -> None:
        with self._lock:
            self._data.pop(job_id, None)


_store = _SessionStore()


# Stable job_id for the "Character Make" tab — a zero-pose session so
# slider changes render the MHR neutral body regardless of any actual
# subject that might have been processed earlier.
MAKE_JOB_ID = "make"


def put(session: PoseSession) -> None:
    _store.put(session)


def get(job_id: str) -> PoseSession | None:
    return _store.get(job_id)


def drop(job_id: str) -> None:
    _store.drop(job_id)


def ensure_make_session() -> PoseSession:
    """Return the fixed-id "make" session, creating it on demand. Its pose
    params are all zero so the MHR rest (T-pose) body is rendered."""
    sess = _store.get(MAKE_JOB_ID)
    if sess is not None:
        return sess
    sess = PoseSession(
        job_id=MAKE_JOB_ID,
        pose_json={},
        global_rot=np.zeros(3, dtype=np.float32),
        body_pose_params=np.zeros(133, dtype=np.float32),
        hand_pose_params=np.zeros(108, dtype=np.float32),
        image_width=0,
        image_height=0,
        orig_focal_length=None,
        orig_cam_t=None,
        orig_keypoints_3d=None,
        bbox_xyxy=None,
    )
    _store.put(sess)
    return sess

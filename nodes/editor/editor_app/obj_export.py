"""Write an OBJ file from a SAM3DBody inference result (vertices + faces)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def write_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    header: str = "# sam3dbody-standalone",
) -> Path:
    """Write `vertices` (V,3) + `faces` (F,3) to `path` as a minimal Wavefront OBJ.

    Faces are 1-indexed per OBJ convention. No texture / normal data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (V,3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {f.shape}")

    # Fastest practical path: build one big string, write once.
    lines: list[str] = [header, f"# vertices: {len(v)} faces: {len(f)}"]
    lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v)
    # +1 for OBJ's 1-based indexing.
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f)
    path.write_text("\n".join(lines), encoding="ascii")
    return path


def write_obj_flip_y(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    **kwargs,
) -> Path:
    """Convert SAM3DBody's OpenCV-style output (Y-down, Z-forward-into-scene)
    to the OpenGL / three.js convention (Y-up, Z-back-toward-camera).

    The correct transform is a 180° rotation around the X axis, i.e. flip
    BOTH Y and Z. Using a *rotation* (two reflections cancel) preserves
    chirality so the body's left/right isn't mirrored, and preserves
    triangle winding so no face re-ordering is needed.
    """
    v = np.asarray(vertices, dtype=np.float32).copy()
    v[:, 1] = -v[:, 1]
    v[:, 2] = -v[:, 2]
    return write_obj(path, v, faces, **kwargs)

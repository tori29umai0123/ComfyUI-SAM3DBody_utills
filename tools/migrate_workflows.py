"""Migrate ComfyUI workflows from the legacy SAM3DBodyRenderFromJson
node (single render-with-body-preset-sliders) to the new pair:

  SAM3DBodySettingBodyPresetJson           (preset + 9 body + 4 bone + N bs widgets)
  SAM3DBodyRenderFromPoseAndBodyPresetJson (model + pose_json + body_preset_json
                                       + camera widgets + bg)

For every workflow JSON found under the supplied roots, every old node
is replaced by a (Setting, Render) pair and all touching links are
re-routed. Per-slider widget values + non-shape widgets (pose_json
literal, camera offsets, lean correction) are preserved.

Usage:
    python tools/migrate_workflows.py [--dry-run] DIR1 [DIR2 ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Slot orderings of the legacy node's widgets_values (matches its
# INPUT_TYPES ordering at the moment of removal). Index = slot in the
# old widgets_values list.
LEGACY_WIDGET_LAYOUT = [
    "pose_json",         # 0
    "preset",            # 1
    "offset_x",          # 2
    "offset_y",          # 3
    "scale_offset",      # 4
    "camera_yaw_deg",    # 5
    "camera_pitch_deg",  # 6
    "width",             # 7
    "height",            # 8
    "pose_adjust",       # 9
    "body_fat",          # 10
    "body_muscle",       # 11
    "body_fat_muscle",   # 12
    "body_limb_girth",   # 13
    "body_limb_muscle",  # 14
    "body_limb_fat",     # 15
    "body_chest_shoulder",  # 16
    "body_waist_hip",    # 17
    "body_thigh_calf",   # 18
    "bone_torso",        # 19
    "bone_neck",         # 20
    "bone_arm",          # 21
    "bone_leg",          # 22
]
# Anything past slot 22 is a bs_<name> entry (variable-length, depends
# on the npz the workflow was authored against).

BODY_PRESET_NODE_TYPE = "SAM3DBodySettingBodyPresetJson"
RENDER_NODE_TYPE = "SAM3DBodyRenderFromPoseAndBodyPresetJson"
LEGACY_NODE_TYPE = "SAM3DBodyRenderFromJson"


def _safe_get(seq, idx, default):
    try:
        return seq[idx]
    except IndexError:
        return default


def _make_body_preset_node(*, node_id, old_node):
    """Build the new SettingBodyPresetJson node from the old node's slider state."""
    wv = old_node.get("widgets_values", [])
    body_preset_widgets = [
        _safe_get(wv,  1, "autosave"),
        _safe_get(wv, 10, 0.0),  # body_fat
        _safe_get(wv, 11, 0.0),  # body_muscle
        _safe_get(wv, 12, 0.0),  # body_fat_muscle
        _safe_get(wv, 13, 0.0),  # body_limb_girth
        _safe_get(wv, 14, 0.0),  # body_limb_muscle
        _safe_get(wv, 15, 0.0),  # body_limb_fat
        _safe_get(wv, 16, 0.0),  # body_chest_shoulder
        _safe_get(wv, 17, 0.0),  # body_waist_hip
        _safe_get(wv, 18, 0.0),  # body_thigh_calf
        _safe_get(wv, 19, 1.0),  # bone_torso
        _safe_get(wv, 20, 1.0),  # bone_neck
        _safe_get(wv, 21, 1.0),  # bone_arm
        _safe_get(wv, 22, 1.0),  # bone_leg
        *wv[23:],                # bs_* values
    ]
    pos = old_node.get("pos") or [0, 0]
    size = old_node.get("size") or [380, 1086]
    return {
        "id": node_id,
        "type": BODY_PRESET_NODE_TYPE,
        "pos": [pos[0] - 460, pos[1]],
        "size": [380, max(600, size[1] if isinstance(size, list) else 1086)],
        "flags": {},
        "order": old_node.get("order", 0),
        "mode": 0,
        "inputs": [],
        "outputs": [{
            "name": "body_preset_json",
            "type": "STRING",
            "links": [],  # filled in below when we add the C→R link
            "slot_index": 0,
        }],
        "properties": {"Node name for S&R": BODY_PRESET_NODE_TYPE},
        "widgets_values": body_preset_widgets,
    }


def _make_render_node(*, node_id, old_node):
    """Build the new RenderFromPoseAndBodyPresetJson node, taking the camera
    + pose_adjust widget values from the old node."""
    wv = old_node.get("widgets_values", [])
    render_widgets = [
        _safe_get(wv, 0, "{}"),       # pose_json widget value (literal)
        "{}",                          # body_preset_json widget value (link replaces)
        _safe_get(wv, 2, 0.0),        # offset_x
        _safe_get(wv, 3, 0.0),        # offset_y
        _safe_get(wv, 4, 1.0),        # scale_offset
        _safe_get(wv, 5, 0.0),        # camera_yaw_deg
        _safe_get(wv, 6, 0.0),        # camera_pitch_deg
        _safe_get(wv, 7, 0),          # width
        _safe_get(wv, 8, 0),          # height
        _safe_get(wv, 9, 0.0),        # pose_adjust
    ]
    pos = old_node.get("pos") or [0, 0]
    size = old_node.get("size") or [380, 1086]
    width = size[0] if isinstance(size, list) else 380
    return {
        "id": node_id,
        "type": RENDER_NODE_TYPE,
        "pos": pos,
        "size": [width, 320],
        "flags": {},
        "order": old_node.get("order", 0),
        "mode": 0,
        "inputs": [
            {"name": "model",            "type": "SAM3D_MODEL", "link": None},
            {"name": "pose_json",        "type": "STRING",
             "widget": {"name": "pose_json"},  "link": None},
            {"name": "body_preset_json",       "type": "STRING",
             "widget": {"name": "body_preset_json"}, "link": None},
            {"name": "background_image", "type": "IMAGE", "shape": 7, "link": None},
        ],
        "outputs": [
            {"name": "image", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "settings_json", "type": "STRING", "links": [], "slot_index": 1},
        ],
        "properties": {"Node name for S&R": RENDER_NODE_TYPE},
        "widgets_values": render_widgets,
    }


def _migrate_workflow(wf: dict) -> int:
    """Replace every legacy node in ``wf`` (in place). Returns the count."""
    legacy_nodes = [n for n in wf.get("nodes", []) if n.get("type") == LEGACY_NODE_TYPE]
    if not legacy_nodes:
        return 0

    last_node_id = int(wf.get("last_node_id", 0)) or max(
        (n.get("id", 0) for n in wf["nodes"]), default=0
    )
    last_link_id = int(wf.get("last_link_id", 0)) or max(
        ((L[0] for L in wf.get("links", [])) if wf.get("links") else [0]),
        default=0,
    )
    if not isinstance(wf.get("links"), list):
        wf["links"] = []

    for old in legacy_nodes:
        last_node_id += 1
        body_preset_id = last_node_id
        last_node_id += 1
        render_id = last_node_id

        body_preset = _make_body_preset_node(node_id=body_preset_id, old_node=old)
        render = _make_render_node(node_id=render_id, old_node=old)

        # 1) C → R body_preset_json link.
        last_link_id += 1
        body_preset_link = [last_link_id, body_preset_id, 0, render_id, 2, "STRING"]
        wf["links"].append(body_preset_link)
        body_preset["outputs"][0]["links"].append(last_link_id)
        render["inputs"][2]["link"] = last_link_id

        # 2) Re-route every link that used to touch the old node.
        legacy_inputs  = old.get("inputs",  [])
        legacy_outputs = old.get("outputs", [])

        for L in wf["links"]:
            link_id, from_id, from_slot, to_id, to_slot, ltype = L
            if to_id == old["id"]:
                inp_name = (legacy_inputs[to_slot] or {}).get("name") if to_slot < len(legacy_inputs) else None
                # Render-side input mapping
                target_idx = None
                for i, inp in enumerate(render["inputs"]):
                    if inp["name"] == inp_name:
                        target_idx = i
                        break
                if target_idx is None:
                    print(f"  warn: dropping incoming link {link_id} "
                          f"(input {inp_name!r} not present on new render node)")
                    continue
                L[3] = render_id
                L[4] = target_idx
                render["inputs"][target_idx]["link"] = link_id
            elif from_id == old["id"]:
                out_name = (legacy_outputs[from_slot] or {}).get("name") if from_slot < len(legacy_outputs) else None
                target_idx = None
                for i, out in enumerate(render["outputs"]):
                    if out["name"] == out_name:
                        target_idx = i
                        break
                if target_idx is None:
                    print(f"  warn: dropping outgoing link {link_id} "
                          f"(output {out_name!r} not present on new render node)")
                    continue
                L[1] = render_id
                L[2] = target_idx
                render["outputs"][target_idx]["links"].append(link_id)

        # 3) Replace old node with the two new ones.
        wf["nodes"].remove(old)
        wf["nodes"].append(body_preset)
        wf["nodes"].append(render)

    wf["last_node_id"] = last_node_id
    wf["last_link_id"] = last_link_id
    return len(legacy_nodes)


def migrate_file(path: Path, *, dry_run: bool) -> tuple[int, str]:
    text = path.read_text(encoding="utf-8")
    try:
        wf = json.loads(text)
    except Exception as exc:
        return -1, f"{path}: JSON parse error: {exc}"
    if not isinstance(wf, dict) or "nodes" not in wf:
        return 0, f"{path}: not a workflow (no 'nodes' key)"
    n = _migrate_workflow(wf)
    if n == 0:
        return 0, f"{path}: 0 legacy nodes (skip)"
    if dry_run:
        return n, f"{path}: would migrate {n} legacy node(s) (dry-run)"
    backup = path.with_suffix(
        path.suffix + ".bak-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    shutil.copy2(path, backup)
    path.write_text(
        json.dumps(wf, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return n, f"{path}: migrated {n} legacy node(s) (backup: {backup.name})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report changes without writing")
    ap.add_argument("roots", nargs="+", type=Path,
                    help="directories to scan for *.json workflows")
    args = ap.parse_args()

    total_files = 0
    total_nodes = 0
    for root in args.roots:
        if not root.is_dir():
            print(f"skip (not a dir): {root}")
            continue
        for f in sorted(root.glob("*.json")):
            n, msg = migrate_file(f, dry_run=args.dry_run)
            print(msg)
            if n > 0:
                total_files += 1
                total_nodes += n
    print(f"\nDONE: {total_files} file(s), {total_nodes} node replacement(s).")


if __name__ == "__main__":
    main()

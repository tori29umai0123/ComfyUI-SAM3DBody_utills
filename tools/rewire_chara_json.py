"""Rewire any downstream ``character_json`` (or ``chara_json``) input
that used to be fed by the legacy Render node's ``settings_json`` output
so it now reads directly from the workflow's
``SAM3DBodySettingCharaJson`` node's ``chara_json`` output.

Run this BEFORE strip_process_image_output.py with --producer
SAM3DBodyRenderFromPoseAndCharaJson --slot 1 --name settings_json so
the strip step doesn't leave export nodes dangling.

Usage:
    python tools/rewire_chara_json.py [--dry-run] DIR1 [DIR2 ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

RENDER_TYPE = "SAM3DBodyRenderFromPoseAndCharaJson"
SETTING_TYPE = "SAM3DBodySettingCharaJson"
SETTINGS_JSON_SLOT = 1            # the slot to be replaced
SETTINGS_JSON_NAME = "settings_json"
CHARA_INPUT_NAMES = {"character_json", "chara_json"}


def _rewire(wf: dict) -> tuple[int, list[str]]:
    """Reroute compatible links. Returns (#rewired, log lines)."""
    nodes = wf.get("nodes", [])
    links = wf.get("links", [])

    setting_nodes = [n for n in nodes if n.get("type") == SETTING_TYPE]
    if not setting_nodes:
        return 0, []
    if len(setting_nodes) > 1:
        return 0, [f"  warn: multiple {SETTING_TYPE} nodes — manual rewire recommended"]
    setting = setting_nodes[0]
    setting_id = setting["id"]
    # Locate the chara_json output slot (slot index in `outputs` list).
    chara_slot = None
    for i, o in enumerate(setting.get("outputs", [])):
        if o.get("name") == "chara_json":
            chara_slot = i
            break
    if chara_slot is None:
        return 0, [f"  warn: {SETTING_TYPE} has no chara_json output"]

    notes: list[str] = []
    rewired = 0
    render_nodes = [n for n in nodes if n.get("type") == RENDER_TYPE]

    for render in render_nodes:
        # Find each link from render.settings_json (slot 1).
        candidate_links = [
            L for L in links
            if L[1] == render["id"] and L[2] == SETTINGS_JSON_SLOT
        ]
        for L in candidate_links:
            link_id, _from_id, _from_slot, to_id, to_slot, _ltype = L
            target_node = next((n for n in nodes if n.get("id") == to_id), None)
            if target_node is None:
                continue
            target_input = (target_node.get("inputs") or [None])[to_slot] \
                if to_slot < len(target_node.get("inputs", [])) else None
            if not target_input:
                continue
            inp_name = target_input.get("name")
            if inp_name not in CHARA_INPUT_NAMES:
                continue  # leave unrelated downstream consumers alone

            # Re-route this link to setting.chara_json.
            L[1] = setting_id
            L[2] = chara_slot
            # Also remove from render.outputs[slot].links and add to setting.outputs[chara_slot].links.
            try:
                render["outputs"][SETTINGS_JSON_SLOT]["links"] = [
                    lid for lid in render["outputs"][SETTINGS_JSON_SLOT].get("links", [])
                    if lid != link_id
                ]
            except (KeyError, IndexError):
                pass
            sout = setting["outputs"][chara_slot]
            sout.setdefault("links", [])
            if link_id not in sout["links"]:
                sout["links"].append(link_id)

            rewired += 1
            notes.append(
                f"  link {link_id}: render({render['id']}).settings_json "
                f"-> setting({setting_id}).chara_json (was -> "
                f"{target_node.get('type')}.{inp_name})"
            )

    return rewired, notes


def _do_file(path: Path, *, dry_run: bool) -> tuple[int, str]:
    try:
        wf = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return -1, f"{path}: JSON parse error: {exc}"
    if not isinstance(wf, dict) or "nodes" not in wf:
        return 0, ""
    n, notes = _rewire(wf)
    if n == 0 and not notes:
        return 0, ""
    if n == 0:
        return 0, f"{path}:\n" + "\n".join(notes)
    if dry_run:
        return n, f"{path}: would rewire {n} link(s) (dry-run)\n" + "\n".join(notes)
    backup = path.with_suffix(
        path.suffix + ".bak-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    shutil.copy2(path, backup)
    path.write_text(
        json.dumps(wf, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return n, f"{path}: rewired {n} link(s) (backup: {backup.name})\n" + "\n".join(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("roots", nargs="+", type=Path)
    args = ap.parse_args()

    total = 0
    for root in args.roots:
        if not root.is_dir():
            continue
        for f in sorted(root.glob("*.json")):
            n, msg = _do_file(f, dry_run=args.dry_run)
            if msg:
                print(msg)
            if n > 0:
                total += 1
    print(f"\nDONE: {total} file(s) rewired.")


if __name__ == "__main__":
    main()

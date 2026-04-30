"""Strip an obsolete output slot from a given node type in workflow JSONs.

Use cases (one CLI call each):
  - SAM3DBodyProcessToJson: drop slot 1 ``image`` (pose_json only now).
  - SAM3DBodyRenderFromPoseAndBodyPresetJson: drop slot 1 ``settings_json``
    (image only now).

Each invocation:
  - drops outgoing links from the dropped slot
  - clears the corresponding entries from the downstream nodes' inputs
  - removes the output entry from the producing node

Usage:
    python tools/strip_process_image_output.py [--dry-run] \\
        --producer NODE_TYPE --slot SLOT --name NAME DIR1 [DIR2 ...]

Defaults preserve the original behaviour (Process pose-only stripping)
so legacy invocations of this script keep working.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def _strip_workflow(wf: dict, *, producer_type: str, slot: int,
                    name: str) -> tuple[int, list[str]]:
    """Return (#producer nodes affected, log lines)."""
    if not isinstance(wf.get("nodes"), list):
        return 0, []
    producers = [n for n in wf["nodes"] if n.get("type") == producer_type]
    if not producers:
        return 0, []

    notes: list[str] = []
    affected = 0

    for prod in producers:
        outputs = prod.get("outputs", [])
        if len(outputs) <= slot:
            # Already stripped.
            continue
        if outputs[slot].get("name") != name:
            notes.append(
                f"node {prod.get('id')}: slot {slot} is "
                f"{outputs[slot].get('name')!r}, expected "
                f"{name!r} — skipping"
            )
            continue

        affected += 1
        # 1) Collect outgoing links from the dropped slot.
        bad_links = [L for L in wf.get("links", [])
                     if L[1] == prod["id"] and L[2] == slot]
        bad_link_ids = {L[0] for L in bad_links}

        # 2) Clear those links from downstream node inputs.
        for L in bad_links:
            link_id, _from_id, _from_slot, to_id, to_slot, _ltype = L
            for n in wf["nodes"]:
                if n.get("id") != to_id:
                    continue
                inputs = n.get("inputs", [])
                if 0 <= to_slot < len(inputs):
                    inputs[to_slot]["link"] = None
                    notes.append(
                        f"  cleared link {link_id} on node {to_id} input "
                        f"slot {to_slot} ({inputs[to_slot].get('name')!r})"
                    )
                break

        # 3) Drop the bad links from the link list.
        wf["links"] = [L for L in wf.get("links", []) if L[0] not in bad_link_ids]

        # 4) Drop the now-orphan output entry from the producer.
        prod["outputs"] = outputs[:slot]

        notes.append(
            f"node {prod.get('id')} ({producer_type}): removed slot "
            f"{slot} ({name!r}); dropped {len(bad_link_ids)} link(s)"
        )

    return affected, notes


def _strip_file(path: Path, *, dry_run: bool, producer_type: str, slot: int,
                name: str) -> tuple[int, str]:
    try:
        wf = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return -1, f"{path}: JSON parse error: {exc}"
    if not isinstance(wf, dict) or "nodes" not in wf:
        return 0, f"{path}: not a workflow (no 'nodes' key)"
    affected, notes = _strip_workflow(wf, producer_type=producer_type,
                                      slot=slot, name=name)
    if affected == 0:
        return 0, f"{path}: no producer nodes need stripping"
    if dry_run:
        return affected, (
            f"{path}: would update {affected} producer node(s) (dry-run)\n  "
            + "\n  ".join(notes)
        )
    backup = path.with_suffix(
        path.suffix + ".bak-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    shutil.copy2(path, backup)
    path.write_text(
        json.dumps(wf, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return affected, (
        f"{path}: stripped {affected} producer node(s) (backup: {backup.name})\n  "
        + "\n  ".join(notes)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report changes without writing")
    ap.add_argument("--producer", default="SAM3DBodyProcessToJson",
                    help="node type to strip a slot from")
    ap.add_argument("--slot", type=int, default=1,
                    help="output slot index to remove")
    ap.add_argument("--name", default="image",
                    help="expected output name (sanity check)")
    ap.add_argument("roots", nargs="+", type=Path,
                    help="directories to scan for *.json workflows")
    args = ap.parse_args()

    total = 0
    for root in args.roots:
        if not root.is_dir():
            print(f"skip (not a dir): {root}")
            continue
        for f in sorted(root.glob("*.json")):
            n, msg = _strip_file(
                f, dry_run=args.dry_run,
                producer_type=args.producer, slot=args.slot, name=args.name,
            )
            if n > 0:
                print(msg)
                total += 1
    print(f"\nDONE: {total} file(s) updated.")


if __name__ == "__main__":
    main()

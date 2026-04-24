"""Convert SAM3DBody MHR rig to a generic Humanoid bone layout.

The target naming scheme is the de-facto humanoid convention used by
Unity / Unreal / most DCCs (``Hips`` / ``Spine`` / ``LeftArm`` / ...), as
documented in ``docs/humanoid_mapping.md``. We use that scheme only as a
reference template — this module does not depend on any specific tool.

Invoked from ``tools/build_rigged_fbx.py`` / ``tools/build_animated_fbx.py``
after the armature is built and the mesh is LBS-skinned, but BEFORE pose
keyframes are inserted. This module:

  1. Transfers vertex-group (LBS) weights of to-be-deleted bones into the
     surviving bone that should "absorb" them.
  2. Re-parents the children of deleted bones onto the nearest surviving
     ancestor (or root), preserving world rest positions.
  3. Deletes the now-orphaned bones and their matching vertex groups.
  4. Renames surviving bones to humanoid names.
  5. Promotes ``pelvis`` to the armature root (``Hips``), discarding the
     top-level ``joint_000`` shell.

Pose keyframing happens AFTER this conversion; callers must recompute each
surviving bone's local-delta rotation using its NEW parent's MHR rotation
(since the merge collapses intermediate bones). Convenience helper
``build_humanoid_to_dense`` produces the lookup the caller needs.
"""
from __future__ import annotations

import bpy
from mathutils import Vector


# ---------------------------------------------------------------------------
# MHR → Humanoid conversion table
# ---------------------------------------------------------------------------
#
# Each entry maps an MHR bone name to (new_humanoid_name, merge_target_mhr).
#
#   - (humanoid_name, None) → bone survives and is renamed to humanoid_name.
#   - (None, target)        → bone is DELETED; its vertex-group weights are
#                             added into `target` (also an MHR name). Any
#                             children are re-parented to the nearest
#                             surviving ancestor.
#   - (None, None)          → bone removed entirely (only used for joint_000,
#                             the MHR root shell; pelvis takes over as Hips).
#
# Derivation of finger / toe / twist mappings was done by dumping an actual
# output FBX under Blender and measuring hand-local / calf-local coordinates
# of each finger / toe root. See ``docs/humanoid_mapping.md``.

BONE_ACTIONS: dict[str, tuple[str | None, str | None]] = {
    # --- Root / body ---
    "joint_000": (None,           None),        # removed; pelvis → Hips
    "pelvis":    ("Hips",         None),
    "joint_034": (None,           "pelvis"),    # spine-root insertion
    "spine_01":  ("Spine",        None),
    "spine_02":  ("Spine1",       None),
    "spine_03":  ("Spine2",       None),
    "neck_01":   ("Neck",         None),
    "joint_111": (None,           "neck_01"),   # neck helper
    "head":      ("Head",         None),
    "joint_114": ("HeadTop_End",  None),

    # --- Left arm ---
    "clavicle_l": ("LeftShoulder", None),
    "upperarm_l": ("LeftArm",      None),
    "joint_105":  (None,           "upperarm_l"),
    "joint_106":  (None,           "upperarm_l"),
    "joint_107":  (None,           "upperarm_l"),
    "joint_108":  (None,           "upperarm_l"),
    "joint_109":  (None,           "upperarm_l"),
    "lowerarm_l": ("LeftForeArm",  None),
    "joint_101":  (None,           "lowerarm_l"),
    "joint_102":  (None,           "lowerarm_l"),
    "joint_103":  (None,           "lowerarm_l"),
    "joint_104":  (None,           "lowerarm_l"),
    "joint_077":  (None,           "lowerarm_l"),  # forearm→hand insertion
    "hand_l":     ("LeftHand",     None),

    # Left fingers (see docs/humanoid_mapping.md § 2)
    "joint_096": ("LeftHandThumb1",  None),
    "joint_097": ("LeftHandThumb2",  None),
    "joint_098": ("LeftHandThumb3",  None),
    "joint_099": (None,              "joint_098"),  # absorb extra knuckle
    "joint_100": ("LeftHandThumb4",  None),

    "joint_092": ("LeftHandIndex1",  None),
    "joint_093": ("LeftHandIndex2",  None),
    "joint_094": ("LeftHandIndex3",  None),
    "joint_095": ("LeftHandIndex4",  None),

    "joint_088": ("LeftHandMiddle1", None),
    "joint_089": ("LeftHandMiddle2", None),
    "joint_090": ("LeftHandMiddle3", None),
    "joint_091": ("LeftHandMiddle4", None),

    "joint_084": ("LeftHandRing1",   None),
    "joint_085": ("LeftHandRing2",   None),
    "joint_086": ("LeftHandRing3",   None),
    "joint_087": ("LeftHandRing4",   None),

    "joint_079": (None,              "hand_l"),  # pinky metacarpal → hand
    "joint_080": ("LeftHandPinky1",  None),
    "joint_081": ("LeftHandPinky2",  None),
    "joint_082": ("LeftHandPinky3",  None),
    "joint_083": ("LeftHandPinky4",  None),

    # --- Right arm ---
    "clavicle_r": ("RightShoulder", None),
    "upperarm_r": ("RightArm",      None),
    "joint_069":  (None,            "upperarm_r"),
    "joint_070":  (None,            "upperarm_r"),
    "joint_071":  (None,            "upperarm_r"),
    "joint_072":  (None,            "upperarm_r"),
    "joint_073":  (None,            "upperarm_r"),
    "lowerarm_r": ("RightForeArm",  None),
    "joint_065":  (None,            "lowerarm_r"),
    "joint_066":  (None,            "lowerarm_r"),
    "joint_067":  (None,            "lowerarm_r"),
    "joint_068":  (None,            "lowerarm_r"),
    "joint_041":  (None,            "lowerarm_r"),
    "hand_r":     ("RightHand",     None),

    # Right fingers
    "joint_060": ("RightHandThumb1",  None),
    "joint_061": ("RightHandThumb2",  None),
    "joint_062": ("RightHandThumb3",  None),
    "joint_063": (None,              "joint_062"),
    "joint_064": ("RightHandThumb4",  None),

    "joint_056": ("RightHandIndex1",  None),
    "joint_057": ("RightHandIndex2",  None),
    "joint_058": ("RightHandIndex3",  None),
    "joint_059": ("RightHandIndex4",  None),

    "joint_052": ("RightHandMiddle1", None),
    "joint_053": ("RightHandMiddle2", None),
    "joint_054": ("RightHandMiddle3", None),
    "joint_055": ("RightHandMiddle4", None),

    "joint_048": ("RightHandRing1",   None),
    "joint_049": ("RightHandRing2",   None),
    "joint_050": ("RightHandRing3",   None),
    "joint_051": ("RightHandRing4",   None),

    "joint_043": (None,              "hand_r"),  # pinky metacarpal → hand
    "joint_044": ("RightHandPinky1",  None),
    "joint_045": ("RightHandPinky2",  None),
    "joint_046": ("RightHandPinky3",  None),
    "joint_047": ("RightHandPinky4",  None),

    # --- Left leg ---
    "thigh_l":   ("LeftUpLeg",    None),
    "joint_014": (None,           "thigh_l"),
    "joint_015": (None,           "thigh_l"),
    "joint_016": (None,           "thigh_l"),
    "joint_017": (None,           "thigh_l"),
    "calf_l":    ("LeftLeg",      None),
    "joint_009": (None,           "calf_l"),  # calf twist helpers
    "joint_010": (None,           "calf_l"),
    "joint_011": (None,           "calf_l"),
    "joint_012": (None,           "calf_l"),
    "foot_l":    ("LeftFoot",     None),
    "joint_005": ("LeftToeBase",  None),
    "joint_006": (None,           "joint_005"),
    "joint_007": (None,           "joint_005"),
    "joint_008": ("LeftToe_End",  None),

    # --- Right leg ---
    "thigh_r":   ("RightUpLeg",    None),
    "joint_030": (None,            "thigh_r"),
    "joint_031": (None,            "thigh_r"),
    "joint_032": (None,            "thigh_r"),
    "joint_033": (None,            "thigh_r"),
    "calf_r":    ("RightLeg",      None),
    "joint_025": (None,            "calf_r"),
    "joint_026": (None,            "calf_r"),
    "joint_027": (None,            "calf_r"),
    "joint_028": (None,            "calf_r"),
    "foot_r":    ("RightFoot",     None),
    "joint_021": ("RightToeBase",  None),
    "joint_022": (None,            "joint_021"),
    "joint_023": (None,            "joint_021"),
    "joint_024": ("RightToe_End",  None),
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def apply_humanoid_conversion(arm_obj, mesh_obj) -> dict[str, str]:
    """Reshape ``arm_obj`` + ``mesh_obj`` in place into Humanoid layout.

    Must be called in OBJECT mode. Returns a ``{mhr_name: humanoid_name}``
    map of surviving (renamed) bones; callers pair this with the original
    MHR rest/posed rotation arrays to drive pose keyframes on the new rig.
    """
    print("[humanoid_convert] starting humanoid conversion")

    current_bones = {b.name for b in arm_obj.data.bones}

    rename_map: dict[str, str] = {}
    to_delete: list[str] = []
    weight_merges: list[tuple[str, str]] = []

    for mhr_name, (humanoid_name, target) in BONE_ACTIONS.items():
        if mhr_name not in current_bones:
            continue
        if humanoid_name is not None and target is None:
            rename_map[mhr_name] = humanoid_name
        elif humanoid_name is None and target is not None:
            weight_merges.append((mhr_name, target))
            to_delete.append(mhr_name)
        elif humanoid_name is None and target is None:
            # joint_000: drop without weight transfer
            to_delete.append(mhr_name)
        else:
            raise ValueError(
                f"BONE_ACTIONS[{mhr_name}] cannot both rename and merge"
            )

    unmapped = current_bones - set(BONE_ACTIONS)
    if unmapped:
        # Not fatal — just keep them as-is. Usually means a new MHR joint
        # survived the LBS prune that we haven't accounted for yet.
        print(f"[humanoid_convert] WARNING {len(unmapped)} unmapped bones "
              f"(kept as-is): {sorted(unmapped)}")

    # 1. Transfer vertex-group weights (still Object mode)
    _transfer_vertex_group_weights(mesh_obj, weight_merges)

    # 2. Reshape armature in Edit mode
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        edit_bones = arm_obj.data.edit_bones
        deleted_set = set(to_delete)

        # Re-parent children whose parent is scheduled for deletion. Walk up
        # until we hit a surviving bone (or None = armature root).
        for bname in [b.name for b in edit_bones]:
            if bname in deleted_set:
                continue
            eb = edit_bones.get(bname)
            if eb is None or eb.parent is None:
                continue
            if eb.parent.name in deleted_set:
                new_parent = eb.parent
                while new_parent is not None and new_parent.name in deleted_set:
                    new_parent = new_parent.parent
                eb.parent = new_parent  # None = root

        # Now delete the flagged bones
        for bname in to_delete:
            eb = edit_bones.get(bname)
            if eb is not None:
                edit_bones.remove(eb)

        # Rename survivors to humanoid names
        for mhr_name, humanoid_name in rename_map.items():
            eb = edit_bones.get(mhr_name)
            if eb is not None:
                eb.name = humanoid_name
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')

    # 3. Rename & prune vertex groups to match
    for mhr_name, humanoid_name in rename_map.items():
        vg = mesh_obj.vertex_groups.get(mhr_name)
        if vg is not None:
            vg.name = humanoid_name
    for bname in to_delete:
        vg = mesh_obj.vertex_groups.get(bname)
        if vg is not None:
            mesh_obj.vertex_groups.remove(vg)

    print(f"[humanoid_convert] renamed={len(rename_map)}  "
          f"merged+deleted={len(to_delete)}  "
          f"final_bones={len(arm_obj.data.bones)}")
    return rename_map


def _transfer_vertex_group_weights(mesh_obj, merges):
    """For each ``(src, dst)`` pair, add per-vertex weights from src's
    vertex group into dst's. We don't remove src here; the caller removes
    the whole vertex group after all transfers land."""
    if not merges:
        return
    mesh = mesh_obj.data
    for src_name, dst_name in merges:
        src = mesh_obj.vertex_groups.get(src_name)
        if src is None:
            continue
        dst = mesh_obj.vertex_groups.get(dst_name)
        if dst is None:
            # Unusual, but create one so the weight isn't lost.
            dst = mesh_obj.vertex_groups.new(name=dst_name)

        src_idx = src.index
        moved = 0
        for v in mesh.vertices:
            w = 0.0
            for g in v.groups:
                if g.group == src_idx:
                    w = g.weight
                    break
            if w <= 0.0:
                continue
            dst.add([v.index], w, 'ADD')
            moved += 1
        if moved:
            print(f"[humanoid_convert]   weight-merge {src_name:>12} → "
                  f"{dst_name:<16} ({moved} verts)")


# ---------------------------------------------------------------------------
# Helper for callers: build "humanoid_name → dense MHR index" lookup
# ---------------------------------------------------------------------------


def reorient_bones_along_chain(arm_obj, default_length: float = 0.05) -> None:
    """Re-set each bone's tail to point toward its best "chain child" in the
    CURRENT hierarchy. Call after ``apply_humanoid_conversion`` so that the
    upper/lower arms and other bones extend all the way to their real
    downstream joint (lowerarm, hand, foot, toe, ...) rather than stopping
    at a twist helper that got absorbed anyway.

    The chain child is chosen by picking whichever child is most aligned
    with the parent→self axis. Leaves inherit the parent direction so tips
    (HeadTop_End, Toe_End, finger Thumb4 etc.) hang sensibly off their
    parent instead of snapping to a default axis.
    """
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        edit_bones = arm_obj.data.edit_bones
        children_by_name: dict[str, list[str]] = {}
        for eb in edit_bones:
            if eb.parent is not None:
                children_by_name.setdefault(eb.parent.name, []).append(eb.name)

        for eb in edit_bones:
            head = eb.head.copy()
            kids = children_by_name.get(eb.name, [])
            tail = _pick_tail_position(eb, head, kids, edit_bones, default_length)
            if (tail - head).length < 1e-4:
                tail = head + Vector((0.0, 0.0, default_length))
            eb.tail = tail
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')


def _pick_tail_position(eb, head, kids, edit_bones, default_length):
    if kids:
        parent = eb.parent
        if parent is not None:
            axis = head - parent.head
            if axis.length > 1e-6:
                axis = axis.normalized()
                best, best_s = None, -1e9
                for kname in kids:
                    kb = edit_bones.get(kname)
                    if kb is None:
                        continue
                    d = kb.head - head
                    if d.length < 1e-4:
                        continue
                    s = d.normalized().dot(axis)
                    if s > best_s:
                        best_s, best = s, kb
                if best is not None:
                    return best.head.copy()
        # Root or degenerate axis: pick highest-Z child (spine-ish choice).
        best = max(kids, key=lambda k: edit_bones[k].head.z)
        return edit_bones[best].head.copy()

    # Leaf: continue in parent→self direction.
    parent = eb.parent
    if parent is not None:
        pd = head - parent.head
        if pd.length > 1e-4:
            return head + pd.normalized() * default_length
    return head + Vector((0.0, 0.0, default_length))


def build_humanoid_to_dense(rename_map: dict[str, str],
                            mhr_names: list[str]) -> dict[str, int]:
    """Map humanoid bone names back to their dense MHR index so callers can
    look up ``rest_rots[j]`` / ``posed_rots[j]`` when computing pose deltas
    on the renamed rig."""
    mhr_to_idx = {n: i for i, n in enumerate(mhr_names)}
    out = {}
    for mhr_name, humanoid_name in rename_map.items():
        if mhr_name in mhr_to_idx:
            out[humanoid_name] = mhr_to_idx[mhr_name]
    return out

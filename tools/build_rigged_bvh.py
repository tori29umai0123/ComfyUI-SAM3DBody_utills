import argparse
import json
import math
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Vector
except ImportError:
    print("ERROR: this script must run under Blender")
    sys.exit(1)


def _parse_args():
    idx = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    return parser.parse_args(sys.argv[idx + 1:])


def _pick_chain_child(joint_idx, parent_idx, children, coords):
    if len(children) == 1:
        return children[0]
    if parent_idx >= 0:
        axis = coords[joint_idx] - coords[parent_idx]
        if axis.length > 1e-6:
            axis.normalize()
            best_child, best_score = children[0], -1e9
            for child_idx in children:
                direction = coords[child_idx] - coords[joint_idx]
                if direction.length < 1e-6:
                    continue
                score = direction.normalized().dot(axis)
                if score > best_score:
                    best_score = score
                    best_child = child_idx
            return best_child
    return max(children, key=lambda idx: coords[idx].z)


def mhr_to_blender_vec(v):
    return Vector((float(v[0]), -float(v[2]), float(v[1])))


def main():
    args = _parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output_path = data["output_path"]

    bpy.ops.wm.read_factory_settings(use_empty=True)

    names = data["joint_names"]
    parents = data["joint_parents"]
    posed_coords = [mhr_to_blender_vec(point) for point in data["posed_joint_coords"]]

    children_by_parent = {}
    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx >= 0:
            children_by_parent.setdefault(parent_idx, []).append(joint_idx)

    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    arm_obj = bpy.context.object
    arm_obj.name = "SAM3D_Armature"
    arm_obj.data.name = "SAM3D_ArmatureData"
    for bone in list(arm_obj.data.edit_bones):
        arm_obj.data.edit_bones.remove(bone)

    default_leaf_length = 0.03
    for joint_idx, name in enumerate(names):
        bone = arm_obj.data.edit_bones.new(name)
        bone.head = posed_coords[joint_idx]
        children = children_by_parent.get(joint_idx, [])
        if children:
            child_idx = _pick_chain_child(joint_idx, parents[joint_idx], children, posed_coords)
            bone.tail = posed_coords[child_idx]
        else:
            parent_idx = parents[joint_idx]
            if parent_idx >= 0:
                axis = posed_coords[joint_idx] - posed_coords[parent_idx]
                if axis.length > 1e-6:
                    bone.tail = bone.head + axis.normalized() * default_leaf_length
                else:
                    bone.tail = bone.head + Vector((0, 0, default_leaf_length))
            else:
                bone.tail = bone.head + Vector((0, 0, default_leaf_length))
        if (bone.tail - bone.head).length < 1e-4:
            bone.tail = bone.head + Vector((0, 0, default_leaf_length))

    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx < 0:
            continue
        child_bone = arm_obj.data.edit_bones.get(names[joint_idx])
        parent_bone = arm_obj.data.edit_bones.get(names[parent_idx])
        if child_bone is None or parent_bone is None:
            continue
        child_bone.parent = parent_bone
        if (child_bone.head - parent_bone.tail).length < 1e-4:
            child_bone.use_connect = True

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.rotation_mode = "XYZ"
    arm_obj.rotation_euler = (math.radians(-90), 0, 0)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_anim.bvh(
        filepath=str(output_path),
        check_existing=False,
        global_scale=1.0,
        frame_start=1,
        frame_end=1,
        rotate_mode="NATIVE",
        root_transform_only=False,
    )
    print(f"[build_rigged_bvh] Wrote {output_path}")


if __name__ == "__main__":
    main()

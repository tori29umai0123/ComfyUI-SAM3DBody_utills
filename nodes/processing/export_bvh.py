"""SAM 3D Body BVH export nodes.

Provides:
* SAM 3D Body: Export Posed BVH
* SAM 3D Body: Export Animated BVH

Both nodes accept the same core inputs:
    model, character_json, pose_json, blender_exe, output_filename

`pose_json` can be either a single pose object or an animation payload:
* single pose:
    {"body_pose_params": [...], "hand_pose_params": [...], "global_rot": [...]}
* animation:
    {"frames": [<pose>, <pose>, ...], "fps": 30}
  or simply [<pose>, <pose>, ...]
"""

import json
import os
import subprocess
import tempfile
import time

import numpy as np
import torch

from .process import (
    _FACE_BS_CACHE,
    _apply_face_blendshapes,
    _apply_bone_length_scales,
    _get_mhr_rest_verts,
    _load_sam3d_model,
    _to_batched_tensor,
    apply_pose_lean_correction_rig,
)
from .export_animated import (
    _comfy_frame_to_bgr,
    _mask_frame_bbox,
)
from .export_rigged import (
    _DEFAULT_BLENDER,
    _KNOWN_JOINT_NAMES,
    _SHAPE_SLIDER_NORM,
    _SHAPE_SLIDER_SIGN,
    _scale_skeleton_rest,
    _unpack_batched,
)


_UTILS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BUILD_RIGGED_BVH_SCRIPT = os.path.join(_UTILS_ROOT, "tools", "build_rigged_bvh.py")
_BUILD_ANIMATED_BVH_SCRIPT = os.path.join(_UTILS_ROOT, "tools", "build_animated_bvh.py")


_HUMANOID_MAP = {
    "pelvis": "Hips",
    "joint_034": "Spine",
    "spine_01": "Chest",
    "spine_02": "UpperChest",
    "neck_01": "Neck",
    "head": "Head",
    "clavicle_l": "LeftShoulder",
    "upperarm_l": "LeftUpperArm",
    "lowerarm_l": "LeftLowerArm",
    "hand_l": "LeftHand",
    "clavicle_r": "RightShoulder",
    "upperarm_r": "RightUpperArm",
    "lowerarm_r": "RightLowerArm",
    "hand_r": "RightHand",
    "thigh_l": "LeftUpperLeg",
    "calf_l": "LeftLowerLeg",
    "foot_l": "LeftFoot",
    "thigh_r": "RightUpperLeg",
    "calf_r": "RightLowerLeg",
    "foot_r": "RightFoot",
    "joint_060": "RightThumbProximal",
    "joint_061": "RightThumbIntermediate",
    "joint_062": "RightThumbDistal",
    "joint_056": "RightIndexProximal",
    "joint_057": "RightIndexIntermediate",
    "joint_058": "RightIndexDistal",
    "joint_052": "RightMiddleProximal",
    "joint_053": "RightMiddleIntermediate",
    "joint_054": "RightMiddleDistal",
    "joint_048": "RightRingProximal",
    "joint_049": "RightRingIntermediate",
    "joint_050": "RightRingDistal",
    "joint_044": "RightLittleProximal",
    "joint_045": "RightLittleIntermediate",
    "joint_046": "RightLittleDistal",
    "joint_096": "LeftThumbProximal",
    "joint_097": "LeftThumbIntermediate",
    "joint_098": "LeftThumbDistal",
    "joint_092": "LeftIndexProximal",
    "joint_093": "LeftIndexIntermediate",
    "joint_094": "LeftIndexDistal",
    "joint_088": "LeftMiddleProximal",
    "joint_089": "LeftMiddleIntermediate",
    "joint_090": "LeftMiddleDistal",
    "joint_084": "LeftRingProximal",
    "joint_085": "LeftRingIntermediate",
    "joint_086": "LeftRingDistal",
    "joint_080": "LeftLittleProximal",
    "joint_081": "LeftLittleIntermediate",
    "joint_082": "LeftLittleDistal",
}


def _parse_json_or_empty(raw_value, label):
    try:
        return json.loads(raw_value) if raw_value.strip() else {}
    except Exception as exc:
        print(f"[SAM3DBody] {label} parse failed: {exc}; using empty payload")
        return {}


def _subset_humanoid(package):
    names = list(package["joint_names"])
    parents = list(package["joint_parents"])
    kept_indices = [i for i, name in enumerate(names) if name in _HUMANOID_MAP]
    if not kept_indices:
        raise RuntimeError("BVH export: no humanoid-compatible joints found.")

    old_to_new = {old: new for new, old in enumerate(kept_indices)}
    new_parents = []
    for old_i in kept_indices:
        parent_i = parents[old_i]
        while parent_i >= 0 and parent_i not in old_to_new:
            parent_i = parents[parent_i]
        new_parents.append(old_to_new[parent_i] if parent_i >= 0 else -1)

    def _subset(arr):
        return [arr[i] for i in kept_indices]

    new_package = {
        "joint_names": [_HUMANOID_MAP[names[i]] for i in kept_indices],
        "joint_parents": new_parents,
        "rest_joint_coords": _subset(package["rest_joint_coords"]),
        "rest_joint_rots": _subset(package["rest_joint_rots"]),
    }
    if "posed_joint_coords" in package:
        new_package["posed_joint_coords"] = _subset(package["posed_joint_coords"])
    if "posed_joint_rots" in package:
        new_package["posed_joint_rots"] = _subset(package["posed_joint_rots"])
    if "frames_posed_joint_rots" in package:
        new_package["frames_posed_joint_rots"] = [
            [frame[i] for i in kept_indices]
            for frame in package["frames_posed_joint_rots"]
        ]
    if "frames_root_trans" in package:
        new_package["frames_root_trans"] = package["frames_root_trans"]
    if "fps" in package:
        new_package["fps"] = package["fps"]
    return new_package


def _build_character_rest(model, preset):
    loaded = _load_sam3d_model(model)
    sam_3d_model = loaded["model"]
    device = torch.device(loaded["device"])
    mhr_head = sam_3d_model.head_pose

    _get_mhr_rest_verts(mhr_head, device)
    parents = _FACE_BS_CACHE["joint_parents"].astype(np.int32)
    num_joints = parents.shape[0]

    bp = preset.get("body_params", {}) if isinstance(preset, dict) else {}
    bl = preset.get("bone_lengths", {}) if isinstance(preset, dict) else {}
    bs = preset.get("blendshapes", {}) if isinstance(preset, dict) else {}

    body_shape_ui = [
        float(bp.get("fat", 0.0)),
        float(bp.get("muscle", 0.0)),
        float(bp.get("fat_muscle", 0.0)),
        float(bp.get("limb_girth", 0.0)),
        float(bp.get("limb_muscle", 0.0)),
        float(bp.get("limb_fat", 0.0)),
        float(bp.get("chest_shoulder", 0.0)),
        float(bp.get("waist_hip", 0.0)),
        float(bp.get("thigh_calf", 0.0)),
    ]
    shape_params = torch.zeros((1, mhr_head.num_shape_comps), dtype=torch.float32, device=device)
    for i in range(min(9, mhr_head.num_shape_comps)):
        shape_params[0, i] = body_shape_ui[i] * _SHAPE_SLIDER_NORM[i] * _SHAPE_SLIDER_SIGN[i]
    scale_params = torch.zeros((1, mhr_head.num_scale_comps), dtype=torch.float32, device=device)
    expr_params = torch.zeros((1, mhr_head.num_face_comps), dtype=torch.float32, device=device)

    zeros3 = torch.zeros((1, 3), dtype=torch.float32, device=device)
    body_zero = torch.zeros((1, 133), dtype=torch.float32, device=device)
    hand_zero = torch.zeros((1, 108), dtype=torch.float32, device=device)
    global_trans = torch.zeros((1, 3), dtype=torch.float32, device=device)

    with torch.no_grad():
        rest_out = mhr_head.mhr_forward(
            global_trans=global_trans,
            global_rot=zeros3,
            body_pose_params=body_zero,
            hand_pose_params=hand_zero,
            scale_params=scale_params,
            shape_params=shape_params,
            expr_params=expr_params,
            return_joint_rotations=True,
            return_joint_coords=True,
        )
    char_rest_verts = rest_out[0].detach().cpu().numpy().astype(np.float32)
    if char_rest_verts.ndim == 3:
        char_rest_verts = char_rest_verts[0]
    char_rest_rots, char_rest_coords = _unpack_batched(rest_out[1:])
    char_rest_rots = char_rest_rots.astype(np.float32)
    char_rest_coords = char_rest_coords.astype(np.float32)

    bs_sliders = {str(k): float(v) for k, v in bs.items()}
    if any(v != 0.0 for v in bs_sliders.values()):
        from ..preset_pack import active_pack_dir as _pack_dir
        presets_dir = str(_pack_dir())
        bs_npz = os.path.join(presets_dir, "face_blendshapes.npz")
        mhr_neutral_rest = _FACE_BS_CACHE["rest_verts"]
        char_rest_verts = _apply_face_blendshapes(
            char_rest_verts,
            mhr_neutral_rest,
            bs_sliders,
            char_rest_rots,
            presets_dir,
            bs_npz,
        )

    bone_scales = {
        "torso": float(bl.get("torso", 1.0)),
        "neck": float(bl.get("neck", 1.0)),
        "arm": float(bl.get("arm", 1.0)),
        "leg": float(bl.get("leg", 1.0)),
    }
    any_bone_scaled = any(v != 1.0 for v in bone_scales.values())
    if any_bone_scaled:
        char_rest_verts = _apply_bone_length_scales(
            char_rest_verts,
            arm_scale=bone_scales["arm"],
            leg_scale=bone_scales["leg"],
            torso_scale=bone_scales["torso"],
            neck_scale=bone_scales["neck"],
            joint_rots_posed=char_rest_rots,
        )
        char_rest_coords = _scale_skeleton_rest(
            char_rest_coords,
            parents,
            _FACE_BS_CACHE["joint_chain_cats"],
            bone_scales,
        )

    names_full = [_KNOWN_JOINT_NAMES.get(i, f"joint_{i:03d}") for i in range(num_joints)]
    return {
        "loaded": loaded,
        "mhr_head": mhr_head,
        "device": device,
        "parents": parents,
        "names_full": names_full,
        "scale_params": scale_params,
        "shape_params": shape_params,
        "expr_params": expr_params,
        "bone_scales": bone_scales,
        "any_bone_scaled": any_bone_scaled,
        "rest_joint_coords": char_rest_coords,
        "rest_joint_rots": char_rest_rots,
    }


def _extract_pose_frames(payload):
    fps = 30.0
    if isinstance(payload, list):
        frames = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("frames"), list):
            frames = payload["frames"]
            fps = float(payload.get("fps", 30.0))
        elif isinstance(payload.get("poses"), list):
            frames = payload["poses"]
            fps = float(payload.get("fps", 30.0))
        else:
            frames = [payload]
    else:
        frames = [{}]
    if not frames:
        frames = [{}]
    return frames, fps


def _frame_root_translation(frame_payload):
    for key in ("frames_root_trans", "root_trans", "global_trans", "pred_cam_t", "camera"):
        value = frame_payload.get(key)
        if value is None:
            continue
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if arr.size >= 3:
            return arr[:3].tolist()
    return [0.0, 0.0, 0.0]


def _run_blender_export(package, build_script, blender_exe, timeout_sec):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(package, tmp)
        tmp_json = tmp.name

    cmd = [blender_exe, "--background", "--python", build_script, "--", "--input", tmp_json]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        if result.returncode != 0:
            print("[SAM3DBody] Blender stdout:\n" + (result.stdout or ""))
            print("[SAM3DBody] Blender stderr:\n" + (result.stderr or ""))
            raise RuntimeError(f"Blender BVH export failed (exit code {result.returncode}).")
        if result.stdout:
            print(result.stdout[-1200:])
    finally:
        if os.path.exists(tmp_json):
            try:
                os.unlink(tmp_json)
            except Exception:
                pass


def _ui_result(output_path):
    return {
        "ui": {
            "text": [output_path],
        },
        "result": (output_path,),
    }


class SAM3DBodyExportPosedBVH:
    @classmethod
    def INPUT_TYPES(cls):
        character_placeholder = (
            "{\n"
            '  "_slot": "=== CHARACTER JSON ===",\n'
            '  "_hint": "Paste Render node\'s settings_json output here.",\n'
            '  "body_params": {},\n'
            '  "bone_lengths": {},\n'
            '  "blendshapes": {}\n'
            "}"
        )
        pose_placeholder = (
            "{\n"
            '  "_slot": "=== POSE JSON ===",\n'
            '  "_hint": "Paste Process Image to Pose JSON output here.",\n'
            '  "body_pose_params": null,\n'
            '  "hand_pose_params": null,\n'
            '  "global_rot": null\n'
            "}"
        )
        return {
            "required": {
                "model": ("SAM3D_MODEL",),
                "character_json": ("STRING", {"multiline": True, "default": character_placeholder}),
                "pose_json": ("STRING", {"multiline": True, "default": pose_placeholder}),
                "blender_exe": ("STRING", {"default": _DEFAULT_BLENDER}),
                "output_filename": ("STRING", {"default": "sam3d_posed.bvh"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("bvh_path",)
    FUNCTION = "export"
    CATEGORY = "SAM3DBody/export"
    OUTPUT_NODE = True

    def export(self, model, character_json, pose_json, blender_exe, output_filename):
        import folder_paths

        preset = _parse_json_or_empty(character_json, "character_json")
        pose_payload = _parse_json_or_empty(pose_json, "pose_json")
        if isinstance(pose_payload, dict) and isinstance(pose_payload.get("frames"), list):
            pose_payload = pose_payload["frames"][0] if pose_payload["frames"] else {}
        elif isinstance(pose_payload, list):
            pose_payload = pose_payload[0] if pose_payload else {}

        state = _build_character_rest(model, preset)
        mhr_head = state["mhr_head"]
        device = state["device"]
        parents = state["parents"]
        char_rest_coords = state["rest_joint_coords"]
        char_rest_rots = state["rest_joint_rots"]
        names_full = state["names_full"]
        scale_params = state["scale_params"]
        shape_params = state["shape_params"]
        expr_params = state["expr_params"]
        bone_scales = state["bone_scales"]
        any_bone_scaled = state["any_bone_scaled"]

        corrected_pose = preset.get("corrected_pose_json") if isinstance(preset, dict) else None
        if (
            isinstance(corrected_pose, dict)
            and isinstance(corrected_pose.get("posed_joint_rots"), list)
            and isinstance(corrected_pose.get("posed_joint_coords"), list)
        ):
            try:
                posed_rots = np.asarray(corrected_pose["posed_joint_rots"], dtype=np.float32)
                posed_coords = np.asarray(corrected_pose["posed_joint_coords"], dtype=np.float32)
            except Exception:
                posed_rots = None
                posed_coords = None
            if (
                posed_rots is not None
                and posed_coords is not None
                and posed_rots.shape == char_rest_rots.shape
                and posed_coords.shape == char_rest_coords.shape
            ):
                if not output_filename or not output_filename.strip():
                    output_filename = f"sam3d_posed_{int(time.time())}.bvh"
                if not output_filename.lower().endswith(".bvh"):
                    output_filename += ".bvh"
                output_path = os.path.abspath(os.path.join(folder_paths.get_output_directory(), output_filename))

                package = _subset_humanoid({
                    "output_path": output_path,
                    "joint_parents": parents.tolist(),
                    "joint_names": names_full,
                    "rest_joint_coords": char_rest_coords.tolist(),
                    "rest_joint_rots": char_rest_rots.tolist(),
                    "posed_joint_coords": posed_coords.tolist(),
                    "posed_joint_rots": posed_rots.tolist(),
                })
                package["output_path"] = output_path

                _run_blender_export(package, _BUILD_RIGGED_BVH_SCRIPT, blender_exe, timeout_sec=600)
                if not os.path.exists(output_path):
                    raise RuntimeError(f"Blender reported success but {output_path} was not created.")
                print(f"[SAM3DBody] Posed BVH: {output_path}")
                return _ui_result(output_path)

        global_trans = torch.zeros((1, 3), dtype=torch.float32, device=device)
        global_rot_t = _to_batched_tensor(pose_payload.get("global_rot"), device, width=3)
        body_pose_t = _to_batched_tensor(pose_payload.get("body_pose_params"), device, width=133)
        hand_pose_t = _to_batched_tensor(pose_payload.get("hand_pose_params"), device, width=108)

        with torch.no_grad():
            posed_out = mhr_head.mhr_forward(
                global_trans=global_trans,
                global_rot=global_rot_t,
                body_pose_params=body_pose_t,
                hand_pose_params=hand_pose_t,
                scale_params=scale_params,
                shape_params=shape_params,
                expr_params=expr_params,
                return_joint_rotations=True,
                return_joint_coords=True,
            )
        posed_rots, posed_coords = _unpack_batched(posed_out[1:])
        posed_rots = posed_rots.astype(np.float32)
        posed_coords = posed_coords.astype(np.float32)
        if any_bone_scaled:
            posed_coords = _scale_skeleton_rest(
                posed_coords,
                parents,
                _FACE_BS_CACHE["joint_chain_cats"],
                bone_scales,
            )

        if not output_filename or not output_filename.strip():
            output_filename = f"sam3d_posed_{int(time.time())}.bvh"
        if not output_filename.lower().endswith(".bvh"):
            output_filename += ".bvh"
        output_path = os.path.abspath(os.path.join(folder_paths.get_output_directory(), output_filename))

        package = _subset_humanoid({
            "output_path": output_path,
            "joint_parents": parents.tolist(),
            "joint_names": names_full,
            "rest_joint_coords": char_rest_coords.tolist(),
            "rest_joint_rots": char_rest_rots.tolist(),
            "posed_joint_coords": posed_coords.tolist(),
            "posed_joint_rots": posed_rots.tolist(),
        })
        package["output_path"] = output_path

        _run_blender_export(package, _BUILD_RIGGED_BVH_SCRIPT, blender_exe, timeout_sec=600)
        if not os.path.exists(output_path):
            raise RuntimeError(f"Blender reported success but {output_path} was not created.")
        print(f"[SAM3DBody] Posed BVH: {output_path}")
        return _ui_result(output_path)


class SAM3DBodyExportAnimatedBVH:
    @classmethod
    def INPUT_TYPES(cls):
        character_placeholder = (
            "{\n"
            '  "_slot": "=== CHARACTER JSON ===",\n'
            '  "_hint": "Paste Render node\'s settings_json output here.",\n'
            '  "body_params": {},\n'
            '  "bone_lengths": {},\n'
            '  "blendshapes": {}\n'
            "}"
        )
        motion_placeholder = (
            "{\n"
            '  "_slot": "=== MOTION JSON ===",\n'
            '  "_hint": "Single pose or { \\"frames\\": [pose, ...], \\"fps\\": 30 }",\n'
            '  "frames": []\n'
            "}"
        )
        return {
            "required": {
                "model": ("SAM3D_MODEL",),
                "images": ("IMAGE",),
                "character_json": ("STRING", {"multiline": True, "default": character_placeholder}),
                "pose_adjust": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 1.0}),
                "bbox_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "inference_type": (["full", "body", "hand"], {"default": "full"}),
                "blender_exe": ("STRING", {"default": _DEFAULT_BLENDER}),
                "output_filename": ("STRING", {"default": "sam3d_animated.bvh"}),
            },
            "optional": {
                "masks": ("MASK",),
                "pose_json": ("STRING", {"multiline": True, "default": motion_placeholder}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("bvh_path",)
    FUNCTION = "export"
    CATEGORY = "SAM3DBody/export"
    OUTPUT_NODE = True

    def export(
        self,
        model,
        images,
        character_json,
        pose_adjust,
        fps,
        bbox_threshold,
        inference_type,
        blender_exe,
        output_filename,
        masks=None,
        pose_json="",
    ):
        import folder_paths
        from ..sam_3d_body import SAM3DBodyEstimator

        preset = _parse_json_or_empty(character_json, "character_json")
        motion_payload = _parse_json_or_empty(pose_json, "pose_json")

        state = _build_character_rest(model, preset)
        sam_3d_model = state["loaded"]["model"]
        mhr_head = state["mhr_head"]
        device = state["device"]
        parents = state["parents"]
        char_rest_coords = state["rest_joint_coords"]
        char_rest_rots = state["rest_joint_rots"]
        names_full = state["names_full"]
        scale_params = state["scale_params"]
        shape_params = state["shape_params"]
        expr_params = state["expr_params"]
        try:
            lean_strength = float(pose_adjust)
        except (TypeError, ValueError):
            lean_strength = 0.0

        frames_posed_rots = []
        frames_root_trans = []
        num_frames = int(images.shape[0]) if images is not None else 0

        if num_frames > 0:
            estimator = SAM3DBodyEstimator(
                sam_3d_body_model=sam_3d_model,
                model_cfg=state["loaded"]["model_cfg"],
                human_detector=None,
                human_segmentor=None,
                fov_estimator=None,
            )

            masks_np = None
            if masks is not None:
                masks_np = masks.detach().cpu().numpy()
                if masks_np.ndim == 2:
                    masks_np = masks_np[None, ...]
                if masks_np.shape[0] != num_frames:
                    print(
                        f"[SAM3DBody] mask count ({masks_np.shape[0]}) != "
                        f"frame count ({num_frames}); ignoring masks"
                    )
                    masks_np = None

            last_good_pose = None
            for i in range(num_frames):
                img_bgr = _comfy_frame_to_bgr(images, i)
                mask_np = None
                bboxes = None
                if masks_np is not None:
                    mask_np = masks_np[i]
                    bboxes = _mask_frame_bbox(mask_np)

                outputs = estimator.process_one_image(
                    img_bgr,
                    bboxes=bboxes,
                    masks=mask_np,
                    bbox_thr=bbox_threshold,
                    use_mask=(mask_np is not None),
                    inference_type=inference_type,
                )

                if not outputs:
                    posed_rots = last_good_pose if last_good_pose is not None else char_rest_rots.copy()
                    frames_posed_rots.append(np.asarray(posed_rots, dtype=np.float32))
                    frames_root_trans.append([0.0, 0.0, 0.0])
                    continue

                raw = outputs[0]
                global_trans = torch.zeros((1, 3), dtype=torch.float32, device=device)
                global_rot_t = _to_batched_tensor(raw.get("global_rot"), device, width=3)
                body_pose_t = _to_batched_tensor(raw.get("body_pose_params"), device, width=133)
                hand_pose_t = _to_batched_tensor(raw.get("hand_pose_params"), device, width=108)
                with torch.no_grad():
                    posed_out = mhr_head.mhr_forward(
                        global_trans=global_trans,
                        global_rot=global_rot_t,
                        body_pose_params=body_pose_t,
                        hand_pose_params=hand_pose_t,
                        scale_params=scale_params,
                        shape_params=shape_params,
                        expr_params=expr_params,
                        return_joint_rotations=True,
                        return_joint_coords=True,
                    )
                posed_rots, posed_coords = _unpack_batched(posed_out[1:])
                posed_rots = posed_rots.astype(np.float32)
                if lean_strength > 1e-6 and posed_coords is not None:
                    posed_rots, _ = apply_pose_lean_correction_rig(
                        posed_rots,
                        np.asarray(posed_coords, dtype=np.float32),
                        parents,
                        lean_strength,
                    )
                last_good_pose = posed_rots
                frames_posed_rots.append(posed_rots)
                frames_root_trans.append(_frame_root_translation(raw))
                if (i + 1) % 20 == 0 or i == num_frames - 1:
                    print(f"[SAM3DBody] BVH motion frames processed: {i + 1}/{num_frames}")
        else:
            frames, fps = _extract_pose_frames(motion_payload)
            for i, frame_payload in enumerate(frames):
                if not isinstance(frame_payload, dict):
                    frame_payload = {}
                global_trans = torch.zeros((1, 3), dtype=torch.float32, device=device)
                global_rot_t = _to_batched_tensor(frame_payload.get("global_rot"), device, width=3)
                body_pose_t = _to_batched_tensor(frame_payload.get("body_pose_params"), device, width=133)
                hand_pose_t = _to_batched_tensor(frame_payload.get("hand_pose_params"), device, width=108)
                with torch.no_grad():
                    posed_out = mhr_head.mhr_forward(
                        global_trans=global_trans,
                        global_rot=global_rot_t,
                        body_pose_params=body_pose_t,
                        hand_pose_params=hand_pose_t,
                        scale_params=scale_params,
                        shape_params=shape_params,
                        expr_params=expr_params,
                        return_joint_rotations=True,
                        return_joint_coords=True,
                    )
                posed_rots, posed_coords = _unpack_batched(posed_out[1:])
                posed_rots = posed_rots.astype(np.float32)
                if lean_strength > 1e-6 and posed_coords is not None:
                    posed_rots, _ = apply_pose_lean_correction_rig(
                        posed_rots,
                        np.asarray(posed_coords, dtype=np.float32),
                        parents,
                        lean_strength,
                    )
                frames_posed_rots.append(posed_rots)
                frames_root_trans.append(_frame_root_translation(frame_payload))
                if (i + 1) % 20 == 0 or i == len(frames) - 1:
                    print(f"[SAM3DBody] BVH motion frames processed: {i + 1}/{len(frames)}")

        if not output_filename or not output_filename.strip():
            output_filename = f"sam3d_animated_{int(time.time())}.bvh"
        if not output_filename.lower().endswith(".bvh"):
            output_filename += ".bvh"
        output_path = os.path.abspath(os.path.join(folder_paths.get_output_directory(), output_filename))

        package = _subset_humanoid({
            "output_path": output_path,
            "fps": float(fps),
            "joint_parents": parents.tolist(),
            "joint_names": names_full,
            "rest_joint_coords": char_rest_coords.tolist(),
            "rest_joint_rots": char_rest_rots.tolist(),
            "frames_posed_joint_rots": [frame.tolist() for frame in frames_posed_rots],
            "frames_root_trans": frames_root_trans,
        })
        package["output_path"] = output_path

        timeout_s = max(600, 30 + 2 * len(frames_posed_rots))
        _run_blender_export(package, _BUILD_ANIMATED_BVH_SCRIPT, blender_exe, timeout_sec=timeout_s)
        if not os.path.exists(output_path):
            raise RuntimeError(f"Blender reported success but {output_path} was not created.")
        print(f"[SAM3DBody] Animated BVH: {output_path}")
        return _ui_result(output_path)


NODE_CLASS_MAPPINGS = {
    "SAM3DBodyExportPosedBVH": SAM3DBodyExportPosedBVH,
    "SAM3DBodyExportAnimatedBVH": SAM3DBodyExportAnimatedBVH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyExportPosedBVH": "SAM 3D Body: Export Posed BVH",
    "SAM3DBodyExportAnimatedBVH": "SAM 3D Body: Export Animated BVH",
}

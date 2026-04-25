# ComfyUI-SAM3DBody_utills

**Language:** [🇯🇵 日本語](README.md) ・ 🇬🇧 English (current)

A streamlined fork of **[PozzettiAndrea/ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody)** focused on **building a rigged 3D character from a single image** inside ComfyUI.

## ✏ Turn clumsy doodles into a pose reference

![sample4](docs/sample4.png)

**From left to right:** ① the character you want to finish ・ ② **a quick hand-drawn rough** ・ ③ the 3D character body this plugin renders (the rough's pose mapped onto your character's body shape) ・ ④ ③ run through an image-edit model for the final output.

You don't need to draw well — as long as the pose reads, this plugin can convert a **stick-figure doodle into a pose reference with correct proportions and 3D consistency**. Hand ③ to img2img / Qwen-Image-Edit and you can carry it all the way to a finished frame like ④ in one shot.

## What this plugin does

### 1. Render a custom-body 3D model in the pose of any input image

Extract the pose from any character image, then render it onto a 3D character body whose shape you control yourself (height, thickness, face shape, bone lengths, ...).

![sample1](docs/sample1.png)

### 2. Export a rigged, animated FBX / BVH

Write the same posed character out as an FBX containing **armature + skinned mesh + pose animation** to `<ComfyUI>/output/`. The FBX opens directly in Blender / Unity / Unreal Engine; the BVH variant is confirmed to load in CLIP STUDIO PAINT.

![sample2](docs/sample2.png)

### 3. Export motion-captured FBX / BVH from a video

Feed a video via `ComfyUI-VideoHelperSuite`'s `VHS_LoadVideo` and the plugin runs SAM 3D Body on every frame, then bakes the **continuous motion onto a character rigged at the rest pose**, writing it to `<ComfyUI>/output/` as an animated FBX. Drop it into Unity and you have a ready-to-play motion clip in Animator / Timeline. The BVH variant is confirmed to load in CLIP STUDIO PAINT.

<!-- DEMO_VIDEO_START -->
https://github.com/user-attachments/assets/4fa43a56-8dd2-4ebf-8abe-61a31ff14e6f
<!-- DEMO_VIDEO_END -->

## Included nodes (10 total)

### Model / inference
1. **Load SAM 3D Body Model** — lazy-loads the checkpoint from `<ComfyUI>/models/sam3dbody/`.
2. **SAM 3D Body: Process Image to Pose JSON** — runs SAM 3D Body on the input image and emits the pose as JSON.
   - When the `mask` input is left unconnected, falls back to internal BiRefNet Lite auto-masking
   - For higher tracking accuracy on tricky footage, connect an explicit `MASK` node

### Character authoring / rendering
3. **SAM 3D Body: Setting Chara JSON** — bundles the preset selector + body / bone / blendshape sliders and emits `chara_json` (STRING). Split out from the old render node so character state can be authored once and reused.
4. **SAM 3D Body: Render Human From Pose And Chara JSON** — takes `pose_json` + `chara_json` as inputs; the only widget-driven controls left are camera (`offset_x/y` / `scale_offset` / `camera_yaw/pitch_deg` / `width` / `height`) and lean correction (`pose_adjust`).

### Web-UI editors (in-page browser modals, JA/EN switchable)
5. **SAM 3D Body: Pose Editor** — click "Open Pose Editor" to launch. Upload an image → segmentation + pose inference → fine-tune bones in the 3D viewport → confirm to emit `pose_json` (STRING) plus `width` / `height` (INT).
6. **SAM 3D Body: Character Editor** — click "Open Character Editor" to launch. Sculpt the body with sliders + 3D preview; confirm to emit `chara_json` (STRING).

### FBX / BVH export (all Blender-required)
7. **SAM 3D Body: Export Rigged FBX** — writes an armature + skinned mesh + posed animation FBX to `<ComfyUI>/output/`.
8. **SAM 3D Body: Export Animated FBX** — writes an animated FBX baked from a video (IMAGE batch). The character is rigged once at the rest pose, then every frame becomes a keyframe.
9. **SAM 3D Body: Export Posed BVH** — writes a single-pose humanoid-compatible BVH to `<ComfyUI>/output/`.
10. **SAM 3D Body: Export Animated BVH** — writes an animated BVH from a video (IMAGE batch) or a supplied pose array.

Based on Meta's **SAM 3D Body** + **Momentum Human Rig (MHR)**; both libraries are vendored under their original licenses. See the [License](#license) section.

## Installation

### Prerequisites

- ComfyUI already installed (Python 3.11 recommended)
- Windows / Linux / macOS (tested on Windows 11 + Python 3.11)
- **Blender 4.1 or newer** (required by some features — see below)
- NVIDIA GPU (CUDA) strongly recommended — see VRAM requirements below

### GPU / VRAM requirements

SAM 3D Body keeps ~3.4 GB of weights resident on the GPU; inference activations stay under 100 MB, so this is a relatively lightweight model. Measurements taken on an RTX A6000 (1024×1024 input, single-person scene):

| Metric | `full` inference | `body` inference |
|---|---:|---:|
| Model weights (resident) | 3,374 MB | 3,374 MB |
| Inference activation (extra) | +93 MB | +69 MB |
| **Overall peak VRAM** | **~3.5 GB** | ~3.5 GB |
| First inference time | 3.06 s | 1.47 s |
| Warm inference time | 1.13 s | 0.27 s |

**Recommended specs:**

| Use case | Recommended VRAM | Examples |
|---|---|---|
| **Absolute minimum** (this plugin alone) | **4 GB** | GTX 1650 |
| **Comfortable** (alongside the OS and other processes) | **6 GB** | RTX 3050 / 4060 |
| **Sharing ComfyUI with other models** (SDXL / Qwen-Image-Edit etc.) | **8 GB+** | RTX 3060 12GB / 4070 (depends on the other model) |

> The model is cached at module scope (`_MODEL_CACHE`), so re-running the node does not allocate additional VRAM. CPU inference works but is much slower; CUDA is strongly recommended.

### Features that require Blender ⚠

The following features need **[Blender](https://www.blender.org/) installed**. Tested with Blender 4.1 (`C:/Program Files/Blender Foundation/Blender 4.1/blender.exe`).

| Feature | Why Blender is needed |
|---|---|
| **Adding / editing blend shapes** | Edit shape keys on `tools/bone_backup/all_parts_bs.fbx` in the Blender GUI, then re-run `tools/extract_face_blendshapes.py` headless to rebuild `presets/face_blendshapes.npz` |
| **`SAM 3D Body: Export Rigged FBX` node** | Calls `blender.exe --background --python tools/build_rigged_fbx.py` internally to assemble the armature / mesh / LBS weights / posed animation and export the FBX |
| **`SAM 3D Body: Export Animated FBX` node** | Calls `blender.exe --background --python tools/build_animated_fbx.py` internally to bake per-frame rotation keyframes from the input video and export the animated FBX |
| **`SAM 3D Body: Export Posed BVH` node** | Calls `blender.exe --background --python tools/build_rigged_bvh.py` internally to export a single-pose BVH |
| **`SAM 3D Body: Export Animated BVH` node** | Calls `blender.exe --background --python tools/build_animated_bvh.py` internally to export a BVH from a video or pose array |

Blender downloads:

- Win/x64: https://download.blender.org/release/Blender4.1/blender-4.1.1-windows-x64.zip
- Linux/x64: https://download.blender.org/release/Blender4.1/blender-4.1.1-linux-x64.tar.xz
- Linux/ARM: https://github.com/tori29umai0123/SAM3DBody_utills/releases/download/blender-arm64-v1.0/ARM_blender41-portable.tar.xz

### Standard installation (recommended)

1. Place this repo under `C:\ComfyUI\custom_nodes\` (right under ComfyUI's `custom_nodes/`)
2. From the ComfyUI Python environment run:

   ```
   cd C:/ComfyUI/custom_nodes/ComfyUI-SAM3DBody_utills
   C:/ComfyUI/.venv/Scripts/python.exe -m pip install -r requirements.txt
   C:/ComfyUI/.venv/Scripts/python.exe install.py
   ```

3. Launch ComfyUI — on first run the SAM 3D Body weights (~1.5 GB) auto-download from `jetjodh/sam-3d-body-dinov3` into `<ComfyUI>/models/sam3dbody/`.

### Manual installation steps

For when the auto-downloader fails or you want finer control over the environment.

#### 1. Install bootstrap dependencies

```
C:/ComfyUI/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

This puts the lightweight bootstrap deps (`comfy-env`, `comfy-3d-viewers`, `numpy`, `pillow`, `opencv-python-headless`, …) into the main venv.

#### 2. Set up the isolated environment (heavy dependencies)

```
C:/ComfyUI/.venv/Scripts/python.exe install.py
```

## License

This project uses a **multi-license structure**. The "what is under which license" split is the following three regions. Full license texts live in `docs/licenses/`.

| Region | Files / assets | License | Copyright |
|---|---|---|---|
| **Our wrapper code** (ComfyUI integration) | `nodes/` (excluding `nodes/sam_3d_body/`), `tools/`, `web/`, `install.py`, `__init__.py`, `prestartup_script.py`, and other ComfyUI integration code | **MIT License** <br>([LICENSE-MIT](docs/licenses/LICENSE-MIT)) | Copyright (c) 2025 Andrea Pozzetti (upstream) |
| **SAM part** (Meta SAM 3D Body itself) | Vendored library under `nodes/sam_3d_body/` (model + inference code) | **SAM License** <br>([LICENSE-SAM](docs/licenses/LICENSE-SAM)) | Copyright (c) Meta Platforms, Inc. and affiliates |
| **MHR part** (Momentum Human Rig + derived data) | `mhr_model.pt` asset, and data authored against MHR topology: `presets/face_blendshapes.npz`, per-object vertex JSONs in `presets/`, `tools/bone_backup/all_parts_bs.fbx` | **Apache License 2.0** <br>([LICENSE-MHR](docs/licenses/LICENSE-MHR) / [NOTICE-MHR](docs/licenses/NOTICE-MHR)) | Copyright (c) Meta Platforms, Inc. and affiliates |

> Notes: **This fork's own changes are added under the same MIT** — the MIT copyright line still credits upstream (Andrea Pozzetti). For MHR, Apache 2.0 covers not just the bundled asset but **any data we authored against MHR's mesh topology** (blend-shape deltas, per-object vertex JSONs), which are derivative works of MHR.

See [LICENSE](LICENSE) for the top-level summary and [THIRD_PARTY_NOTICES](docs/licenses/THIRD_PARTY_NOTICES) for third-party attributions.

### Using This Project

- ✅ Wrapper code is free to use, modify, and redistribute under MIT
- ✅ SAM 3D Body is usable for research and commercial purposes under the SAM License
- ✅ MHR (and any blend-shape / region data derived from its topology) is usable commercially under Apache 2.0
- ⚠️ When redistributing, ship **LICENSE-MIT + LICENSE-SAM + LICENSE-MHR + NOTICE-MHR**
- ⚠️ Acknowledge SAM 3D Body in publications (required by the SAM License)

## Community

For issues and feature requests specific to this fork, use the [tori29umai0123/ComfyUI-SAM3DBody_utills](https://github.com/tori29umai0123/ComfyUI-SAM3DBody_utills) repo (Issues / Discussions).

For SAM 3D Body / MHR core topics and upstream plugin chat, see the [PozzettiAndrea/ComfyUI-SAM3DBody Discussions](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody/discussions) and the [Comfy3D Discord](https://discord.gg/bcdQCUjnHE).

## Example workflows

Seven ready-made workflows ship under `workflows/`. Load them from ComfyUI's `Workflow → Open` menu. The bundled `workflows/input_image*.png` / `workflows/input_mask*.png` work as drop-in test inputs.

| File | What it does | Needs Blender |
|---|---|---|
| **`SAM3Dbody_webUI.json`** | **Web-UI editor variant.** Drives `SAM 3D Body: Pose Editor` and `SAM 3D Body: Character Editor` from a fullscreen browser modal and feeds the confirmed JSON straight into the Render node — minimal scaffolding. | ❌ |
| **`SAM3Dbody_image.json`** | Minimal image-rendering workflow. Takes the pose from your input image and renders it onto an arbitrary body shape. | ❌ |
| **`SAM3Dbody_FBX.json`** | FBX export workflow. Takes the pose from your input image, applies it to an arbitrary body shape, and exports a rigged FBX with a posed animation track — importable into Unity / Unreal Engine. | ✅ |
| **`SAM3Dbody_FBXAnimation.json`** | **Video motion-capture workflow.** Pipes a video loaded via `VHS_LoadVideo` into `SAM 3D Body: Export Animated FBX` and writes an animated FBX covering every frame. | ✅ |
| **`SAM3Dbody_ BVH.json`** | BVH export workflow. Takes the pose from your input image, applies it to an arbitrary body shape, and exports a single-pose BVH. No 3D preview node is included. | ✅ |
| **`SAM3Dbody_BVHAnimation.json`** | **Video motion-capture BVH workflow.** Pipes a video loaded via `VHS_LoadVideo` into `SAM 3D Body: Export Animated BVH` and writes an animated BVH covering every frame. No 3D preview node is included. | ✅ |
| **`SAM3Dbody _QIE_VNCCSpose.json`** | A real-world usage example. Combines [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit) + the VNCCSpose LoRA: extract the pose from a reference character of a different body shape, render it onto an arbitrary 3D character body, then image-edit the result. | ❌ |

### How `SAM3Dbody_webUI.json` fits together — confirm pose / character in your browser

Instead of poking at slider widgets on the ComfyUI canvas, two purpose-built nodes open a **fullscreen modal inside the browser** with a 3D preview, segmentation + pose estimation, and bone editing baked in. The confirmed JSON is stored in the workflow file so re-opening the node never loses your work. The UI has a JA / EN toggle.

#### `SAM 3D Body: Pose Editor` — confirm the pose in the browser

![Pose Editor](docs/sample5.png)

The node carries an **"Open Pose Editor"** button. Click it to upload an image, run segmentation + pose inference, fine-tune bones in the 3D viewport (rotation / IK translate / lean correction), and finally hit **"Confirm & Close / 確定して閉じる"**. The node then emits `pose_json` (STRING) plus the source image's `width` / `height` (INT) so it plugs straight into `SAM 3D Body: Render Human From Pose And Chara JSON`.

#### `SAM 3D Body: Character Editor` — confirm the body shape in the browser

![Character Editor](docs/sample6.png)

Likewise the **"Open Character Editor"** button drops you into a modal where the MHR neutral T-pose body responds in real time to **9-axis PCA shape sliders, four bone-length sliders, and the face / body blendshape sliders**. You can also load `presets/<active>/chara_settings_presets/*.json` from this view. On confirm the node outputs `chara_json` (STRING) — feed it directly into the Render node, `SAM 3D Body: Export Rigged FBX`, etc.

### How `SAM3Dbody _QIE_VNCCSpose.json` fits together

![sample3](docs/sample3.png)

- **Left two panels** — inputs (a reference character for the pose + a different character for the target body shape)
- **Middle panel** — intermediate output (the 3D character body, posed by this plugin, that matches the target body shape)
- **Right panel** — final output (the middle panel handed to an image-editing model and composed into the finished art)

The intended use is treating this plugin's render as an **intermediate artifact fed into an image-editing model** (here Qwen-Image-Edit). It lets you merge two separate inputs — "pose from character A, body type from character B" — via a geometrically correct 3D body in the middle.

## SAM 3D Body: Render Human From Pose And Chara JSON node

Takes `pose_json` + `chara_json` as inputs and renders the 3D body. Character authoring (PCA shape / bone length / blend shapes) is delegated to `chara_json`, so the only widgets left on this node are **camera controls + lean correction**.

- Character shape (`shape_params` / `scale_params`) carried inside pose_json is **ignored**.
- Only pose-ish fields (`global_rot` / `body_pose_params` / `hand_pose_params` / `expr_params`) are taken from pose_json.
- Body shape is fully driven by `chara_json` (from the Setting Chara JSON node or the Character Editor).

### Required inputs

| Parameter | Default | Range | Notes |
|---|---|---|---|
| model | — | — | SAM 3D Body model (Load node output) |
| pose_json | `"{}"` | — | Pose JSON (Process Image to Pose JSON node or Pose Editor output) |
| chara_json | `"{}"` | — | Character JSON (Setting Chara JSON node or Character Editor output) |
| offset_x | 0.0 | −5.0 … 5.0 | Horizontal positional offset (added to `camera[0]` in meters). **Pans the subject left/right within the image** |
| offset_y | 0.0 | −5.0 … 5.0 | Vertical positional offset (added to `camera[1]`). **Pans the subject up/down within the image** |
| scale_offset | 1.0 | 0.1 … 5.0 | Camera distance **multiplier**. 1.0 = identity, 0.1 = extreme zoom in, 5.0 = zoom out |
| camera_yaw_deg | 0.0 | −180 … 180 | **Horizontal orbit** around the subject centroid. `+` rotates the camera to the viewer's right (subject appears to turn left). `±180` = back view |
| camera_pitch_deg | 0.0 | −89 … 89 | **Vertical orbit** around the subject centroid. `+` raises the camera (looking **down** at the subject), `−` looks **up**. The subject stays centered in the frame |
| width | 0 | 0 … 8192 | Output width (0 = use pose_json's source image size) |
| height | 0 | 0 … 8192 | Output height (0 = same) |
| pose_adjust | 0.0 | 0.0 … 1.0 | **Lean-correction** strength. SAM 3D Body tends to estimate standing subjects as slightly forward-leaning; this slider tilts the spine→neck chain backwards to compensate. 1.0 = full correction; live action: 0.0, illustrated input: ~0.5 as a guideline |

### Optional inputs

| Parameter | Notes |
|---|---|
| background_image | Background image. If left unconnected, the background is black |

### Outputs

| Output | Notes |
|---|---|
| image | The rendered RGB image |

## SAM 3D Body: Setting Chara JSON node

The **character-authoring half** split out from the legacy Render node. Drives preset selection + 9-axis PCA shape sliders + 4 bone-length sliders + face/body blendshape sliders, and emits `chara_json` (STRING). Plug it directly into the Render node's `chara_json` or any Export node's `character_json`.

- Output JSON is schema-compatible with `chara_settings_presets/*.json`.
- Picking a preset (autosave / female / male / chibi …) overwrites every slider.
- **Autosave**: on every execute (when `preset` ≠ `reset`), the current values are written to `chara_settings_presets/autosave.json` and become the next session's slider defaults.

### Inputs

| Parameter | Default | Notes |
|---|---|---|
| preset | `autosave` | Preset selector. `autosave` is pinned to the top |
| body_* | 0.0 | PCA 9-axis body shape sliders (see below) |
| bone_* | 1.0 | Four bone-length scale sliders (see below) |
| bs_* | 0.0 | 20 blend-shape sliders (see below) |

### Outputs

| Output | Notes |
|---|---|
| chara_json | Character settings JSON string. Three blocks: `body_params` / `bone_lengths` / `blendshapes`. Schema-compatible with `chara_settings_presets/*.json` |

### Body Params (PCA body shape) — `body_*`

Nine sliders driving the **first 9 of MHR's 45-dim PCA body shape basis**.

- Default: **0.0 (neutral)**
- Range: **−5.0 … +5.0**
- Practical range: ±1 is subtle, ±3 is extreme, ±5 starts to break
- **Internally normalised**: PCA basis magnitudes shrink rapidly past PC0, so we multiply each slider internally by `[1.00, 2.78, 4.42, 8.74, 10.82, 11.70, 13.39, 13.83, 16.62]`. ±1 on any axis produces a comparable-sized deformation.

Slider names are educated guesses at the axes' learned semantics — **the actual direction may not perfectly match the label** (try them).

| Parameter | PCA component | Intended effect |
|---|---:|---|
| body_fat              | `shape_params[0]` | Body fat. + = fatter, − = leaner |
| body_muscle           | `shape_params[1]` | Muscle mass. + = muscular, − = delicate |
| body_fat_muscle       | `shape_params[2]` | Fat vs muscle balance |
| body_limb_girth       | `shape_params[3]` | Limb (arm / leg) thickness |
| body_limb_muscle      | `shape_params[4]` | Limb muscularity |
| body_limb_fat         | `shape_params[5]` | Limb subcutaneous fat |
| body_chest_shoulder   | `shape_params[6]` | Chest + shoulder width. + = wider, − = narrower |
| body_waist_hip        | `shape_params[7]` | Waist / hip thickness. + = thicker, − = slimmer |
| body_thigh_calf       | `shape_params[8]` | Thigh / calf thickness |

### Bone length sliders — `bone_*`

Four sliders that rescale link lengths along specific MHR bone chains. Implemented as an extension of LBS with **per-joint isotropic scaling** — bone-length changes also isotropically scale the mesh around that joint, so proportions stay locked (`bone_torso=0.6` shrinks the torso to 0.6× **and** thins it 0.6× — you get a "short" character, not a "stubby" one).

- Scale 1.0 = unchanged. 0.5 = half, 2.0 = double
- Branch joints (`clavicle_l/r`, `thigh_l/r`) stay at 1.0 so shoulder / hip width are preserved

| Parameter | Default | Range | Affected MHR joints |
|---|---|---|---|
| bone_torso | 1.0 | 0.3 … 1.8 | pelvis itself + pelvis → neck_01 chain (crotch through neck base). Including pelvis lets the lower-belly mesh shrink alongside the skeleton |
| bone_neck  | 1.0 | 0.3 … 2.0 | `neck_01` → `head` link (neck length). `head`'s own mesh_scale is pinned to 1.0 — head size unchanged, only the neck stretches |
| bone_arm   | 1.0 | 0.3 … 2.0 | Descendants of `clavicle_l/r` (`upperarm`, `lowerarm`, `hand`, fingers) |
| bone_leg   | 1.0 | 0.3 … 2.0 | Descendants of `thigh_l/r` (`calf`, `foot`, toes) |

Implemented in `_apply_bone_length_scales` (`nodes/processing/process.py`). We split `joint_scale` (drives bone length) from `mesh_scale` (drives local isotropic mesh scale) and soften the latter with `_MESH_SCALE_STRENGTH=0.5` so shrinking the bone by 40% only thins the mesh by 20%.

### Blend shapes — `bs_*`

20 sliders blending FBX-derived morph targets from `tools/bone_backup/all_parts_bs.fbx` into the posed mesh. Each slider defaults to **0.0**, range **0.0…1.0**, fully active at 1.0. Combining sliders is additive.

The shape list is auto-discovered from `presets/face_blendshapes.npz` — add a new shape key in Blender, regenerate the npz, and the slider appears in the UI on next reload (no code changes). The `bs_` prefix is a UI-only label; FBX shape-key names and `chara_json` keys use the bare name.

#### Face

| Parameter | Effect |
|---|---|
| bs_face_big   | Face up |
| bs_face_small | Face down |
| bs_face_mangabig | Manga-inflated face |
| bs_face_manga | Anime-style face |
| bs_chin_sharp | Sharper chin |

#### Neck

| Parameter | Effect |
|---|---|
| bs_neck_thick | Thicker neck |
| bs_neck_thin  | Thinner neck |

#### Chest

| Parameter | Effect |
|---|---|
| bs_breast_full | Larger breasts |
| bs_breast_flat | Flatter breasts |
| bs_chest_slim  | Slimmer ribcage |

#### Shoulder

| Parameter | Effect |
|---|---|
| bs_shoulder_wide   | Wider shoulders |
| bs_shoulder_narrow | Narrower shoulders |
| bs_shoulder_slope  | Sloped / square shoulders |

#### Waist

| Parameter | Effect |
|---|---|
| bs_waist_slim | Slimmer waist |

#### Limbs

| Parameter | Effect |
|---|---|
| bs_limb_thick | Thicker limbs |
| bs_limb_thin  | Thinner limbs |
| bs_hand_big   | Bigger hands |
| bs_foot_big   | Bigger feet |

#### Muscle

| Parameter | Effect |
|---|---|
| bs_MuscleScale | Bulk up full-body muscle (macho) |

### Preset system

Drop a JSON of the slider shape into `chara_settings_presets/<name>.json` and it appears in the `preset` dropdown. Selecting a preset overwrites the slider widgets entirely (missing keys treat as neutral). The shipped `female.json` / `male.json` are reference examples.

## ⚠ Export Rigged FBX node (Blender required)

Writes a rigged FBX (**armature + skinned mesh + 30-frame static pose animation**) using the same character setup + pose as the Render node, into `<ComfyUI>/output/`. Imports directly into Blender / Unity / Unreal Engine.

> **⚠ Blender 4.1+ is required.** The node spawns `blender.exe --background --python tools/build_rigged_fbx.py` as a subprocess to assemble the armature, bind LBS weights, and export the FBX. Without Blender installed, only this node errors at runtime (the other nodes still work).

### Inputs

| Parameter | Notes |
|---|---|
| model | Output of `Load SAM 3D Body Model` |
| **character_json** | **Character settings JSON.** Wire the Render node's `settings_json` output, or paste any `chara_settings_presets/*.json` content (`body_params` / `bone_lengths` / `blendshapes`). The text area starts with `=== CHARACTER JSON ===` so you can tell which slot is which. |
| **pose_json** | **Pose JSON.** Wire the `pose_json` output of `SAM 3D Body: Process Image to Pose JSON` (`body_pose_params` / `hand_pose_params` / `global_rot`). The text area starts with `=== POSE JSON ===`. |
| blender_exe | Path to `blender.exe` (default `C:/Program Files/Blender Foundation/Blender 4.1/blender.exe`). Subprocess invocation requires Blender 4.1+ |
| output_filename | Output FBX name (default `sam3d_rigged.fbx`). Leave blank for a timestamped name |

### Outputs

| Output | Notes |
|---|---|
| fbx_path | Absolute path of the written FBX (`<ComfyUI>/output/<name>.fbx`) |


## ⚠ Export Animated FBX node (video motion capture, Blender required)

Writes a **Unity / Unreal-ready animated FBX from a video (IMAGE batch)** in a single node. Pass a frame sequence loaded via `ComfyUI-VideoHelperSuite`'s `VHS_LoadVideo` etc.; each frame goes through SAM 3D Body, joint rotations are estimated, and **all frames are baked as keyframes onto a character rigged at its base pose (body_pose=0)**. The resulting FBX plays directly in Animator / Animation / Timeline.

For example output, see the [demo video](#3-export-motion-captured-fbx--bvh-from-a-video) (`docs/sample1.mp4`) at the top of this README.

## ⚠ Export Posed BVH node (Blender required)

Writes a single pose to `<ComfyUI>/output/` as a **BVH**. Uses the same character settings from the Render node and pose from `Process Image to Pose JSON`. Output is skeleton / motion only (no mesh), and the bones are first reduced to a humanoid-compatible subset before writing.

> **⚠ Blender 4.1+ is required.** Calls `blender.exe --background --python tools/build_rigged_bvh.py` as a subprocess internally.

## Developer guide (for users editing blend shapes in Blender)

> **The rest of this section is for advanced users who want to edit `tools/bone_backup/all_parts_bs.fbx` in Blender to add or update their own blend shapes.** If you only render with the 18 shipped blend shapes, you can skip it.

Steps for adding a new blend shape, the `extract_face_blendshapes.py` / `rebuild_vertex_jsons.py` commands, and the full `tools/` script reference all live in a dedicated document:

- 📖 **[docs/DEVELOPER_GUIDE.en.md](docs/DEVELOPER_GUIDE.en.md)** — Developer guide (English)
- 📖 **[docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)** — 開発者ガイド (日本語)

## Credits

[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) by Meta AI ([paper](https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/))

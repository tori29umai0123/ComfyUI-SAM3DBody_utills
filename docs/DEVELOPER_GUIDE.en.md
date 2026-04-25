# Developer guide: blend-shape authoring + tools/ reference

**Language:** [🇯🇵 日本語](DEVELOPER_GUIDE.md) ・ 🇬🇧 English (current)

> This document is for **advanced users who want to edit `tools/bone_backup/all_parts_bs.fbx` in Blender to add or update their own blend shapes (shape keys)**. If you only want to render with the 18 shipped blend shapes, you don't need to read this — go back to the main [README.en.md](../README.en.md).

### ⚠ Editing blend shapes requires Blender (advanced users)

> **This section is for Blender users who want to author their own blend shapes or tweak existing ones.** If you only want to render using the 18 shipped blend shapes, you can skip it entirely — Blender is not required for that workflow.
>
> **Why Blender is required:** editing shape keys on the source FBX in the GUI, then re-running `tools/extract_face_blendshapes.py` headless to regenerate `presets/face_blendshapes.npz`.

The plugin's blend shapes live as **shape keys** on `tools/bone_backup/all_parts_bs.fbx`. Any add / edit / value change must be done in [Blender](https://www.blender.org/).

- **Tested with:** Blender 4.1 (`C:/Program Files/Blender Foundation/Blender 4.1/blender.exe`). Other versions likely work, but the hardcoded path inside `tools/extract_face_blendshapes.py` assumes 4.1.

Workflow:

1. Open `tools/bone_backup/all_parts_bs.fbx` in Blender
2. Add / edit / rename shape keys on the `mhr_reference` object
3. Save as FBX (use the export settings below)
4. From a shell, run `tools/extract_face_blendshapes.py` via headless Blender — it auto-syncs `presets/face_blendshapes.npz` + `chara_settings_presets/*.json` + `process.py`'s `_UI_BLENDSHAPE_ORDER`

#### FBX export settings

![Blender FBX export settings](blender_fbx_export_settings.png)

| Option | Value |
|---|---|
| Path Mode | Automatic |
| Batch Mode | Off |
| **Limit to** | **all OFF** (don't check Selected / Visible / Active Collection) |
| **Object Types** | **Armature + Mesh** only |
| Custom Properties | OFF |
| **Scale** | **1.00** |
| **Apply Scalings** | **FBX All** |
| **Forward** | **-Z Forward** |
| **Up** | **Y Up** |
| Apply Unit | ON |
| Use Space Transform | ON |
| Apply Transform | ON |
| Bake Animation | ON |

**Important:** Forward = -Z / Up = Y / Scale 1.0 are required to match the internal axis-swap matrix `_FBX_TO_MHR_ROT`. Other axis setups will misalign the extracted blend-shape deltas.

## Developer guide: adding new blend shapes

The `bs_*` sliders are auto-discovered from `presets/face_blendshapes.npz`, so **no Python changes are needed** to add one.

### Steps

1. **Open `tools/bone_backup/all_parts_bs.fbx` in Blender**
2. Select the `mhr_reference` mesh
3. In Object Properties → **Shape Keys**, add a new key and sculpt it (Basis first if it's missing)
4. Export the FBX with the settings documented earlier (Forward=-Z / Up=Y / Scale 1.0)
5. Run the extractor — this single command also syncs every preset and the UI order:

   ```
   "C:/Program Files/Blender Foundation/Blender 4.1/blender.exe" ^
       --background --python tools/extract_face_blendshapes.py
   ```

   - `presets/face_blendshapes.npz` is rebuilt from the FBX
   - `chara_settings_presets/*.json` get the new key added (default 0.0) and stale keys removed
   - `process.py`'s `_UI_BLENDSHAPE_ORDER` gets the new key inserted at the end of its category (face/neck/chest/shoulder/waist/limbs inferred from the name prefix)

6. Refresh ComfyUI — the new slider appears in the UI.

### Editing existing shape values

Re-sculpt the key in Blender, re-export FBX, re-run step 5. Values are **overwritten from source** each time, so there's no manual diff to manage.

### Rebuild everything from scratch

```
del presets\face_blendshapes.npz
del presets\*_vertices.json
```

Then re-run steps 5 and 6. `tools/rebuild_vertex_jsons.py` regenerates the per-object vertex JSON from the npz.

## FBX update workflow (one-liner)

After editing the FBX in Blender:

### 1. Extract blend shapes + auto-sync presets and UI order

```
"C:/Program Files/Blender Foundation/Blender 4.1/blender.exe" ^
    --background --python custom_nodes/ComfyUI-SAM3DBody_utills/tools/extract_face_blendshapes.py
```

### 2. Regenerate the per-object vertex JSON

```
.venv/Scripts/python.exe custom_nodes/ComfyUI-SAM3DBody_utills/tools/rebuild_vertex_jsons.py
```

### 3. Refresh ComfyUI

Reload the browser — the new shape keys and updated UI order show up automatically.

## tools/ script reference

| Script | Run in | Role |
|---|---|---|
| `tools/extract_face_blendshapes.py` | **Blender** (`--background --python`) | Walks all mesh objects + shape keys in `tools/bone_backup/all_parts_bs.fbx`, writes `presets/face_blendshapes.npz` (`base__<obj>`, `delta__<obj>__<shape>`, `meta_shapes`, `meta_objects`, `all_base__<obj>`). Calls `sync_presets_with_npz.sync_all()` at the end to keep presets + UI aligned. |
| `tools/rebuild_vertex_jsons.py` | **ComfyUI venv** | Nearest-neighbour-matches the npz's `all_base__<obj>` positions against MHR rest vertices and writes `presets/<obj>_vertices.json` files. Auto-removes stale JSONs when the FBX object list changes. |
| `tools/sync_presets_with_npz.py` | **ComfyUI venv** or Blender | Standalone sync (normally called automatically at the end of `extract_face_blendshapes.py`). Adds missing keys to preset JSONs with value 0.0, drops stale keys, rewrites `_UI_BLENDSHAPE_ORDER` in `process.py`. |
| `tools/build_rigged_fbx.py` | **Blender** (spawned by the Export Rigged FBX node) | Builds the rigged FBX from the node's intermediate JSON (armature + mesh + LBS weights + posed animation + FBX export). |
| `tools/rename_blendshape_keys.py` | **Blender** | One-off helper for renaming shape keys in `all_parts_bs.fbx` (edit the `RENAMES` dict inside and run). |
| `tools/export_reference_obj.py` | **ComfyUI venv** | Writes `tools/bone_backup/mhr_reference.obj` — a single non-partitioned MHR rest-pose OBJ. Used as a safety backup and as a starting point if the FBX is ever rebuilt from scratch. |

## Preset packs (distributable blend-shape definitions)

This plugin bundles its blend-shape definitions, vertex mapping, and character preset JSONs into a single unit called a **preset pack**. The pack system exists so users can author their own blend-shape sets and share them with others as self-contained folders.

### Switching packs

To use a pack someone else shared (say `my_custom_pack`):

1. Drop the pack folder into `presets/my_custom_pack/` (same layout as above)
2. Open the repo-root `config.ini` and change `[active] pack = default` to `pack = my_custom_pack`
3. Reload ComfyUI (F5)

If the named pack is missing, the runtime falls back to `default` automatically.

### Creating your own pack

1. Copy `presets/default/` to `presets/my_pack/`
2. Edit `config.ini` so `[active] pack = my_pack`
3. Open `tools/bone_backup/all_parts_bs.fbx` in Blender and add / edit the shape keys you want
4. Run `tools/extract_face_blendshapes.py` — it writes into the active pack (`my_pack`), regenerates `face_blendshapes.npz`, and auto-syncs both `chara_settings_presets/*.json` and `process.py`'s `_UI_BLENDSHAPE_ORDER`
5. Optionally run `tools/rebuild_vertex_jsons.py` to refresh `mhr_reference_vertices.json`
6. Zip the `presets/my_pack/` folder and distribute

A pack is fully self-contained — the recipient just unzips it under `presets/` and edits `config.ini`'s `[active] pack`, no code change needed.


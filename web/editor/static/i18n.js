// Static-text i18n for the 3D Body Standalone UI.
//
// - Every static DOM string carries a `data-i18n="key"` attribute (or
//   `data-i18n-placeholder` / `data-i18n-title` for input placeholders and
//   button tooltips). `applyLang()` walks the tree and overwrites those
//   fragments with the current dictionary's value.
// - Language preference persists in localStorage under `body3d.lang`.
// - app.js emits a `body3d:langchange` event after switching so dynamic
//   components can refresh their own copy if needed.

const DICT = {
  ja: {
    "app.title": "3D Body Standalone",
    "tab.image": "画像",
    "tab.video": "動画",
    "tab.make": "キャラメイク",
    "tab.admin": "Preset Admin",

    "section.input": "Input",
    "section.video": "動画 → アニメーション FBX",
    "section.make": "キャラメイク",
    "section.pack_admin": "Preset Pack 管理",
    "section.fbx_rebuild": "FBX Rebuild",
    "section.character": "Character",
    "section.health": "Health",
    "section.pose_adjust": "Pose 補正",

    "pose.lean_correction": "前かがみ補正",
    "pose.lean_correction_hint": "ポーズが前のめりになるとき、後ろに反らせる。リアルタイムで 3D プレビューに反映されます。",
    "pose.edit_enter": "ポーズ調整（回転・移動）",
    "pose.edit_exit":  "ポーズ調整（回転・移動）を終了",
    "pose.edit_locked": "ポーズ調整（回転・移動）中 — 他の操作はロックされています",
    "pose.selected_bone": "選択中のボーン",
    "pose.rot_x": "X 軸 (°)",
    "pose.rot_y": "Y 軸 (°)",
    "pose.rot_z": "Z 軸 (°)",
    "pose.reset_bone": "このボーンをリセット",
    "pose.reset_all": "全ボーンをリセット",
    "pose.undo": "元に戻す (Ctrl+Z)",
    "pose.redo": "やり直す (Ctrl+Y)",
    "pose.edit_hint": "3D ビューはボーン選択 + IK 移動専用です。ハンドルをクリックして選択 → 出てくる移動ギズモ (X/Y/Z 矢印) をドラッグすると、IK で親ボーンが追従します。細かい回転は左側の X / Y / Z スライダーで行ってください。",
    "pose.bone.Hips": "Hips (腰)",
    "pose.bone.Spine": "Spine (脊椎下)",
    "pose.bone.Spine1": "Spine1 (脊椎中)",
    "pose.bone.Spine2": "Spine2 (脊椎上)",
    "pose.bone.Neck": "Neck (首)",
    "pose.bone.Head": "Head (頭)",
    "pose.bone.LeftShoulder": "LeftShoulder (左肩)",
    "pose.bone.LeftArm": "LeftArm (左上腕)",
    "pose.bone.LeftForeArm": "LeftForeArm (左前腕)",
    "pose.bone.LeftHand": "LeftHand (左手)",
    "pose.bone.LeftHandThumb1":  "LeftHandThumb1 (左親指1)",
    "pose.bone.LeftHandThumb2":  "LeftHandThumb2 (左親指2)",
    "pose.bone.LeftHandThumb3":  "LeftHandThumb3 (左親指3)",
    "pose.bone.LeftHandThumb4":  "LeftHandThumb4 (左親指先)",
    "pose.bone.LeftHandIndex1":  "LeftHandIndex1 (左人差し指1)",
    "pose.bone.LeftHandIndex2":  "LeftHandIndex2 (左人差し指2)",
    "pose.bone.LeftHandIndex3":  "LeftHandIndex3 (左人差し指3)",
    "pose.bone.LeftHandIndex4":  "LeftHandIndex4 (左人差し指先)",
    "pose.bone.LeftHandMiddle1": "LeftHandMiddle1 (左中指1)",
    "pose.bone.LeftHandMiddle2": "LeftHandMiddle2 (左中指2)",
    "pose.bone.LeftHandMiddle3": "LeftHandMiddle3 (左中指3)",
    "pose.bone.LeftHandMiddle4": "LeftHandMiddle4 (左中指先)",
    "pose.bone.LeftHandRing1":   "LeftHandRing1 (左薬指1)",
    "pose.bone.LeftHandRing2":   "LeftHandRing2 (左薬指2)",
    "pose.bone.LeftHandRing3":   "LeftHandRing3 (左薬指3)",
    "pose.bone.LeftHandRing4":   "LeftHandRing4 (左薬指先)",
    "pose.bone.LeftHandPinky1":  "LeftHandPinky1 (左小指1)",
    "pose.bone.LeftHandPinky2":  "LeftHandPinky2 (左小指2)",
    "pose.bone.LeftHandPinky3":  "LeftHandPinky3 (左小指3)",
    "pose.bone.LeftHandPinky4":  "LeftHandPinky4 (左小指先)",
    "pose.bone.RightShoulder": "RightShoulder (右肩)",
    "pose.bone.RightArm": "RightArm (右上腕)",
    "pose.bone.RightForeArm": "RightForeArm (右前腕)",
    "pose.bone.RightHand": "RightHand (右手)",
    "pose.bone.RightHandThumb1":  "RightHandThumb1 (右親指1)",
    "pose.bone.RightHandThumb2":  "RightHandThumb2 (右親指2)",
    "pose.bone.RightHandThumb3":  "RightHandThumb3 (右親指3)",
    "pose.bone.RightHandThumb4":  "RightHandThumb4 (右親指先)",
    "pose.bone.RightHandIndex1":  "RightHandIndex1 (右人差し指1)",
    "pose.bone.RightHandIndex2":  "RightHandIndex2 (右人差し指2)",
    "pose.bone.RightHandIndex3":  "RightHandIndex3 (右人差し指3)",
    "pose.bone.RightHandIndex4":  "RightHandIndex4 (右人差し指先)",
    "pose.bone.RightHandMiddle1": "RightHandMiddle1 (右中指1)",
    "pose.bone.RightHandMiddle2": "RightHandMiddle2 (右中指2)",
    "pose.bone.RightHandMiddle3": "RightHandMiddle3 (右中指3)",
    "pose.bone.RightHandMiddle4": "RightHandMiddle4 (右中指先)",
    "pose.bone.RightHandRing1":   "RightHandRing1 (右薬指1)",
    "pose.bone.RightHandRing2":   "RightHandRing2 (右薬指2)",
    "pose.bone.RightHandRing3":   "RightHandRing3 (右薬指3)",
    "pose.bone.RightHandRing4":   "RightHandRing4 (右薬指先)",
    "pose.bone.RightHandPinky1":  "RightHandPinky1 (右小指1)",
    "pose.bone.RightHandPinky2":  "RightHandPinky2 (右小指2)",
    "pose.bone.RightHandPinky3":  "RightHandPinky3 (右小指3)",
    "pose.bone.RightHandPinky4":  "RightHandPinky4 (右小指先)",
    "pose.bone.LeftUpLeg": "LeftUpLeg (左もも)",
    "pose.bone.LeftLeg": "LeftLeg (左すね)",
    "pose.bone.LeftFoot": "LeftFoot (左足)",
    "pose.bone.LeftToe_End": "LeftToe_End (左つま先)",
    "pose.bone.RightUpLeg": "RightUpLeg (右もも)",
    "pose.bone.RightLeg": "RightLeg (右すね)",
    "pose.bone.RightFoot": "RightFoot (右足)",
    "pose.bone.RightToe_End": "RightToe_End (右つま先)",
    "tooltip.pose_edit": "3D ビュー上でボーンをクリックして回転を微調整する。終了するまで他の操作はロックされます",

    "make.hint": "MHR 既定の T-pose 素体に対して、下の Character スライダーで体形を調整できます。完成したら「json をダウンロード」で JSON をローカル保存してください。",
    "make.download": "json をダウンロード",
    "make.downloaded": "JSON を保存しました",

    "input.drop": "クリックして画像を選択 / ドロップ",
    "input.segmentation_settings": "Segmentation",
    "input.use_segmentation": "セグメンテーションを使う",
    "input.prompt": "prompt",
    "input.threshold": "threshold",
    "input.min_w": "min w (px)",
    "input.min_h": "min h (px)",
    "input.run": "ポーズを推定",
    "input.stop": "推定を停止",
    "input.cancel": "推定を中止",
    "input.cancelled": "キャンセルしました",
    "input.done": "推定終了",

    "video.drop": "動画を選択 / ドロップ",
    "video.params": "パラメータ",
    "video.use_segmentation": "セグメンテーションを使う",
    "video.segmentation_thr": "segmentation threshold",
    "video.bbox_thr": "bbox thr",
    "video.root_motion": "root motion",
    "video.fps": "fps (0 = 自動)",
    "video.stride": "stride",
    "video.max_frames": "max frames (0 = 全フレーム)",
    "video.run": "モーションを推定",
    "video.stop": "推定を停止",
    "video.cancel": "推定を中止",
    "video.cancelled": "キャンセルしました",
    "video.done": "推定終了",
    "video.download": "FBX をダウンロード",
    "video.download_bvh": "BVH をダウンロード",
    "video.rebuilding": "キャラクター差し替え中…",
    "video.hint": "※ 体形は下の Character パネル (preset / JSON) で指定した値を rig にベイクします。モーション推定後に preset を切り替えると、キャラを差し替えた FBX が自動再生されます。",

    "character.preset_placeholder": "(preset)",
    "character.load": "Load",
    "character.reset": "Reset",
    "character.export_fbx": "FBX をダウンロード",
    "character.export_bvh": "BVH をダウンロード",
    "character.body_summary": "Body (PCA 9 軸)",
    "character.bone_summary": "Bone length (4 部位)",
    "character.bs_summary": "Blendshapes",
    "character.json_drop": "キャラクター JSON をアップロード (任意)",
    "character.json_loaded": "JSON 適用済み:",
    "character.preset_loaded": "preset 適用済み:",

    "admin.active_label": "active: -",
    "admin.switch": "Switch",
    "admin.new_name": "新しい pack 名...",
    "admin.clone": "Clone",
    "admin.delete": "選択中の pack を削除",
    "admin.fbx_hint": "Blender で tools/bone_backup/all_parts_bs.fbx を編集してシェイプキーを追加・編集した後、下にドロップして Rebuild を押すと face_blendshapes.npz と <obj>_vertices.json が再生成されて、スライダー一覧にも反映されます。",
    "admin.fbx_drop": "新 FBX を選択 / ドロップ (all_parts_bs.fbx)",
    "admin.rebuild": "Rebuild",
    "admin.placeholder": "Preset Pack の管理タブです。左のセクションで pack の switch / clone / delete、FBX の再構築を行えます。",

    "viewport.initial": "画像を読み込んで「ポーズを推定」を押すと 3D メッシュが表示されます",
    "viewport.save_png": "PNG 保存",
    "viewport.range_enter": "画像範囲",
    "viewport.range_exit": "画像範囲指定を終了",
    "viewport.range_reset": "画像範囲指定reset",
    "viewport.bg_color": "背景色指定",
    "input.lhand_drop": "Left hand image",
    "input.rhand_drop": "Right hand image",
    "input.flip_lr": "左右反転",
    "status.init": "initializing…",

    "tooltip.pack_switch": "選択したプリセットを適用",
    "tooltip.reset": "全スライダーを既定値に戻す",
    "tooltip.export_fbx": "Blender 経由でリグ付き FBX を書き出す",
    "tooltip.export_bvh": "Blender 経由で FBX を BVH に変換して書き出す",
    "tooltip.pack_clone": "default を複製して新規 pack 作成",
    "tooltip.pack_delete": "選択中の pack を削除 (default は不可)",
    "tooltip.fbx_rebuild": "Blender で npz と vertex JSON を再生成",
    "tooltip.save_png": "現在のビュワー表示を PNG として保存",
    "tooltip.range_select": "ドラッグでレンダリング画像の出力範囲を指定（pose_image 出力に反映）",
    "tooltip.range_reset": "保存された範囲指定をリセットして全画面出力に戻す",
    "tooltip.bg_color": "pose_image 出力の背景色を指定（既定は白）",
    "tooltip.hand_flip": "読み込んだ手の画像を左右反転する",
    "tooltip.make_download": "現在の Character スライダー値を character JSON としてダウンロード",
    "tooltip.video_download": "現在表示中のアニメーション FBX をローカルに保存",
    "tooltip.video_download_bvh": "現在のアニメーション FBX を BVH に変換して保存",

    "lang.label": "言語",
  },
  en: {
    "app.title": "3D Body Standalone",
    "tab.image": "Image",
    "tab.video": "Video",
    "tab.make": "Character Make",
    "tab.admin": "Preset Admin",

    "section.input": "Input",
    "section.video": "Video → animated FBX",
    "section.make": "Character Make",
    "section.pack_admin": "Preset pack admin",
    "section.fbx_rebuild": "FBX rebuild",
    "section.character": "Character",
    "section.health": "Health",
    "section.pose_adjust": "Pose adjust",

    "pose.lean_correction": "Lean correction",
    "pose.lean_correction_hint": "When the pose leans forward, tilt it back. The 3D preview updates in real time.",
    "pose.edit_enter": "Adjust pose (rotate / translate)",
    "pose.edit_exit":  "Finish pose adjust",
    "pose.edit_locked": "Pose editor active — other controls locked",
    "pose.selected_bone": "Selected bone",
    "pose.rot_x": "X axis (°)",
    "pose.rot_y": "Y axis (°)",
    "pose.rot_z": "Z axis (°)",
    "pose.reset_bone": "Reset this bone",
    "pose.reset_all": "Reset all bones",
    "pose.undo": "Undo (Ctrl+Z)",
    "pose.redo": "Redo (Ctrl+Y)",
    "pose.edit_hint": "The 3D view is for bone selection + IK move only: click a handle to pick a bone, then drag the translate gizmo (X / Y / Z arrows) that appears on it — parents follow via IK. Use the X / Y / Z sliders on the side panel for fine rotation.",
    "pose.bone.Hips": "Hips",
    "pose.bone.Spine": "Spine (lower)",
    "pose.bone.Spine1": "Spine1 (mid)",
    "pose.bone.Spine2": "Spine2 (upper)",
    "pose.bone.Neck": "Neck",
    "pose.bone.Head": "Head",
    "pose.bone.LeftShoulder": "LeftShoulder",
    "pose.bone.LeftArm": "LeftArm",
    "pose.bone.LeftForeArm": "LeftForeArm",
    "pose.bone.LeftHand": "LeftHand",
    "pose.bone.LeftHandThumb1":  "LeftHandThumb1 (thumb 1)",
    "pose.bone.LeftHandThumb2":  "LeftHandThumb2 (thumb 2)",
    "pose.bone.LeftHandThumb3":  "LeftHandThumb3 (thumb 3)",
    "pose.bone.LeftHandThumb4":  "LeftHandThumb4 (thumb tip)",
    "pose.bone.LeftHandIndex1":  "LeftHandIndex1 (index 1)",
    "pose.bone.LeftHandIndex2":  "LeftHandIndex2 (index 2)",
    "pose.bone.LeftHandIndex3":  "LeftHandIndex3 (index 3)",
    "pose.bone.LeftHandIndex4":  "LeftHandIndex4 (index tip)",
    "pose.bone.LeftHandMiddle1": "LeftHandMiddle1 (middle 1)",
    "pose.bone.LeftHandMiddle2": "LeftHandMiddle2 (middle 2)",
    "pose.bone.LeftHandMiddle3": "LeftHandMiddle3 (middle 3)",
    "pose.bone.LeftHandMiddle4": "LeftHandMiddle4 (middle tip)",
    "pose.bone.LeftHandRing1":   "LeftHandRing1 (ring 1)",
    "pose.bone.LeftHandRing2":   "LeftHandRing2 (ring 2)",
    "pose.bone.LeftHandRing3":   "LeftHandRing3 (ring 3)",
    "pose.bone.LeftHandRing4":   "LeftHandRing4 (ring tip)",
    "pose.bone.LeftHandPinky1":  "LeftHandPinky1 (pinky 1)",
    "pose.bone.LeftHandPinky2":  "LeftHandPinky2 (pinky 2)",
    "pose.bone.LeftHandPinky3":  "LeftHandPinky3 (pinky 3)",
    "pose.bone.LeftHandPinky4":  "LeftHandPinky4 (pinky tip)",
    "pose.bone.RightShoulder": "RightShoulder",
    "pose.bone.RightArm": "RightArm",
    "pose.bone.RightForeArm": "RightForeArm",
    "pose.bone.RightHand": "RightHand",
    "pose.bone.RightHandThumb1":  "RightHandThumb1 (thumb 1)",
    "pose.bone.RightHandThumb2":  "RightHandThumb2 (thumb 2)",
    "pose.bone.RightHandThumb3":  "RightHandThumb3 (thumb 3)",
    "pose.bone.RightHandThumb4":  "RightHandThumb4 (thumb tip)",
    "pose.bone.RightHandIndex1":  "RightHandIndex1 (index 1)",
    "pose.bone.RightHandIndex2":  "RightHandIndex2 (index 2)",
    "pose.bone.RightHandIndex3":  "RightHandIndex3 (index 3)",
    "pose.bone.RightHandIndex4":  "RightHandIndex4 (index tip)",
    "pose.bone.RightHandMiddle1": "RightHandMiddle1 (middle 1)",
    "pose.bone.RightHandMiddle2": "RightHandMiddle2 (middle 2)",
    "pose.bone.RightHandMiddle3": "RightHandMiddle3 (middle 3)",
    "pose.bone.RightHandMiddle4": "RightHandMiddle4 (middle tip)",
    "pose.bone.RightHandRing1":   "RightHandRing1 (ring 1)",
    "pose.bone.RightHandRing2":   "RightHandRing2 (ring 2)",
    "pose.bone.RightHandRing3":   "RightHandRing3 (ring 3)",
    "pose.bone.RightHandRing4":   "RightHandRing4 (ring tip)",
    "pose.bone.RightHandPinky1":  "RightHandPinky1 (pinky 1)",
    "pose.bone.RightHandPinky2":  "RightHandPinky2 (pinky 2)",
    "pose.bone.RightHandPinky3":  "RightHandPinky3 (pinky 3)",
    "pose.bone.RightHandPinky4":  "RightHandPinky4 (pinky tip)",
    "pose.bone.LeftUpLeg": "LeftUpLeg (thigh)",
    "pose.bone.LeftLeg": "LeftLeg (shin)",
    "pose.bone.LeftFoot": "LeftFoot",
    "pose.bone.LeftToe_End": "LeftToe_End (toe tip)",
    "pose.bone.RightUpLeg": "RightUpLeg (thigh)",
    "pose.bone.RightLeg": "RightLeg (shin)",
    "pose.bone.RightFoot": "RightFoot",
    "pose.bone.RightToe_End": "RightToe_End (toe tip)",
    "tooltip.pose_edit": "Click a bone handle in the 3D view to fine-tune rotation. Other controls are locked until you exit.",

    "make.hint": "Tweak the MHR neutral T-pose character with the sliders below. Hit \"Download JSON\" to save the current body shape as a JSON file.",
    "make.download": "Download JSON",
    "make.downloaded": "JSON saved",

    "input.drop": "Click to pick an image, or drop one here",
    "input.segmentation_settings": "Segmentation",
    "input.use_segmentation": "use segmentation",
    "input.prompt": "prompt",
    "input.threshold": "threshold",
    "input.min_w": "min w (px)",
    "input.min_h": "min h (px)",
    "input.run": "Run pose estimation",
    "input.stop": "Stop estimation",
    "input.cancel": "Cancel",
    "input.cancelled": "Cancelled",
    "input.done": "Inference done",

    "video.drop": "Click to pick a video, or drop one here",
    "video.params": "Parameters",
    "video.use_segmentation": "use segmentation",
    "video.segmentation_thr": "segmentation threshold",
    "video.bbox_thr": "bbox thr",
    "video.root_motion": "root motion",
    "video.fps": "fps (0 = auto)",
    "video.stride": "stride",
    "video.max_frames": "max frames (0 = all)",
    "video.run": "Infer motion",
    "video.stop": "Stop estimation",
    "video.cancel": "Cancel",
    "video.cancelled": "Cancelled",
    "video.done": "Inference done",
    "video.download": "Download FBX",
    "video.download_bvh": "Download BVH",
    "video.rebuilding": "rebuilding FBX with new character…",
    "video.hint": "The Character panel below (preset / JSON) is baked into the rig. After motion inference, swapping a preset re-builds the FBX with the new body automatically.",

    "character.preset_placeholder": "(preset)",
    "character.load": "Load",
    "character.reset": "Reset",
    "character.export_fbx": "Download FBX",
    "character.export_bvh": "Download BVH",
    "character.body_summary": "Body (PCA 9 axes)",
    "character.bone_summary": "Bone length (4 chains)",
    "character.bs_summary": "Blendshapes",
    "character.json_drop": "Upload character JSON (optional)",
    "character.json_loaded": "JSON applied:",
    "character.preset_loaded": "preset applied:",

    "admin.active_label": "active: -",
    "admin.switch": "Switch",
    "admin.new_name": "new pack name...",
    "admin.clone": "Clone",
    "admin.delete": "Delete selected",
    "admin.fbx_hint": "Edit tools/bone_backup/all_parts_bs.fbx in Blender (add or edit shape keys), drop it below and hit Rebuild. face_blendshapes.npz and <obj>_vertices.json are regenerated and the slider list refreshes.",
    "admin.fbx_drop": "Drop a new FBX here (all_parts_bs.fbx)",
    "admin.rebuild": "Rebuild",
    "admin.placeholder": "Preset Pack management. Use the left panel to switch / clone / delete packs and rebuild blend-shape data from an FBX.",

    "viewport.initial": "Pick an image and press \"Run pose estimation\" to see the 3D mesh here.",
    "viewport.save_png": "Save PNG",
    "viewport.range_enter": "Image range",
    "viewport.range_exit": "Finish range select",
    "viewport.range_reset": "Reset range",
    "viewport.bg_color": "Background color",
    "input.lhand_drop": "Left hand image",
    "input.rhand_drop": "Right hand image",
    "input.flip_lr": "Mirror",
    "status.init": "initializing…",

    "tooltip.pack_switch": "Apply the selected preset",
    "tooltip.reset": "Reset all sliders to defaults",
    "tooltip.export_fbx": "Export a rigged FBX via Blender",
    "tooltip.export_bvh": "Build a rigged FBX and convert it to BVH via Blender",
    "tooltip.pack_clone": "Clone default into a new pack",
    "tooltip.pack_delete": "Delete the selected pack (default is protected)",
    "tooltip.fbx_rebuild": "Rebuild npz + vertex JSON via Blender",
    "tooltip.save_png": "Save the current 3D view as PNG",
    "tooltip.range_select": "Drag to define the output region for pose_image",
    "tooltip.range_reset": "Clear the saved range and fall back to full-viewport output",
    "tooltip.bg_color": "Background color used for the pose_image output (default: white)",
    "tooltip.hand_flip": "Mirror the loaded hand image in place",
    "tooltip.make_download": "Download the current Character slider values as a character JSON",
    "tooltip.video_download": "Download the currently-playing animated FBX",
    "tooltip.video_download_bvh": "Convert the current animated FBX to BVH and download it",

    "lang.label": "Language",
  },
};

const SUPPORTED = Object.keys(DICT);
const STORAGE_KEY = "body3d.lang";

function resolveInitialLang() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && SUPPORTED.includes(saved)) return saved;
  } catch (_e) {}
  // Browser preference: anything that starts with `ja` → Japanese; else English.
  const nav = (navigator.language || "en").toLowerCase();
  return nav.startsWith("ja") ? "ja" : "en";
}

let currentLang = resolveInitialLang();

function t(key) {
  const d = DICT[currentLang] || DICT.ja;
  return d[key] ?? DICT.ja[key] ?? key;
}

function applyLang(lang) {
  if (!SUPPORTED.includes(lang)) lang = "ja";
  currentLang = lang;
  document.documentElement.setAttribute("lang", lang);
  try { localStorage.setItem(STORAGE_KEY, lang); } catch (_e) {}

  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const v = t(el.dataset.i18n);
    if (v !== undefined) el.textContent = v;
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    const v = t(el.dataset.i18nPlaceholder);
    if (v !== undefined) el.placeholder = v;
  });
  document.querySelectorAll("[data-i18n-title]").forEach((el) => {
    const v = t(el.dataset.i18nTitle);
    if (v !== undefined) el.title = v;
  });

  // Keep the <select> in sync if someone else called applyLang().
  const sel = document.getElementById("lang-select");
  if (sel && sel.value !== lang) sel.value = lang;

  // Let the rest of the app (dynamic strings) react if they want.
  document.dispatchEvent(new CustomEvent("body3d:langchange", { detail: { lang } }));
}

function initLanguagePicker() {
  const sel = document.getElementById("lang-select");
  if (!sel) return;
  sel.value = currentLang;
  sel.addEventListener("change", (e) => applyLang(e.target.value));
  applyLang(currentLang);
}

// Expose a tiny helper so app.js can localize dynamic strings too.
window.i18n = {
  t,
  get lang() { return currentLang; },
  applyLang,
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initLanguagePicker);
} else {
  initLanguagePicker();
}

// Copyright (c) 2025 Andrea Pozzetti
// SPDX-License-Identifier: MIT
//
// SAM 3D Body — Pose Editor + (self-contained ES module).
//
// Backend contracts:
//   GET  /sam3d/api/units                — display_unit + default_adult_height_m
//   POST /sam3d/api/plus/process        — multipart: image + payload + lhand_<id>/rhand_<id>
//   POST /sam3d/api/plus/render         — JSON: {plus_job_id, per_person_settings}
//   POST /sam3d/api/plus/drop           — JSON: {plus_job_id}
//
// The popup is hosted in an iframe; on confirm we postMessage back to the
// parent ComfyUI window with a {pose_image, pose_images, input_image,
// hand_l_images, hand_r_images} bundle.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";

const CONFIRM_MSG  = "sam3d-pose-plus-confirmed";
const CANCEL_MSG   = "sam3d-pose-plus-cancelled";
const _URL_PARAMS  = new URLSearchParams(location.search);
const NODE_ID = _URL_PARAMS.get("node_id");
// Blender executable path the iframe was opened with. The widget pulls
// this from the node's ``blender_exe`` text field at openEditor time so
// the in-editor FBX / BVH buttons honour what the user typed without
// needing to run the node first. Falls back to whatever is stored in
// config.ini when the value isn't present in the URL (e.g. after a
// dev-time direct navigation).
let _blenderExe = (_URL_PARAMS.get("blender_exe") || "").trim();
if (!_blenderExe) {
  fetch("/sam3d/api/blender_path")
    .then((r) => (r.ok ? r.json() : null))
    .then((j) => {
      if (j && typeof j.blender_exe === "string") _blenderExe = j.blender_exe;
    })
    .catch(() => {});
}

// One-time cleanup: earlier versions persisted editor state in localStorage
// and image dataUrls in an IndexedDB database. The cache feature was
// removed (users found it surprising); wipe leftover entries on load so
// the data does not linger on disk.
try {
  const legacyKey = `sam3d_pose_plus_state_${NODE_ID || "default"}`;
  localStorage.removeItem(legacyKey);
} catch (_e) {}
try { indexedDB.deleteDatabase("sam3d_pose_plus"); } catch (_e) {}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  inputImage: null,   // {dataUrl, width, height, file}   — raw upload
  inputBlob:  null,   // Blob (re-used for multipart POST)
  inputImageElement: null,  // pre-decoded HTMLImageElement for sync canvas crops
  imageDisplaySize: { displayedW: 0, displayedH: 0, offsetX: 0, offsetY: 0 },
  view: "image",      // "image" | "3d"
  drawMode: false,
  drawType: "primary",   // "primary" | "object" (set when entering drawMode)
  persons: [],        // see _newPerson()
  activeId: null,
  multiJobId: null,
  inferenceRunning: false,
  unit: { display_unit: "cm", default_adult_height_m: 1.70 },
  // 2D overlay sprites cropped from the input image with SAM 2. They
  // render on top of the 3D scene at capture time. Each entry: see
  // _newObject().
  objects: [],
  activeObjectId: null,
  // Capture-region rectangle for the composite pose_image. Stored in
  // viewport CSS-pixel coords; we multiply by devicePixelRatio at
  // capture time to convert to canvas backing-store pixels.
  captureRange: null,    // { x, y, w, h } | null
  rangeMode: false,
  // Background colour for the OUTPUT pose_image. The live 3D view keeps
  // the editor's neutral grey so the user can see the mesh clearly
  // while editing — only the capture path applies this colour.
  bgColor: "#ffffff",
};

let _personCounter = 0;
let _objectCounter = 0;

function _newObject() {
  return {
    id: `o${_objectCounter++}`,
    objUrl: null,        // backend tmp URL for the cropped RGBA PNG
    dataUrl: null,        // base64 dataUrl of the cropped image (loaded once for compositing)
    naturalWidth: 0,
    naturalHeight: 0,
    bbox: null,          // primary bbox used for cropping (in image pixel coords)
    // Mirrors the person flow: additional positive boxes extend the
    // SAM 2 mask, negative boxes subtract from it. The backend re-runs
    // SAM 2 with all three on every recrop call.
    additionalBboxes: [],
    negativeBboxes: [],
    // Display state — normalised against the viewport CSS size:
    //   posX, posY ∈ [0, 1] are the sprite's centre point.
    posX: 0.5,
    posY: 0.5,
    scale: 1.0,           // multiplier of natural size at canvas's native dpr
    rotationDeg: 0,
    // ``opacity`` follows CSS semantics (1 = fully opaque, 0 = fully
    // transparent). The sidebar slider is labelled "透明度" — that's
    // (1 − opacity), so its default of 0% maps to opacity=1.0.
    opacity: 1.0,
  };
}

function _newPerson() {
  const id = `p${_personCounter++}`;
  return {
    id,
    bbox: null,         // [x1,y1,x2,y2] in image-pixel coords (null until drawn)
    heightInputValue: null,   // displayed value in current unit (null until typed/defaulted)
    heightMeters: state.unit.default_adult_height_m,
    lhandImage: null,   // {dataUrl, width, height, blob}
    rhandImage: null,
    pendingBboxDraw: false,   // true between "+ Add Person" click and the draw being completed
    // Pose / shape settings — populated by /multi_process and updated by
    // user edits (lean slider, bone rotation, preset load). Sent on every
    // /multi_render so the backend re-applies them.
    settings: null,           // {body_params, bone_lengths, blendshapes, pose_adjust}
    skeleton: null,           // {bones: [{name, joint_id, parent_name, world_position, world_quaternion}, ...]}
    selectedBoneName: null,   // currently selected bone in the rotation editor
    poseEditOpen: false,      // expand/collapse the Pose Edit sub-panel
    expanded: false,          // expand/collapse the whole Edit section
    // Per-person whole-body transform applied on top of the predicted
    // ``pred_cam_t * height_scale`` placement. Translation is in scene
    // meters, rotation is degrees (XYZ Euler, Three.js convention).
    transform: { translate: [0, 0, 0], rotate_deg: [0, 0, 0] },
    transformOpen: false,
    // SAM 2 masked-and-cropped preview URL — populated by /sam2_preview
    // shortly after the bbox is drawn. Falls back to the synchronous
    // bbox crop in ``rebuildPersonList`` while empty.
    trimmedUrl: null,
    maskScore: 0,
    // Additional positive bboxes (Shift+left-drag in the image view).
    // Each is segmented separately by SAM 2 and OR-ed onto the primary
    // person mask, so missed regions can be reclaimed.
    additionalBboxes: [],
    // User-drawn "exclude this region" rectangles (right-click drag in
    // the image view). Each entry is [x1, y1, x2, y2] in image-pixel
    // coords. Each is segmented by SAM 2 and the resulting mask is
    // SUBTRACTED from the person mask.
    negativeBboxes: [],
  };
}

// Available preset names (populated once from /sam3d/api/presets).
let _presetNames = [];

// ---------------------------------------------------------------------------
// Object overlay (2D sprites cropped from input image, composited at confirm)
// ---------------------------------------------------------------------------

// NOTE: ``addObjectBtn`` is declared further down in the file alongside
// the other DOM refs; the click handler is wired in the same block as
// ``addPersonBtn.addEventListener`` to avoid a temporal-dead-zone ref.

async function _commitObjectFromBbox(bbox) {
  if (!state.inputBlob) return;
  addObjectHint.textContent = "SAM 2 で切り抜き中…";
  const fd = new FormData();
  fd.append("image", state.inputBlob, "input.png");
  fd.append("payload", JSON.stringify({
    bbox_xyxy: bbox,
    name: `obj${_objectCounter}`,
  }));
  const res = await fetch("/sam3d/api/object_crop", { method: "POST", body: fd });
  const j = await res.json();
  if (!res.ok) throw new Error(j.error || `HTTP ${res.status}`);
  // Pre-load the cropped PNG into a dataUrl so the sprite displays
  // immediately and the composite pass at confirm time has a usable
  // bitmap without an extra fetch.
  const dataUrl = await fetch(j.obj_url).then((r) => r.blob()).then((b) => new Promise((resolve) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.readAsDataURL(b);
  }));
  const o = _newObject();
  o.objUrl = j.obj_url;
  o.dataUrl = dataUrl;
  o.naturalWidth = j.natural_width;
  o.naturalHeight = j.natural_height;
  o.bbox = bbox;
  // Default scale: keep object at its natural size relative to the
  // input image (so it looks correct overlaid on the rendered scene
  // when the camera framing matches). User can tweak via the slider.
  o.scale = 1.0;
  state.objects.push(o);
  setActiveObject(o.id);
  addObjectHint.textContent = `✓ ${o.id} を追加`;
  setTimeout(() => {
    if (addObjectHint.textContent.startsWith("✓")) addObjectHint.textContent = "";
  }, 2000);
  refreshButtons();
}

// Re-run /object_crop for an object after its primary / additional /
// negative bbox set has changed. Debounced so a fast series of edits
// doesn't fire one request per drag.
const _objectRecropTimers = new Map();   // id → setTimeout handle
function scheduleObjectRecrop(id) {
  const prev = _objectRecropTimers.get(id);
  if (prev) clearTimeout(prev);
  _objectRecropTimers.set(id, setTimeout(() => {
    _objectRecropTimers.delete(id);
    _recropObject(id).catch((e) => {
      console.warn("[pose_editor_plus] object recrop failed:", e);
    });
  }, 350));
}

async function _recropObject(id) {
  const o = state.objects.find((q) => q.id === id);
  if (!o || !o.bbox || !state.inputBlob) return;
  const fd = new FormData();
  fd.append("image", state.inputBlob, "input.png");
  fd.append("payload", JSON.stringify({
    bbox_xyxy: o.bbox,
    additional_bboxes: o.additionalBboxes || [],
    negative_bboxes: o.negativeBboxes || [],
    name: o.id,
  }));
  const res = await fetch("/sam3d/api/object_crop", { method: "POST", body: fd });
  const j = await res.json();
  if (!res.ok) throw new Error(j.error || `HTTP ${res.status}`);
  const dataUrl = await fetch(j.obj_url).then((r) => r.blob()).then((b) => new Promise((resolve) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.readAsDataURL(b);
  }));
  o.objUrl = j.obj_url;
  o.dataUrl = dataUrl;
  o.naturalWidth = j.natural_width;
  o.naturalHeight = j.natural_height;
  rebuildObjectList();
  rebuildObjectOverlay();
  rebuildBboxOverlay();
}

function _resetObjectExtras(id) {
  const o = state.objects.find((q) => q.id === id);
  if (!o) return;
  o.additionalBboxes = [];
  o.negativeBboxes = [];
  rebuildObjectList();
  rebuildBboxOverlay();
  scheduleObjectRecrop(id);
}

function removeObject(id) {
  const idx = state.objects.findIndex((o) => o.id === id);
  if (idx < 0) return;
  state.objects.splice(idx, 1);
  if (state.activeObjectId === id) state.activeObjectId = state.objects[0]?.id || null;
  rebuildObjectList();
  rebuildObjectOverlay();
  rebuildBboxOverlay();
}

function setActiveObject(id) {
  // No-op when the active object hasn't changed AND no person is
  // active. Skipping the heavy DOM rebuild here is critical for
  // pointerdown handlers on the sprite: rebuilding ``rebuildObjectOverlay``
  // detaches the very element that just captured the pointer, which
  // silently kills the in-progress drag.
  if (state.activeObjectId === id && !state.activeId) return;
  state.activeObjectId = id;
  // Mutual exclusion with persons (see setActive). When editing an
  // object, drags target its mask additions / subtractions instead of
  // any person's.
  state.activeId = null;
  rebuildObjectList();
  rebuildObjectOverlay();
  rebuildPersonList();
  rebuildBboxOverlay();
  _refreshViewportToolbar();
}

function rebuildObjectList() {
  objectListEl.innerHTML = "";
  if (state.objects.length === 0) {
    const m = document.createElement("div");
    m.className = "muted";
    m.textContent = "「+ Object」で入力画像から切り抜き追加";
    objectListEl.appendChild(m);
    return;
  }
  for (const o of state.objects) {
    const card = document.createElement("div");
    card.className = "object-card" + (o.id === state.activeObjectId ? " active" : "");
    card.addEventListener("click", () => setActiveObject(o.id));

    const head = document.createElement("div");
    head.className = "head";
    const name = document.createElement("div");
    name.style.fontWeight = "600";
    name.textContent = o.id;
    const del = document.createElement("button");
    del.className = "btn btn-danger";
    del.textContent = "削除";
    del.style.cssText = "padding: 2px 8px; pointer-events: auto;";
    del.addEventListener("click", (ev) => {
      ev.stopPropagation();
      removeObject(o.id);
    });
    head.appendChild(name);
    head.appendChild(del);
    card.appendChild(head);

    // Bbox readout (matches person card layout).
    if (o.bbox) {
      const bboxRO = document.createElement("div");
      bboxRO.className = "bbox-readout";
      bboxRO.style.cssText = "font-size: 10px; color: var(--muted);";
      const [x1, y1, x2, y2] = o.bbox.map((v) => Math.round(v));
      bboxRO.textContent = `bbox: (${x1},${y1})–(${x2},${y2})  ${x2 - x1}×${y2 - y1}px`;
      card.appendChild(bboxRO);
    }

    if (o.dataUrl) {
      const thumb = document.createElement("img");
      thumb.className = "thumb";
      thumb.src = o.dataUrl;
      card.appendChild(thumb);
    }

    // Extras summary tag (count badge only — individual × delete + bulk
    // reset are now in the image overlay; bulk reset removed per spec).
    const addCnt = (o.additionalBboxes?.length || 0);
    const negCnt = (o.negativeBboxes?.length || 0);
    if (addCnt > 0 || negCnt > 0) {
      const extras = document.createElement("div");
      extras.className = "muted";
      extras.style.fontSize = "11px";
      const tags = [];
      if (addCnt > 0) tags.push(`+${addCnt} 正`);
      if (negCnt > 0) tags.push(`−${negCnt} 除外`);
      extras.textContent = `[${tags.join(", ")}]`;
      card.appendChild(extras);
    } else if (o.id === state.activeObjectId) {
      const hint = document.createElement("div");
      hint.className = "muted";
      hint.style.fontSize = "10px";
      hint.textContent = "ビュー上でドラッグ移動 / 隅でリサイズ / ホイールで回転";
      card.appendChild(hint);
    }

    // Transparency slider. The slider's 0–100 values map to "how
    // transparent" the sprite is, so 0 = fully opaque (default), 100 =
    // invisible. Internally we store CSS-style ``opacity`` (1 −
    // transparency) on the object.
    {
      const row = document.createElement("div");
      row.className = "row";
      const lbl = document.createElement("span");
      lbl.style.cssText = "font-size: 11px; color: var(--muted); min-width: 60px;";
      lbl.textContent = "透明度:";
      const rng = document.createElement("input");
      rng.type = "range";
      rng.min = "0"; rng.max = "100"; rng.step = "1";
      rng.style.flex = "1 1 auto";
      const transparencyPct = Math.round((1 - (o.opacity ?? 1)) * 100);
      rng.value = String(transparencyPct);
      const num = document.createElement("input");
      num.type = "number";
      num.min = "0"; num.max = "100"; num.step = "1";
      num.style.cssText = "width: 60px; background: #1a1a1a; border: 1px solid var(--border); color: var(--text); padding: 2px 4px; border-radius: 3px;";
      num.value = rng.value;
      const commit = (v) => {
        const x = Math.max(0, Math.min(100, parseFloat(v) || 0));
        rng.value = String(x);
        num.value = String(Math.round(x));
        o.opacity = 1 - x / 100;
        rebuildObjectOverlay();
      };
      rng.addEventListener("input", (ev) => { ev.stopPropagation(); commit(rng.value); });
      num.addEventListener("input", (ev) => { ev.stopPropagation(); commit(num.value); });
      const pct = document.createElement("span");
      pct.className = "unit-suffix";
      pct.textContent = "%";
      row.appendChild(lbl); row.appendChild(rng); row.appendChild(num); row.appendChild(pct);
      card.appendChild(row);
    }

    objectListEl.appendChild(card);
  }
}

function rebuildObjectOverlay() {
  objectOverlay.innerHTML = "";
  if (state.objects.length === 0) return;
  const r = viewportEl.getBoundingClientRect();
  for (const o of state.objects) {
    if (!o.dataUrl) continue;
    const wrap = document.createElement("div");
    const isActive = o.id === state.activeObjectId;
    wrap.className = "object-sprite" + (isActive ? " active" : "");
    wrap.dataset.objectId = o.id;
    const cssW = o.naturalWidth * o.scale;
    const cssH = o.naturalHeight * o.scale;
    const cx = o.posX * r.width;
    const cy = o.posY * r.height;
    wrap.style.left = `${cx - cssW / 2}px`;
    wrap.style.top  = `${cy - cssH / 2}px`;
    wrap.style.width = `${cssW}px`;
    wrap.style.height = `${cssH}px`;
    wrap.style.transform = `rotate(${o.rotationDeg}deg)`;
    wrap.style.opacity = String(Math.max(0, Math.min(1, o.opacity ?? 1)));
    const im = document.createElement("img");
    im.src = o.dataUrl;
    im.alt = o.id;
    im.draggable = false;
    wrap.appendChild(im);

    // Corner resize handles — only meaningful when the sprite is the
    // active object. CSS hides them otherwise.
    if (isActive) {
      for (const corner of ["tl", "tr", "bl", "br"]) {
        const h = document.createElement("div");
        h.className = `corner-handle ${corner}`;
        h.dataset.corner = corner;
        wrap.appendChild(h);
      }
    }

    _attachObjectDrag(wrap, o);
    objectOverlay.appendChild(wrap);
  }
}

// Apply the current state of an object to its sprite element WITHOUT
// rebuilding the DOM. This is the hot path for live drag updates:
// rebuilding the whole overlay would detach the captured pointer.
function _applySpriteStyles(o, wrap) {
  if (!wrap) return;
  const r = viewportEl.getBoundingClientRect();
  const cssW = o.naturalWidth * o.scale;
  const cssH = o.naturalHeight * o.scale;
  const cx = o.posX * r.width;
  const cy = o.posY * r.height;
  wrap.style.left = `${cx - cssW / 2}px`;
  wrap.style.top  = `${cy - cssH / 2}px`;
  wrap.style.width = `${cssW}px`;
  wrap.style.height = `${cssH}px`;
  wrap.style.transform = `rotate(${o.rotationDeg}deg)`;
  wrap.style.opacity = String(Math.max(0, Math.min(1, o.opacity ?? 1)));
}

function _attachObjectDrag(wrap, o) {
  let drag = null;
  // Pointerdown on the sprite — corner handles → resize, body → move.
  wrap.addEventListener("pointerdown", (ev) => {
    ev.stopPropagation();
    if (ev.button !== 0) return;
    setActiveObject(o.id);   // no-op when already active (see setActiveObject)
    const r = viewportEl.getBoundingClientRect();
    const corner = ev.target?.dataset?.corner;
    if (corner) {
      const cxAbs = (o.posX * r.width) + r.left;
      const cyAbs = (o.posY * r.height) + r.top;
      const startDist = Math.hypot(ev.clientX - cxAbs, ev.clientY - cyAbs);
      drag = {
        kind: "resize",
        pointerId: ev.pointerId,
        cxAbs, cyAbs,
        startDist: Math.max(8, startDist),
        startScale: o.scale,
      };
    } else {
      drag = {
        kind: "move",
        pointerId: ev.pointerId,
        startClientX: ev.clientX,
        startClientY: ev.clientY,
        startCx: o.posX * r.width,
        startCy: o.posY * r.height,
        vpW: r.width, vpH: r.height,
      };
    }
    wrap.classList.add("dragging");
    try { wrap.setPointerCapture(ev.pointerId); } catch (_e) {}
  });
  wrap.addEventListener("pointermove", (ev) => {
    if (!drag) return;
    if (drag.kind === "resize") {
      const dist = Math.hypot(ev.clientX - drag.cxAbs, ev.clientY - drag.cyAbs);
      const ratio = dist / drag.startDist;
      const newScale = Math.max(0.05, Math.min(10, drag.startScale * ratio));
      o.scale = newScale;
    } else {
      const dx = ev.clientX - drag.startClientX;
      const dy = ev.clientY - drag.startClientY;
      const newCx = drag.startCx + dx;
      const newCy = drag.startCy + dy;
      o.posX = Math.max(-1, Math.min(2, newCx / Math.max(1, drag.vpW)));
      o.posY = Math.max(-1, Math.min(2, newCy / Math.max(1, drag.vpH)));
    }
    // Update only this sprite's styles — keep the DOM tree intact so
    // setPointerCapture stays bound.
    _applySpriteStyles(o, wrap);
  });
  const endDrag = (ev) => {
    if (!drag) return;
    try { wrap.releasePointerCapture(drag.pointerId); } catch (_e) {}
    drag = null;
    wrap.classList.remove("dragging");
    // Re-sync the sidebar list now that the drag is done (counts /
    // sliders may need to reflect the final scale, opacity, etc.).
    rebuildObjectList();
  };
  wrap.addEventListener("pointerup", endDrag);
  wrap.addEventListener("pointercancel", endDrag);

  // Wheel rotates the active object. We always activate first so a
  // wheel over an inactive sprite still rotates it (matches the
  // sprite-as-handle interaction model).
  wrap.addEventListener("wheel", (ev) => {
    if (state.view !== "3d" && state.view !== "image") return;
    ev.preventDefault();
    setActiveObject(o.id);
    const step = ev.shiftKey ? 1 : 5;   // hold Shift for fine rotation
    const delta = ev.deltaY > 0 ? -step : step;
    o.rotationDeg = Math.max(-180, Math.min(180, (o.rotationDeg || 0) + delta));
    _applySpriteStyles(o, wrap);
  }, { passive: false });
}

// ---------------------------------------------------------------------------
// SAM 2 preview (per-person trimmed PNG) plumbing. Each bbox change
// schedules a debounced refresh — the backend re-runs SAM 2 over ALL
// current bboxes (the segmenter uses other-person bboxes as negative
// prompts, so adding a person can shift previous masks too).
let _sam2PreviewTimer = null;
let _sam2PreviewInflight = false;

function scheduleSam2Preview() {
  clearTimeout(_sam2PreviewTimer);
  _sam2PreviewTimer = setTimeout(refreshSam2Preview, 350);
}

async function refreshSam2Preview() {
  if (!state.inputBlob) return;
  const persons = state.persons.filter((p) => p.bbox);
  if (persons.length === 0) return;
  if (_sam2PreviewInflight) {
    // Drop this attempt and re-arm — the in-flight call will return soon
    // and the trailing scheduleSam2Preview will pick up the latest state.
    scheduleSam2Preview();
    return;
  }
  _sam2PreviewInflight = true;
  setStatus("SAM 2 preview…");
  try {
    const fd = new FormData();
    fd.append("image", state.inputBlob, "input.png");
    fd.append("payload", JSON.stringify({
      persons: persons.map((p) => ({
        id: p.id,
        bbox_xyxy: p.bbox,
        additional_bboxes: p.additionalBboxes || [],
        negative_bboxes: p.negativeBboxes || [],
      })),
    }));
    const res = await fetch("/sam3d/api/sam2_preview", { method: "POST", body: fd });
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || `HTTP ${res.status}`);
    for (const pp of j.per_person) {
      const slot = state.persons.find((q) => q.id === pp.id);
      if (!slot) continue;
      slot.trimmedUrl = pp.trimmed_url || null;
      slot.maskScore = pp.mask_score || 0;
    }
    rebuildPersonList();
    setStatus("ready");
  } catch (e) {
    console.warn("sam2_preview failed:", e);
    setStatus("preview 失敗 — bbox crop で代用");
  } finally {
    _sam2PreviewInflight = false;
  }
}

async function _loadPresetList() {
  try {
    const r = await fetch("/sam3d/api/presets");
    const j = await r.json();
    if (Array.isArray(j.presets)) _presetNames = j.presets.map((p) => p.name || p);
  } catch (e) {
    console.warn("preset list load failed:", e);
  }
}

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

const fileInputEl     = $("file-input");
const fileDropEl      = $("file-drop");
const fileLabelEl     = $("file-label");
const previewEl       = $("preview");
const imageInfoEl     = $("image-info");
const addPersonBtn    = $("add-person-btn");
const addHintEl       = $("add-person-hint");
const personListEl    = $("person-list");
const runBtn          = $("run-btn");
const runInfoEl       = $("run-info");
const statusEl        = $("status");
const confirmBtn      = $("confirm-btn");
const viewportEl      = $("viewport");
const overlayEl       = $("viewport-overlay");
const canvasEl        = $("three-canvas");
const bboxOverlay     = $("bbox-overlay");
const bboxRectActive  = $("bbox-rect-active");
const tab3dBtn        = $("tab-3d");
const tabImageBtn     = $("tab-image");
const toggleImgBtn    = $("toggle-img-btn");
const imageDisplay    = $("image-display");
const imageDisplayImg = $("image-display-img");
const gizmoTranslateBtn = $("gizmo-translate-btn");
const gizmoRotateBtn    = $("gizmo-rotate-btn");
// IK mode is now triggered from the per-person sidebar card (no
// dedicated viewport-toolbar button); this ref is kept null-safe in
// case a stale HTML still has the button.
const ikModeBtn         = $("ik-mode-btn");   // may be null

const resetTranslateBtn = $("reset-translate-btn");
const resetRotateBtn    = $("reset-rotate-btn");
const negBboxHint       = $("neg-bbox-hint");
const addObjectBtn      = $("add-object-btn");
const addObjectHint     = $("add-object-hint");
const objectListEl      = $("object-list");
const objectOverlay     = $("object-overlay");
const rangeSelectBtn    = $("range-select-btn-plus");
const rangeResetBtn     = $("range-reset-btn-plus");
const rangeOverlayEl    = $("range-overlay-plus");
const rangeRectEl       = $("range-rect-plus");
const bgColorBtn        = $("bg-color-btn-plus");
const bgColorSwatch     = $("bg-color-swatch-plus");
const bgColorInput      = $("bg-color-input-plus");

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function setStatus(text) { statusEl.textContent = text; }

function dataUrlToBlob(dataUrl) {
  if (!dataUrl) return null;
  const m = dataUrl.match(/^data:([^;]+);base64,(.+)$/);
  if (!m) return null;
  const bytes = atob(m[2]);
  const buf = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i);
  return new Blob([buf], { type: m[1] });
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error || new Error("file read failed"));
    r.readAsDataURL(file);
  });
}

async function decodeImage(file) {
  const dataUrl = await readFileAsDataUrl(file);
  const dim = await new Promise((resolve) => {
    const im = new Image();
    im.onload = () => resolve({ w: im.naturalWidth, h: im.naturalHeight });
    im.onerror = () => resolve({ w: 0, h: 0 });
    im.src = dataUrl;
  });
  return { dataUrl, width: dim.w, height: dim.h, blob: dataUrlToBlob(dataUrl) };
}

function metersFromDisplay(value, unit) {
  const v = parseFloat(value);
  if (!isFinite(v)) return null;
  if (unit === "cm")   return v * 0.01;
  if (unit === "inch") return v * 0.0254;
  return v;
}

function displayFromMeters(meters, unit) {
  if (unit === "cm")   return Math.round(meters / 0.01 * 10) / 10;     // 1 dp
  if (unit === "inch") return Math.round(meters / 0.0254 * 100) / 100; // 2 dp
  return meters;
}

function unitSuffix() { return state.unit.display_unit; }

// ---------------------------------------------------------------------------
// Three.js viewer
// ---------------------------------------------------------------------------

const viewer = (() => {
  const renderer = new THREE.WebGLRenderer({
    canvas: canvasEl, antialias: true, alpha: true, preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x2a2c30, 1);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 200);
  camera.position.set(0, 1.2, 4.5);

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dl = new THREE.DirectionalLight(0xffffff, 1.1);
  dl.position.set(3, 5, 3);
  scene.add(dl);
  const fl = new THREE.DirectionalLight(0xaaccff, 0.3);
  fl.position.set(-3, 2, -4);
  scene.add(fl);

  const grid = new THREE.GridHelper(8, 32, 0x555555, 0x333333);
  scene.add(grid);

  const controls = new OrbitControls(camera, canvasEl);
  controls.target.set(0, 1.0, 0);
  controls.update();

  // Per-person anchors: one Object3D per detected person. The body mesh
  // sits inside as a child whose own vertices are body-local (hip at the
  // origin). The anchor's position / rotation is the user-applied
  // transform on top of the predicted hip world placement.
  //
  // anchors[] entries: { id, anchor, mesh, hipWorld:[x,y,z], baseColor }.
  let anchors = [];

  // TransformControls — attaches to either an IK target (pose-edit mode)
  // or the active person's body anchor (translate / rotate gizmo). The
  // ``change`` and ``dragging-changed`` listeners that route between
  // those two modes are installed in the pose-edit section below; this
  // is just the constructor call so the instance is in scope.
  const tcontrols = new TransformControls(camera, canvasEl);
  tcontrols.setSize(0.85);
  tcontrols.setMode("translate");
  // ``_syncRaf`` throttles the body-anchor transform sync (used by the
  // pose-edit ``change`` handler when poseEdit.active is false).
  let _syncRaf = 0;
  // TransformControls in three r170 returns the gizmo as a helper via
  // ``getHelper()``; older builds expose the controls themselves as the
  // scene-graph node. Use whichever is available so the gizmo actually
  // renders (a missing scene.add() makes the whole feature invisible).
  const tcHelper =
    typeof tcontrols.getHelper === "function" ? tcontrols.getHelper() : tcontrols;
  scene.add(tcHelper);

  // The owner registers a callback to receive transform-change ticks.
  let _onTransformChanged = null;
  function setOnTransformChanged(cb) { _onTransformChanged = cb; }

  function resize() {
    const w = canvasEl.clientWidth;
    const h = canvasEl.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);

  let _dirty = true;
  controls.addEventListener("change", () => { _dirty = true; });

  (function tick() {
    controls.update();
    if (_dirty) {
      renderer.render(scene, camera);
      _dirty = false;
    }
    requestAnimationFrame(tick);
  })();

  function clearAnchors() {
    // Only detach the gizmo when it was NOT bound to the pose-edit IK
    // target — an IK / Hips drag mid-render must stay live so the user
    // can keep pulling after the mesh updates underneath them.
    if (!poseEdit.active) tcontrols.detach();
    for (const entry of anchors) {
      if (entry.anchor.parent) entry.anchor.parent.remove(entry.anchor);
      entry.mesh.geometry?.dispose?.();
      if (Array.isArray(entry.mesh.material)) {
        entry.mesh.material.forEach((m) => m?.dispose?.());
      } else {
        entry.mesh.material?.dispose?.();
      }
    }
    anchors = [];
    _dirty = true;
  }

  function frame(box3) {
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box3.getCenter(center);
    box3.getSize(size);
    const fov = camera.fov * Math.PI / 180;
    const distH = (Math.max(size.y, 0.5) / 2) / Math.tan(fov / 2);
    const distW = (Math.max(size.x, 0.5) / 2)
                  / (Math.max(camera.aspect, 0.1) * Math.tan(fov / 2));
    const dist = Math.max(distH, distW) * 1.4 + Math.max(size.z, 0.5);
    controls.target.copy(center);
    camera.position.set(center.x, center.y, center.z + dist);
    camera.lookAt(center);
    controls.update();
    _dirty = true;
  }

  const PALETTE = [0xb1cfff, 0xffcfa1, 0xb6f0c1, 0xf5b6c8, 0xd9c8f5, 0xc8e8a0, 0xa0e0e8, 0xe0e0a8];

  // Build per-person anchored meshes. ``perPerson[i]`` carries the OBJ
  // face range and the predicted hip-world position; vertices in the
  // returned merged geometry sit at scene-world coords (cam_t-baked) so
  // we shift them back to body-local before parenting under the anchor.
  //
  // Group centring + ground snap: SAM 3D Body's ``cam_t`` can place the
  // baked-in world position several metres off origin (e.g. the focal
  // depth along Z), which makes the bodies float far from the grid and
  // feels disorienting on first load. We post-process by subtracting the
  // group's XZ centroid and the overall min-Y so the cohort lands centred
  // on the grid with feet at y=0, while preserving each body's relative
  // offset from the others (the layout from the source photo is kept).
  // Track the camera-fit fingerprint so subsequent renders for the SAME
  // cohort (same person ids) don't snap the orbit camera away from the
  // angle the user picked. Mirrors the single Pose Editor's behaviour
  // where the camera frames once per inference job and stays put for
  // pose / IK edits.
  let _framedFingerprint = null;

  // Cohort-layout cache. Re-rendering the SAME cohort (same set of person
  // IDs) MUST keep every anchor at the same scene-world position — the
  // bbox of a posed mesh shifts after IK, so recomputing localCenter /
  // group-centring / ground-snap each render translates the whole body
  // even when the user only moved a hand. Mirrors the single editor's
  // ``_meshFitOffset`` mechanism. Invalidated on input-image change and
  // on fresh full inference (cohort identity reset).
  //
  // Shape: { fingerprint: "p0|p1|...",
  //          perPerson: Map<id, {
  //            localCenter: [x,y,z],          // bbox centre, anchor pivot
  //            hipWorldCentered: [x,y,z],     // post group-centring + ground-snap
  //            naturalPos: [x,y,z],           // anchor.position at translate=0
  //          }> }
  let _cohortLayoutCache = null;

  async function loadObjPerPerson(url, perPerson, persons, activePersonId) {
    // Preserve the pose-edit overlay across the rebuild so in-flight IK
    // drags + gizmo attachments survive a sub-render. Detach the root
    // from the old anchor here; we re-parent it to the new active
    // anchor at the end.
    const carriedPoseRoot = (poseEdit.active && poseEdit.root) ? poseEdit.root : null;
    if (carriedPoseRoot && carriedPoseRoot.parent) {
      carriedPoseRoot.parent.remove(carriedPoseRoot);
    }
    clearAnchors();
    const loader = new OBJLoader();
    const root = await new Promise((resolve, reject) => {
      loader.load(url, resolve, undefined, reject);
    });

    let primaryMesh = null;
    root.traverse((o) => { if (o.isMesh && !primaryMesh) primaryMesh = o; });
    if (!primaryMesh) throw new Error("OBJ contained no mesh");
    const positions = primaryMesh.geometry.attributes.position.array;

    const overall = new THREE.Box3();

    perPerson.forEach((pp, i) => {
      const [fStart, fEnd] = pp.face_range;
      const [hwx, hwy, hwz] = pp.hip_world || [0, 0, 0];
      // Per-face groups span 3 vertices each in non-indexed BufferGeometry,
      // 9 floats per face (3 vertices × xyz).
      const startFloat = fStart * 9;
      const endFloat = fEnd * 9;
      const slice = new Float32Array(endFloat - startFloat);
      for (let j = 0; j < slice.length; j += 3) {
        slice[j]     = positions[startFloat + j]     - hwx;
        slice[j + 1] = positions[startFloat + j + 1] - hwy;
        slice[j + 2] = positions[startFloat + j + 2] - hwz;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(slice, 3));
      geo.computeVertexNormals();
      // Mesh bounding-box centre relative to the hip. We park the anchor
      // there (instead of at the hip) so the gizmo's translate handles
      // and rotation rings appear at — and pivot around — the body's
      // visual centroid rather than the hip joint at the foot. The mesh
      // is shifted by -localCenter inside the anchor so the rendered
      // body stays in the same scene-world position when transform=0.
      geo.computeBoundingBox();
      const bb = geo.boundingBox;
      const localCenter = new THREE.Vector3();
      bb.getCenter(localCenter);

      const baseColor = PALETTE[i % PALETTE.length];
      const mat = new THREE.MeshStandardMaterial({
        color: baseColor, roughness: 0.65, metalness: 0.05, flatShading: false,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(localCenter).multiplyScalar(-1);

      const anchor = new THREE.Object3D();
      anchor.name = `person_anchor_${pp.id}`;
      anchor.userData.personId = pp.id;
      anchor.add(mesh);

      anchors.push({
        id: pp.id, anchor, mesh,
        hipWorld: [hwx, hwy, hwz],
        localCenter: [localCenter.x, localCenter.y, localCenter.z],
        baseColor,
      });
    });

    // Cohort fingerprint = sorted set of person IDs. Same cohort across
    // re-renders → reuse the cached layout so IK edits don't translate
    // the body by the bbox-centre delta. Different cohort (add/remove
    // person, fresh inference) → recompute and refresh the cache.
    const layoutFingerprint = anchors.map((e) => e.id).slice().sort().join("|");
    const cacheHit = _cohortLayoutCache
      && _cohortLayoutCache.fingerprint === layoutFingerprint
      && anchors.every((e) => _cohortLayoutCache.perPerson.has(e.id));

    if (cacheHit) {
      // Reuse cached layout. Fresh ``localCenter`` from the new mesh's
      // bbox is discarded; mesh.position is shifted by the cached offset
      // so the body stays in its previous scene-world spot regardless
      // of the bbox shift caused by the new pose.
      for (const e of anchors) {
        const cached = _cohortLayoutCache.perPerson.get(e.id);
        e.localCenter = cached.localCenter.slice();
        e.hipWorld    = cached.hipWorldCentered.slice();
        e.mesh.position.set(-cached.localCenter[0], -cached.localCenter[1], -cached.localCenter[2]);

        const personState = persons.find((q) => q.id === e.id);
        const tr = personState?.transform || { translate: [0, 0, 0], rotate_deg: [0, 0, 0] };
        e.anchor.position.set(
          cached.naturalPos[0] + (tr.translate[0] || 0),
          cached.naturalPos[1] + (tr.translate[1] || 0),
          cached.naturalPos[2] + (tr.translate[2] || 0),
        );
        e.anchor.rotation.set(
          THREE.MathUtils.degToRad(tr.rotate_deg[0] || 0),
          THREE.MathUtils.degToRad(tr.rotate_deg[1] || 0),
          THREE.MathUtils.degToRad(tr.rotate_deg[2] || 0),
          "XYZ",
        );

        scene.add(e.anchor);
        overall.expandByObject(e.anchor);
      }
    } else {
      // Group centre + ground snap (only for fresh cohorts).
      let cx = 0, cz = 0;
      for (const e of anchors) {
        cx += e.hipWorld[0] + e.localCenter[0];
        cz += e.hipWorld[2] + e.localCenter[2];
      }
      cx /= Math.max(anchors.length, 1);
      cz /= Math.max(anchors.length, 1);

      let minY = Infinity;
      for (const e of anchors) {
        e.anchor.position.set(
          e.hipWorld[0] - cx + e.localCenter[0],
          e.hipWorld[1]      + e.localCenter[1],
          e.hipWorld[2] - cz + e.localCenter[2],
        );
        e.anchor.updateMatrixWorld(true);
        const bb = new THREE.Box3().setFromObject(e.anchor);
        if (bb.min.y < minY) minY = bb.min.y;
      }
      if (!isFinite(minY)) minY = 0;

      const newCache = { fingerprint: layoutFingerprint, perPerson: new Map() };
      for (const e of anchors) {
        const newHipY = e.hipWorld[1] - minY;
        const newHipX = e.hipWorld[0] - cx;
        const newHipZ = e.hipWorld[2] - cz;
        e.hipWorld = [newHipX, newHipY, newHipZ];

        const naturalPos = [
          newHipX + e.localCenter[0],
          newHipY + e.localCenter[1],
          newHipZ + e.localCenter[2],
        ];
        newCache.perPerson.set(e.id, {
          localCenter:      e.localCenter.slice(),
          hipWorldCentered: e.hipWorld.slice(),
          naturalPos:       naturalPos.slice(),
        });

        const personState = persons.find((q) => q.id === e.id);
        const tr = personState?.transform || { translate: [0, 0, 0], rotate_deg: [0, 0, 0] };
        e.anchor.position.set(
          naturalPos[0] + (tr.translate[0] || 0),
          naturalPos[1] + (tr.translate[1] || 0),
          naturalPos[2] + (tr.translate[2] || 0),
        );
        e.anchor.rotation.set(
          THREE.MathUtils.degToRad(tr.rotate_deg[0] || 0),
          THREE.MathUtils.degToRad(tr.rotate_deg[1] || 0),
          THREE.MathUtils.degToRad(tr.rotate_deg[2] || 0),
          "XYZ",
        );

        scene.add(e.anchor);
        overall.expandByObject(e.anchor);
      }
      _cohortLayoutCache = newCache;
    }

    // Only auto-frame on first load or when the cohort changes — repeated
    // pose / IK / slider re-renders MUST NOT yank the camera. The user
    // expects the orbit they picked to stay put across render cycles
    // (single-editor parity).
    const fingerprint = anchors.map((e) => e.id).join("|");
    if (anchors.length > 0 && fingerprint !== _framedFingerprint) {
      frame(overall);
      _framedFingerprint = fingerprint;
    }

    // Re-attach the pose-edit overlay to the new active anchor. Bones'
    // anchor-local positions are unchanged — the user's IK manipulations
    // are preserved across renders.
    if (carriedPoseRoot) {
      const targetEntry = anchors.find((a) => a.id === activePersonId)
        || anchors.find((a) => a.id === poseEdit.activePersonId)
        || anchors[0];
      if (targetEntry) {
        targetEntry.anchor.add(carriedPoseRoot);
        poseEdit.activePersonId = targetEntry.id;
      } else {
        // No surviving anchor — pose-edit can't continue, exit cleanly.
        carriedPoseRoot.traverse?.((o) => {
          if (o.isMesh || o.isLine) o.geometry?.dispose?.();
        });
        poseEdit.bones = [];
        poseEdit.byName = new Map();
        poseEdit.selected = null;
        poseEdit.root = null;
        poseEdit.active = false;
      }
    }
    _dirty = true;
  }

  function resetFramedFingerprint() { _framedFingerprint = null; }

  function resetCohortLayoutCache() { _cohortLayoutCache = null; }

  function getAnchorEntry(personId) {
    if (!personId) return null;
    return anchors.find((a) => a.id === personId) || null;
  }

  function setActiveAnchor(personId /* string | null */) {
    if (!personId) {
      tcontrols.detach();
      _dirty = true;
      return;
    }
    const entry = anchors.find((a) => a.id === personId);
    if (!entry) {
      tcontrols.detach();
      _dirty = true;
      return;
    }
    tcontrols.attach(entry.anchor);
    _dirty = true;
  }

  function setGizmoMode(mode /* "translate" | "rotate" */) {
    tcontrols.setMode(mode === "rotate" ? "rotate" : "translate");
    _dirty = true;
  }

  function setSoloVisibility(activeIndex /* number | null */) {
    anchors.forEach((entry, i) => {
      entry.anchor.visible = (activeIndex === null) || (i === activeIndex);
    });
    _dirty = true;
  }

  function captureComposite(bgColor) {
    const prevColor = renderer.getClearColor(new THREE.Color());
    const prevAlpha = renderer.getClearAlpha();
    const prevGrid = grid.visible;
    const prevGizmo = tcHelper.visible;
    // 撮影画像にポーズ編集オーバーレイ (ボーンハンドル + 骨ライン) が
    // 写り込まないよう、root を一時非表示。
    const prevPoseRootVis = poseEdit.root ? poseEdit.root.visible : null;
    grid.visible = false;
    tcHelper.visible = false;
    if (poseEdit.root) poseEdit.root.visible = false;
    if (bgColor === null) {
      renderer.setClearColor(0x000000, 0);
    } else {
      renderer.setClearColor(bgColor, 1);
    }
    try {
      renderer.render(scene, camera);
      const dataUrl = canvasEl.toDataURL("image/png");
      return { dataUrl, width: canvasEl.width, height: canvasEl.height };
    } finally {
      renderer.setClearColor(prevColor, prevAlpha);
      grid.visible = prevGrid;
      tcHelper.visible = prevGizmo;
      if (poseEdit.root && prevPoseRootVis !== null) {
        poseEdit.root.visible = prevPoseRootVis;
      }
      _dirty = true;
    }
  }

  // Read the current anchor's transform (relative to the body's mesh
  // centre, which is the gizmo pivot) and return ``{translate, rotate_deg}``.
  // Returns null if no active anchor.
  function readActiveTransform() {
    if (!tcontrols.object) return null;
    const a = tcontrols.object;
    const entry = anchors.find((e) => e.anchor === a);
    if (!entry) return null;
    const [hwx, hwy, hwz] = entry.hipWorld;
    const [lcx, lcy, lcz] = entry.localCenter;
    return {
      id: entry.id,
      translate: [
        a.position.x - hwx - lcx,
        a.position.y - hwy - lcy,
        a.position.z - hwz - lcz,
      ],
      rotate_deg: [
        THREE.MathUtils.radToDeg(a.rotation.x),
        THREE.MathUtils.radToDeg(a.rotation.y),
        THREE.MathUtils.radToDeg(a.rotation.z),
      ],
    };
  }

  // Push a transform value into the named anchor (used by the sidebar
  // sliders to drive the gizmo). ``translate`` is the offset from the
  // body's mesh centre (the gizmo pivot), not from the hip.
  function writeAnchorTransform(personId, translate, rotate_deg) {
    const entry = anchors.find((a) => a.id === personId);
    if (!entry) return;
    const [hwx, hwy, hwz] = entry.hipWorld;
    const [lcx, lcy, lcz] = entry.localCenter;
    entry.anchor.position.set(
      hwx + lcx + (translate[0] || 0),
      hwy + lcy + (translate[1] || 0),
      hwz + lcz + (translate[2] || 0),
    );
    entry.anchor.rotation.set(
      THREE.MathUtils.degToRad(rotate_deg[0] || 0),
      THREE.MathUtils.degToRad(rotate_deg[1] || 0),
      THREE.MathUtils.degToRad(rotate_deg[2] || 0),
      "XYZ",
    );
    _dirty = true;
  }

  function getAnchorCount() { return anchors.length; }

  // =====================================================================
  // Pose-edit / IK overlay — direct port of the single Pose Editor's
  // ``poseEdit`` system (``editor_core.js``).
  //
  // Multi-person adaptation:
  //   - ``poseEdit.root`` parents under the ACTIVE PERSON's anchor (instead
  //     of the single editor's ``meshGroup``).
  //   - Hips drag adds the gizmo delta to the active anchor's position
  //     (instead of meshGroup.position).
  //   - Switching active person: ``exitPoseEdit()`` then ``enterPoseEdit()``
  //     with the new skeleton + anchor.
  //   - Mesh re-render across IK drags: carry-across (see loadObjPerPerson).
  // =====================================================================

  const poseEdit = {
    active: false,
    root: null,
    bones: [],
    byName: new Map(),
    selected: null,
    activePersonId: null,
    baseHandleScale: 1.0,
    lastAnchorScale: 1.0,
    onBoneChange:    null,   // (boneName, eulerRad | null)
    onBonePick:      null,   // (boneName)
    onDragStart:     null,
    onDragEnd:       null,
    onIkChainChange: null,   // ([{boneName, eulerRad | null}, ...])
    onHipsTranslate: null,   // (dx, dy, dz) — multi-only
  };

  // Finger handles are shrunk to 1/8 of body-bone size — cluster of five
  // fingers × four joints per hand makes full-size spheres unpickable.
  const _FINGER_HANDLE_SCALE_RATIO = 1 / 8;
  const _FINGER_BONE_RE = /^(Left|Right)Hand(Thumb|Index|Middle|Ring|Pinky)\d$/;
  function _isFingerBoneName(name) { return _FINGER_BONE_RE.test(name || ""); }

  const _poseLineMat            = new THREE.LineBasicMaterial({ color: 0x33aaff, transparent: true, opacity: 0.85, depthTest: false });
  const _poseHandleMat          = new THREE.MeshBasicMaterial({ color: 0x33aaff, transparent: true, opacity: 0.9,  depthTest: false });
  const _poseHandleSelectedMat  = new THREE.MeshBasicMaterial({ color: 0xff5252, transparent: true, opacity: 0.95, depthTest: false });
  const _poseHandleGeom         = new THREE.SphereGeometry(1, 12, 10);

  // Invisible Object3D the gizmo grabs. World-space (parented to scene).
  const _ikTarget = new THREE.Object3D();
  _ikTarget.name = "_ik_target";
  scene.add(_ikTarget);

  const _HIPS_BONE_NAME = "Hips";
  const _hipsDrag = { active: false, lastTargetWorld: new THREE.Vector3() };

  const _IK_MAX_CHAIN_LEN = 3;
  const _IK_ITERATIONS    = 12;
  const _IK_ANGLE_TOLERANCE = 0.001;
  const _IK_DAMPING       = 0.5;
  const _ikDrag = { active: false, bone: null, chain: [] };

  // Bones that act as IK ceilings: chain may include them but never their
  // parents. Stops shoulder / spine / hips from rotating when the user
  // IK-drags a hand or foot. Exception: chain[0] === ceiling bone releases
  // the ceiling so dragging the upper arm still rotates shoulder + spine.
  const _IK_CEILING_BONES = new Set([
    "LeftArm", "RightArm",
    "LeftUpLeg", "RightUpLeg",
  ]);

  // Hinge bones (knees / elbows): natural-bend world-X sign.
  // Wrong sign → hyperextension → snap localDelta to identity.
  const _IK_HINGE_NATURAL_X = {
    "LeftLeg":      +1,
    "RightLeg":     +1,
    "LeftForeArm":  -1,
    "RightForeArm": -1,
  };
  const _ikHingeTmpInv   = new THREE.Quaternion();
  const _ikHingeTmpWorld = new THREE.Quaternion();
  function _enforceHingeNoHyperextend(localDelta, parentPropQ, naturalSign) {
    _ikHingeTmpInv.copy(parentPropQ).invert();
    _ikHingeTmpWorld.copy(parentPropQ).multiply(localDelta).multiply(_ikHingeTmpInv);
    if (_ikHingeTmpWorld.x * naturalSign < 0) localDelta.set(0, 0, 0, 1);
  }

  function _propagateBoneTransform(bone) {
    const queue = [...bone.children];
    while (queue.length) {
      const c = queue.shift();
      const pWorldQ = c.parentBone.handle.quaternion;
      const pWorldPos = c.parentBone.handle.position;
      const lp = c.base.localPos.clone().applyQuaternion(pWorldQ);
      c.handle.position.copy(pWorldPos).add(lp);
      c.handle.quaternion.copy(pWorldQ).multiply(c.base.localQuat).multiply(c.localDelta);
      queue.push(...c.children);
    }
  }

  function _rebuildBoneLines() {
    if (!poseEdit.root) return;
    const linesGroup = poseEdit.root.getObjectByName("_bone_lines");
    if (!linesGroup) return;
    while (linesGroup.children.length) {
      const child = linesGroup.children[0];
      linesGroup.remove(child);
      child.geometry?.dispose?.();
    }
    for (const bone of poseEdit.bones) {
      if (!bone.parentBone) continue;
      const a = bone.parentBone.handle.position;
      const b = bone.handle.position;
      const g = new THREE.BufferGeometry().setFromPoints([a.clone(), b.clone()]);
      const line = new THREE.Line(g, _poseLineMat);
      line.renderOrder = 998;
      linesGroup.add(line);
    }
  }

  function _parentPropagatedWorldQuat(bone) {
    if (!bone.parentBone) return bone.base.worldQuat.clone();
    return bone.parentBone.handle.quaternion.clone().multiply(bone.base.localQuat);
  }

  function _isSpineSideBone(bone) {
    let cur = bone;
    while (cur) {
      if (cur.name === "Spine") return true;
      if (cur.name === "Hips")  return false;
      cur = cur.parentBone;
    }
    return false;
  }

  function _buildIkChain(endBone, maxLen) {
    const chain = [];
    let cur = endBone;
    const spineSide = _isSpineSideBone(endBone);
    while (cur && chain.length < maxLen) {
      chain.push(cur);
      if (chain.length > 1 && _IK_CEILING_BONES.has(cur.name)) break;
      if (spineSide && cur.parentBone && cur.parentBone.name === "Hips") break;
      cur = cur.parentBone;
    }
    return chain;
  }

  // CCD scratch values.
  const _ikTmpV1   = new THREE.Vector3();
  const _ikTmpV2   = new THREE.Vector3();
  const _ikTmpAxis = new THREE.Vector3();
  const _ikTmpDq   = new THREE.Quaternion();
  const _ikTmpNewQ = new THREE.Quaternion();

  function _runCcdIk(chain, targetAnchorLocal) {
    if (chain.length < 2) return;
    const endBone = chain[0];
    for (let iter = 0; iter < _IK_ITERATIONS; iter++) {
      let maxAngle = 0;
      for (let i = 1; i < chain.length; i++) {
        const joint = chain[i];
        _ikTmpV1.copy(endBone.handle.position).sub(joint.handle.position);
        _ikTmpV2.copy(targetAnchorLocal).sub(joint.handle.position);
        if (_ikTmpV1.lengthSq() < 1e-10 || _ikTmpV2.lengthSq() < 1e-10) continue;
        _ikTmpV1.normalize();
        _ikTmpV2.normalize();
        const dot = Math.max(-1, Math.min(1, _ikTmpV1.dot(_ikTmpV2)));
        let angle = Math.acos(dot);
        if (angle < _IK_ANGLE_TOLERANCE) continue;
        angle *= _IK_DAMPING;
        _ikTmpAxis.crossVectors(_ikTmpV1, _ikTmpV2);
        if (_ikTmpAxis.lengthSq() < 1e-12) continue;
        _ikTmpAxis.normalize();
        _ikTmpDq.setFromAxisAngle(_ikTmpAxis, angle);
        _ikTmpNewQ.copy(_ikTmpDq).multiply(joint.handle.quaternion);

        const parentPropQ = _parentPropagatedWorldQuat(joint);
        const parentPropQInv = parentPropQ.clone().invert();
        joint.localDelta.copy(parentPropQInv).multiply(_ikTmpNewQ);
        const hingeSign = _IK_HINGE_NATURAL_X[joint.name];
        if (hingeSign !== undefined) {
          _enforceHingeNoHyperextend(joint.localDelta, parentPropQ, hingeSign);
        }
        joint.handle.quaternion.copy(parentPropQ).multiply(joint.localDelta);
        _propagateBoneTransform(joint);
        if (angle > maxAngle) maxAngle = angle;
      }
      if (maxAngle < _IK_ANGLE_TOLERANCE) break;
    }
  }

  function _runSelfRotateIk(bone, targetAnchorLocal) {
    if (!bone.children || bone.children.length === 0) return;
    const tip = bone.children[0];
    for (let iter = 0; iter < _IK_ITERATIONS; iter++) {
      _ikTmpV1.copy(tip.handle.position).sub(bone.handle.position);
      _ikTmpV2.copy(targetAnchorLocal).sub(bone.handle.position);
      if (_ikTmpV1.lengthSq() < 1e-10 || _ikTmpV2.lengthSq() < 1e-10) break;
      _ikTmpV1.normalize();
      _ikTmpV2.normalize();
      const dot = Math.max(-1, Math.min(1, _ikTmpV1.dot(_ikTmpV2)));
      let angle = Math.acos(dot);
      if (angle < _IK_ANGLE_TOLERANCE) break;
      angle *= _IK_DAMPING;
      _ikTmpAxis.crossVectors(_ikTmpV1, _ikTmpV2);
      if (_ikTmpAxis.lengthSq() < 1e-12) break;
      _ikTmpAxis.normalize();
      _ikTmpDq.setFromAxisAngle(_ikTmpAxis, angle);
      _ikTmpNewQ.copy(_ikTmpDq).multiply(bone.handle.quaternion);
      const parentPropQ = _parentPropagatedWorldQuat(bone);
      const parentPropQInv = parentPropQ.clone().invert();
      bone.localDelta.copy(parentPropQInv).multiply(_ikTmpNewQ);
      bone.handle.quaternion.copy(parentPropQ).multiply(bone.localDelta);
      _propagateBoneTransform(bone);
    }
  }

  function _emitIkChainChange(chain) {
    if (!poseEdit.onIkChainChange) return;
    const eps = 1e-6;
    const changes = [];
    const startIdx = chain.length === 1 ? 0 : 1;
    for (let i = startIdx; i < chain.length; i++) {
      const joint = chain[i];
      const e = new THREE.Euler().setFromQuaternion(joint.localDelta, "XYZ");
      const isIdent = Math.abs(e.x) < eps && Math.abs(e.y) < eps && Math.abs(e.z) < eps;
      changes.push({ boneName: joint.name, eulerRad: isIdent ? null : [e.x, e.y, e.z] });
    }
    poseEdit.onIkChainChange(changes);
  }

  // Reusable temporaries for the TC-change handler.
  const _ikTargetLocal  = new THREE.Vector3();
  const _ikInvAnchorMat = new THREE.Matrix4();

  // Active person's anchor — the Object3D whose matrixWorld defines the
  // anchor-local frame our bones live in. Resolved per-call so it stays
  // valid across mesh re-renders (the anchor instance changes each time).
  function _activeAnchorObject() {
    if (!poseEdit.activePersonId) return null;
    const e = anchors.find((a) => a.id === poseEdit.activePersonId);
    return e?.anchor || null;
  }

  // Click-to-select handler — raycasts against bone handles, swaps the
  // selected material, attaches the gizmo to ``_ikTarget``.
  const _ikRay = new THREE.Raycaster();
  const _ikRayMouse = new THREE.Vector2();
  canvasEl.addEventListener("pointerdown", (ev) => {
    if (!poseEdit.active || ev.button !== 0) return;
    if (tcontrols.dragging) return;
    const r = canvasEl.getBoundingClientRect();
    _ikRayMouse.x = ((ev.clientX - r.left) / r.width)  *  2 - 1;
    _ikRayMouse.y = -((ev.clientY - r.top)  / r.height) * 2 + 1;
    _ikRay.setFromCamera(_ikRayMouse, camera);
    const handles = poseEdit.bones.map((b) => b.handle);
    const hits = _ikRay.intersectObjects(handles, false);
    if (hits.length === 0) return;
    const hit = hits[0].object;
    const bone = poseEdit.bones.find((b) => b.handle === hit);
    if (!bone) return;
    ev.preventDefault();
    ev.stopPropagation();
    selectPoseBone(bone.name);
    if (poseEdit.onBonePick) poseEdit.onBonePick(bone.name);
  });

  tcontrols.addEventListener("dragging-changed", (ev) => {
    controls.enabled = !ev.value;
    if (!poseEdit.active) return;
    if (ev.value) {
      if (poseEdit.selected) {
        if (poseEdit.selected.name === _HIPS_BONE_NAME) {
          _hipsDrag.active = true;
          _hipsDrag.lastTargetWorld.copy(_ikTarget.position);
          if (poseEdit.onDragStart) poseEdit.onDragStart();
        } else {
          _ikDrag.bone = poseEdit.selected;
          _ikDrag.chain = _buildIkChain(poseEdit.selected, _IK_MAX_CHAIN_LEN);
          const chainLen = _ikDrag.chain.length;
          const canSelfRotate = chainLen === 1
            && _ikDrag.chain[0].children
            && _ikDrag.chain[0].children.length > 0;
          _ikDrag.active = (chainLen >= 2) || canSelfRotate;
          if (_ikDrag.active && poseEdit.onDragStart) poseEdit.onDragStart();
        }
      }
    } else {
      // Drag end — snap target back to the bone's actual position so the
      // next drag starts from the bone, not the (possibly unreachable)
      // place the gizmo was left.
      if (poseEdit.selected) {
        poseEdit.selected.handle.getWorldPosition(_ikTarget.position);
      }
      const wasActive = _ikDrag.active || _hipsDrag.active;
      _ikDrag.active = false;
      _ikDrag.bone   = null;
      _ikDrag.chain  = [];
      _hipsDrag.active = false;
      if (wasActive && poseEdit.onDragEnd) poseEdit.onDragEnd();
    }
  });

  tcontrols.addEventListener("change", () => {
    if (!poseEdit.active) {
      // Outside pose-edit, the gizmo is bound to a body anchor (per-person
      // translate / rotate). The change listener for that path is a
      // separate ``onTransformChanged`` callback registered via
      // ``setOnTransformChanged``.
      _dirty = true;
      if (_syncRaf) return;
      _syncRaf = requestAnimationFrame(() => {
        _syncRaf = 0;
        _onTransformChanged?.();
      });
      return;
    }
    // Hips drag — translate the active person's anchor position.
    if (_hipsDrag.active) {
      const dx = _ikTarget.position.x - _hipsDrag.lastTargetWorld.x;
      const dy = _ikTarget.position.y - _hipsDrag.lastTargetWorld.y;
      const dz = _ikTarget.position.z - _hipsDrag.lastTargetWorld.z;
      const a = _activeAnchorObject();
      if (a) {
        a.position.x += dx;
        a.position.y += dy;
        a.position.z += dz;
      }
      if (poseEdit.onHipsTranslate) poseEdit.onHipsTranslate(dx, dy, dz);
      _hipsDrag.lastTargetWorld.copy(_ikTarget.position);
      _dirty = true;
      return;
    }
    if (!_ikDrag.active || !_ikDrag.bone) { _dirty = true; return; }

    // Convert world-space gizmo target into anchor-local for CCD math.
    const a = _activeAnchorObject();
    _ikTargetLocal.copy(_ikTarget.position);
    if (a) {
      a.updateMatrixWorld(true);
      _ikInvAnchorMat.copy(a.matrixWorld).invert();
      _ikTargetLocal.applyMatrix4(_ikInvAnchorMat);
    }
    if (_ikDrag.chain.length >= 2) {
      _runCcdIk(_ikDrag.chain, _ikTargetLocal);
    } else {
      _runSelfRotateIk(_ikDrag.chain[0], _ikTargetLocal);
    }
    // Cascade refresh from every root bone so descendants outside the
    // current chain don't drift out of sync with their stored localDelta.
    for (const rb of poseEdit.bones) {
      if (!rb.parentBone) _propagateBoneTransform(rb);
    }
    _rebuildBoneLines();
    _emitIkChainChange(_ikDrag.chain);
    _dirty = true;
  });

  // ---- Public pose-edit API ------------------------------------------------

  function enterPoseEdit(skeleton, storedOverrides, personId) {
    if (poseEdit.active) exitPoseEdit();
    if (!skeleton || !Array.isArray(skeleton.bones)) return;
    const anchorEntry = anchors.find((a) => a.id === personId);
    if (!anchorEntry) return;

    // Detach the gizmo from whatever it was previously bound to (likely
    // the active body anchor's translate/rotate gizmo). Without this the
    // hidden body-anchor gizmo can still capture the very first drag in
    // pose-edit mode and translate the whole body before the user has
    // selected a bone — exactly the "末端を動かすと体全体が移動" issue
    // that's not present in the single Pose Editor (which detaches via
    // its own setup flow).
    tcontrols.detach();

    poseEdit.activePersonId = personId;
    const lcx = anchorEntry.localCenter[0] || 0;
    const lcy = anchorEntry.localCenter[1] || 0;
    const lcz = anchorEntry.localCenter[2] || 0;

    // Handle radius in world units, scaled to anchor-local frame. Multi
    // anchors don't apply a uniform scale (they use rotation + translate),
    // so anchor-local == world-scale.
    const baseScale = 0.022;       // matches single editor's mid-range size
    poseEdit.baseHandleScale = baseScale;
    poseEdit.lastAnchorScale = 1.0;

    const root = new THREE.Group();
    root.name = "_pose_edit_root";
    root.renderOrder = 999;
    anchorEntry.anchor.add(root);
    const linesGroup = new THREE.Group();
    linesGroup.name = "_bone_lines";
    root.add(linesGroup);

    poseEdit.bones = [];
    poseEdit.byName = new Map();

    // First pass: create handles at anchor-local positions
    // (world_position - localCenter).
    for (const b of skeleton.bones) {
      const handle = new THREE.Mesh(_poseHandleGeom, _poseHandleMat);
      const perBoneScale = _isFingerBoneName(b.name)
        ? baseScale * _FINGER_HANDLE_SCALE_RATIO
        : baseScale;
      handle.scale.setScalar(perBoneScale);
      handle.renderOrder = 1000;
      handle.position.set(
        b.world_position[0] - lcx,
        b.world_position[1] - lcy,
        b.world_position[2] - lcz,
      );
      handle.quaternion.set(
        b.world_quaternion[0], b.world_quaternion[1],
        b.world_quaternion[2], b.world_quaternion[3],
      );
      handle.name = `_pose_bone_${b.name}`;
      root.add(handle);

      const bone = {
        name: b.name,
        jointId: b.joint_id,
        parentName: b.parent_name,
        handle,
        base: {
          worldPos:  handle.position.clone(),
          worldQuat: handle.quaternion.clone(),
          localPos:  new THREE.Vector3(),
          localQuat: new THREE.Quaternion(),
        },
        localDelta: new THREE.Quaternion(),
        parentBone: null,
        children: [],
      };
      poseEdit.bones.push(bone);
      poseEdit.byName.set(b.name, bone);
    }

    // Second pass: parents + base local transforms.
    for (const bone of poseEdit.bones) {
      if (bone.parentName && poseEdit.byName.has(bone.parentName)) {
        const parent = poseEdit.byName.get(bone.parentName);
        bone.parentBone = parent;
        parent.children.push(bone);
        const pQ = parent.base.worldQuat.clone().invert();
        const dp = bone.base.worldPos.clone().sub(parent.base.worldPos);
        bone.base.localPos.copy(dp).applyQuaternion(pQ);
        bone.base.localQuat.copy(pQ).multiply(bone.base.worldQuat);
      } else {
        bone.base.localPos.copy(bone.base.worldPos);
        bone.base.localQuat.copy(bone.base.worldQuat);
      }
    }

    // Restore stored overrides — topological order (server emits parents first).
    if (storedOverrides) {
      for (const bone of poseEdit.bones) {
        const ov = storedOverrides[String(bone.jointId)];
        if (!ov || ov.length !== 3) continue;
        const e = new THREE.Euler(ov[0], ov[1], ov[2], "XYZ");
        bone.localDelta.setFromEuler(e);
        const parentPropQ = _parentPropagatedWorldQuat(bone);
        bone.handle.quaternion.copy(parentPropQ).multiply(bone.localDelta);
        _propagateBoneTransform(bone);
      }
    }

    poseEdit.root = root;
    _rebuildBoneLines();
    tcontrols.enabled = true;
    tcHelper.visible = false;   // shown on selection
    poseEdit.active = true;
    _dirty = true;
  }

  function exitPoseEdit() {
    if (!poseEdit.active) return;
    tcontrols.detach();
    tcontrols.enabled = false;
    tcHelper.visible = false;
    if (poseEdit.root) {
      if (poseEdit.root.parent) poseEdit.root.parent.remove(poseEdit.root);
      poseEdit.root.traverse?.((o) => {
        if (o.isMesh || o.isLine) o.geometry?.dispose?.();
      });
      poseEdit.root = null;
    }
    poseEdit.bones = [];
    poseEdit.byName = new Map();
    poseEdit.selected = null;
    poseEdit.activePersonId = null;
    poseEdit.active = false;
    _dirty = true;
  }

  function selectPoseBone(name) {
    if (!poseEdit.active) return;
    for (const b of poseEdit.bones) b.handle.material = _poseHandleMat;
    const bone = name ? poseEdit.byName.get(name) : null;
    if (!bone) {
      poseEdit.selected = null;
      tcontrols.detach();
      tcHelper.visible = false;
      _dirty = true;
      return;
    }
    bone.handle.material = _poseHandleSelectedMat;
    poseEdit.selected = bone;
    bone.handle.getWorldPosition(_ikTarget.position);
    tcontrols.attach(_ikTarget);
    tcontrols.size = _isFingerBoneName(bone.name) ? 0.35 : 1.0;
    tcHelper.visible = true;
    _dirty = true;
  }

  function resetPoseBone(name) {
    const bone = poseEdit.byName.get(name);
    if (!bone) return;
    bone.localDelta.identity();
    if (bone.parentBone) {
      const pQ = bone.parentBone.handle.quaternion;
      const pP = bone.parentBone.handle.position;
      const lp = bone.base.localPos.clone().applyQuaternion(pQ);
      bone.handle.position.copy(pP).add(lp);
      bone.handle.quaternion.copy(pQ).multiply(bone.base.localQuat);
    } else {
      bone.handle.position.copy(bone.base.worldPos);
      bone.handle.quaternion.copy(bone.base.worldQuat);
    }
    _propagateBoneTransform(bone);
    _rebuildBoneLines();
    if (poseEdit.onBoneChange) poseEdit.onBoneChange(bone.name, null);
    _dirty = true;
  }

  function resetAllPoseBones() {
    for (const bone of poseEdit.bones) {
      bone.localDelta.identity();
      bone.handle.position.copy(bone.base.worldPos);
      bone.handle.quaternion.copy(bone.base.worldQuat);
      if (poseEdit.onBoneChange) poseEdit.onBoneChange(bone.name, null);
    }
    if (poseEdit.selected) {
      poseEdit.selected.handle.getWorldPosition(_ikTarget.position);
    }
    _rebuildBoneLines();
    _dirty = true;
  }

  function setPoseBoneLocalEuler(name, rx, ry, rz) {
    const bone = poseEdit.byName.get(name);
    if (!bone) return;
    const e = new THREE.Euler(rx, ry, rz, "XYZ");
    bone.localDelta.setFromEuler(e);
    const parentPropQ = _parentPropagatedWorldQuat(bone);
    bone.handle.quaternion.copy(parentPropQ).multiply(bone.localDelta);
    _propagateBoneTransform(bone);
    _rebuildBoneLines();
    if (poseEdit.selected === bone) {
      bone.handle.getWorldPosition(_ikTarget.position);
    }
    _dirty = true;
  }

  function getPoseBoneLocalEuler(name) {
    const bone = poseEdit.byName.get(name);
    if (!bone) return null;
    const e = new THREE.Euler().setFromQuaternion(bone.localDelta, "XYZ");
    return [e.x, e.y, e.z];
  }

  function setPoseEditCallbacks(cb) {
    cb = cb || {};
    poseEdit.onBoneChange    = cb.onBoneChange    || null;
    poseEdit.onBonePick      = cb.onBonePick      || null;
    poseEdit.onDragStart     = cb.onDragStart     || null;
    poseEdit.onDragEnd       = cb.onDragEnd       || null;
    poseEdit.onIkChainChange = cb.onIkChainChange || null;
    poseEdit.onHipsTranslate = cb.onHipsTranslate || null;
  }

  function isPoseEditActive() { return poseEdit.active; }

  function getPoseEditActivePersonId() { return poseEdit.activePersonId; }

  // Re-sync the gizmo target to the selected bone — used after a render
  // settles so the gizmo doesn't hang in stale space when the user wasn't
  // mid-drag.
  function syncIkTargetToSelected() {
    if (!poseEdit.active || !poseEdit.selected) return;
    if (_ikDrag.active || _hipsDrag.active) return;
    poseEdit.selected.handle.getWorldPosition(_ikTarget.position);
    _dirty = true;
  }

  return {
    resize,
    clearAnchors,
    loadObjPerPerson,
    resetFramedFingerprint,
    resetCohortLayoutCache,
    getAnchorEntry,
    setActiveAnchor,
    setGizmoMode,
    setSoloVisibility,
    enterPoseEdit,
    exitPoseEdit,
    selectPoseBone,
    resetPoseBone,
    resetAllPoseBones,
    setPoseBoneLocalEuler,
    getPoseBoneLocalEuler,
    setPoseEditCallbacks,
    isPoseEditActive,
    getPoseEditActivePersonId,
    syncIkTargetToSelected,
    captureComposite,
    readActiveTransform,
    writeAnchorTransform,
    setOnTransformChanged,
    getAnchorCount,
    showCanvas: (yes) => { canvasEl.style.display = yes ? "block" : "none"; },
  };
})();

// ---------------------------------------------------------------------------
// View switching (3D / Image)
// ---------------------------------------------------------------------------

function setView(name) {
  state.view = name;
  tab3dBtn.classList.toggle("active", name === "3d");
  tabImageBtn.classList.toggle("active", name === "image");
  viewer.showCanvas(name === "3d");
  // Range / bg-color buttons act on the 3D capture only — hide them
  // entirely when the user's looking at the input image.
  const leftTb = document.querySelector(".viewport-toolbar-left");
  if (leftTb) leftTb.style.display = name === "3d" ? "flex" : "none";
  _refreshViewportToolbar();
  // Image display rules:
  //   - "image" tab → big centred image, drawing overlay enabled.
  //   - "3d" tab    → small thumbnail in the corner, no drawing.
  if (name === "image") {
    if (state.inputImage) {
      imageDisplay.classList.add("visible");
      imageDisplay.style.cssText = `
        position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        z-index: 2; padding: 0; border: none; background: #1a1a1a;
        display: flex; align-items: center; justify-content: center;
      `;
      imageDisplayImg.style.cssText =
        "max-width: 100%; max-height: 100%; object-fit: contain;";
      imageDisplayImg.src = state.inputImage.dataUrl;
      $("image-display-caption").style.display = "none";
    }
  } else {
    if (state.inputImage) {
      imageDisplay.classList.toggle("visible", _imgPreviewOn);
      imageDisplay.style.cssText = "";
      imageDisplayImg.style.cssText = "";
      imageDisplayImg.src = state.inputImage.dataUrl;
      $("image-display-caption").style.display = "";
    }
  }
  rebuildBboxOverlay();
  refreshFloatingAddBtn();
  // Toggle bbox-overlay pointer-events so 3D-view drags reach the canvas
  // (TransformControls gizmo); image-view drags reach the overlay.
  if (typeof _refreshBboxOverlayPointerEvents === "function") {
    _refreshBboxOverlayPointerEvents();
  }
}

let _imgPreviewOn = false;
toggleImgBtn.addEventListener("click", () => {
  _imgPreviewOn = !_imgPreviewOn;
  toggleImgBtn.classList.toggle("active", _imgPreviewOn);
  if (state.view === "3d") {
    imageDisplay.classList.toggle("visible", _imgPreviewOn);
  }
});

tab3dBtn.addEventListener("click", () => setView("3d"));
tabImageBtn.addEventListener("click", () => setView("image"));

// ---------------------------------------------------------------------------
// Gizmo mode toggle (translate / rotate)
// ---------------------------------------------------------------------------

let _gizmoMode = "translate";
function setGizmoMode(mode) {
  _gizmoMode = mode === "rotate" ? "rotate" : "translate";
  viewer.setGizmoMode(_gizmoMode);
  gizmoTranslateBtn.classList.toggle("active", _gizmoMode === "translate");
  gizmoRotateBtn.classList.toggle("active", _gizmoMode === "rotate");
}
gizmoTranslateBtn.addEventListener("click", () => {
  if (_ikMode) setIkMode(false);
  setGizmoMode("translate");
});
gizmoRotateBtn.addEventListener("click", () => {
  if (_ikMode) setIkMode(false);
  setGizmoMode("rotate");
});
document.addEventListener("keydown", (ev) => {
  // Match Blender / Three.js convention: W = translate, E = rotate.
  if (ev.target?.tagName === "INPUT" || ev.target?.tagName === "TEXTAREA") return;
  if (ev.key === "w" || ev.key === "W") setGizmoMode("translate");
  if (ev.key === "e" || ev.key === "E") setGizmoMode("rotate");
});

// Reset translate / rotate for the active person. Touches the anchor
// directly + zeroes the relevant axis on state.persons[active].transform,
// then rebuilds the person card so the slider values mirror the change.
function _resetActiveTransform(component /* "translate" | "rotate" */) {
  if (!state.activeId) return;
  const p = state.persons.find((q) => q.id === state.activeId);
  if (!p?.transform) return;
  if (component === "translate") {
    p.transform.translate = [0, 0, 0];
  } else {
    p.transform.rotate_deg = [0, 0, 0];
  }
  if (viewer.getAnchorCount() > 0) {
    viewer.writeAnchorTransform(p.id, p.transform.translate, p.transform.rotate_deg);
  }
  rebuildPersonList();
}
resetTranslateBtn.addEventListener("click", () => _resetActiveTransform("translate"));
resetRotateBtn.addEventListener("click",    () => _resetActiveTransform("rotate"));

// ---------------------------------------------------------------------------
// Pose-edit / IK mode — wires the viewer's ported ``poseEdit`` system
// (CCD multi-bone IK, hinge limits, Hips translate, full bone selection)
// to the per-person ``settings.pose_adjust.rotation_overrides`` blob the
// backend renderer consumes. This is a 1:1 port of the single Pose Editor's
// ``initPoseEditUi`` callback wiring (``editor_core.js``) — algorithmic
// behaviour matches the single editor exactly.
// ---------------------------------------------------------------------------

let _ikMode = false;

// Set on drag end → consumed in ``reRender``'s finally block once the
// final render settles. Triggers a full pose-edit overlay rebuild against
// the freshest skeleton so the IK bones snap exactly onto the rendered
// mesh's joint positions (eliminates any cascade drift the CCD picked
// up during a long drag).
let _poseRebuildAfterRender = false;

function _rebuildPoseOverlayFromCurrent() {
  if (!_ikMode || !viewer.isPoseEditActive()) return;
  const personId = viewer.getPoseEditActivePersonId();
  const p = state.persons.find((q) => q.id === personId);
  if (!p?.skeleton?.bones) return;
  const stored = p.settings?.pose_adjust?.rotation_overrides || {};
  const sel = p.selectedBoneName;
  // exitPoseEdit + enterPoseEdit reseats every bone's ``base`` against
  // the just-arrived skeleton, then re-applies the stored overrides on
  // top — bones end up at the same world position the renderer's mesh
  // is showing, with no mathematical drift.
  viewer.exitPoseEdit();
  viewer.enterPoseEdit(p.skeleton, stored, p.id);
  if (sel) viewer.selectPoseBone(sel);
}

// Pull the joint id for a bone name out of the active person's skeleton
// (the renderer emits joint_id alongside name; we use it as the override
// dict's key so backend lookup is O(1)).
function _jointIdOf(boneName) {
  const p = state.persons.find((q) => q.id === state.activeId);
  if (!p?.skeleton?.bones) return null;
  const b = p.skeleton.bones.find((x) => x.name === boneName);
  return b ? String(b.joint_id) : null;
}

function _writeOverride(boneName, eulerRad) {
  const p = state.persons.find((q) => q.id === state.activeId);
  if (!p) return;
  if (!p.settings) p.settings = _emptySettings();
  if (!p.settings.pose_adjust) p.settings.pose_adjust = { lean_correction: 0, rotation_overrides: {} };
  if (!p.settings.pose_adjust.rotation_overrides) p.settings.pose_adjust.rotation_overrides = {};
  const key = _jointIdOf(boneName);
  if (!key) return;
  if (!eulerRad) {
    delete p.settings.pose_adjust.rotation_overrides[key];
  } else {
    p.settings.pose_adjust.rotation_overrides[key] = [
      Number(eulerRad[0]), Number(eulerRad[1]), Number(eulerRad[2]),
    ];
  }
}

// Wire pose-edit callbacks once. Each callback writes the user's edits
// into ``state.persons[active].settings.pose_adjust.rotation_overrides``
// and triggers a real-time render.
viewer.setPoseEditCallbacks({
  onBoneChange: (boneName, eulerRad) => {
    _writeOverride(boneName, eulerRad);
    triggerReRender();
  },
  onBonePick: (boneName) => {
    const p = state.persons.find((q) => q.id === state.activeId);
    if (!p) return;
    p.selectedBoneName = boneName;
    rebuildPersonList();
  },
  onIkChainChange: (changes) => {
    for (const { boneName, eulerRad } of changes) {
      _writeOverride(boneName, eulerRad);
    }
    triggerReRender();
  },
  onDragStart: () => { /* hook for undo/redo if added later */ },
  onDragEnd:   () => {
    // One final render after release in case the last drag event was
    // coalesced into an in-flight render and its follow-up was skipped.
    // After that render settles, ``reRender`` will rebuild the pose-edit
    // overlay against the freshest skeleton (see ``_poseRebuildAfterRender``)
    // so any cascade drift accumulated during the drag is wiped — IK
    // bones land exactly on the mesh's posed joint positions.
    _poseRebuildAfterRender = true;
    triggerReRender();
  },
  // Hips translate — the gizmo delta has already been applied to the
  // active person's anchor by the viewer; we just need to mirror it
  // into ``state.persons[active].transform.translate`` so subsequent
  // renders preserve the offset.
  onHipsTranslate: (dx, dy, dz) => {
    const p = state.persons.find((q) => q.id === state.activeId);
    if (!p) return;
    if (!p.transform) p.transform = { translate: [0, 0, 0], rotate_deg: [0, 0, 0] };
    p.transform.translate = [
      (p.transform.translate[0] || 0) + dx,
      (p.transform.translate[1] || 0) + dy,
      (p.transform.translate[2] || 0) + dz,
    ];
  },
});

function setIkMode(on) {
  _ikMode = !!on;
  if (_ikMode) {
    const p = state.persons.find((q) => q.id === state.activeId);
    if (!p?.skeleton?.bones) {
      _ikMode = false;
      rebuildPersonList();
      _refreshViewportToolbar();
      return;
    }
    const stored = p.settings?.pose_adjust?.rotation_overrides || {};
    viewer.enterPoseEdit(p.skeleton, stored, p.id);
    if (!viewer.isPoseEditActive()) {
      _ikMode = false;
      rebuildPersonList();
      _refreshViewportToolbar();
      return;
    }
    if (p.selectedBoneName) {
      viewer.selectPoseBone(p.selectedBoneName);
    }
    gizmoTranslateBtn.classList.remove("active");
    gizmoRotateBtn.classList.remove("active");
  } else {
    viewer.exitPoseEdit();
    gizmoTranslateBtn.classList.toggle("active", _gizmoMode === "translate");
    gizmoRotateBtn.classList.toggle("active",    _gizmoMode === "rotate");
    if (state.activeId) viewer.setActiveAnchor(state.activeId);
    viewer.setGizmoMode(_gizmoMode);
  }
  rebuildPersonList();
  _refreshViewportToolbar();
}

// Rebuild the pose-edit overlay against a different person — used when
// the active person changes while IK mode is on.
function refreshPoseEditForActive() {
  if (!_ikMode) return;
  const p = state.persons.find((q) => q.id === state.activeId);
  if (!p?.skeleton?.bones) {
    viewer.exitPoseEdit();
    return;
  }
  if (viewer.getPoseEditActivePersonId() === p.id) return;
  const stored = p.settings?.pose_adjust?.rotation_overrides || {};
  viewer.enterPoseEdit(p.skeleton, stored, p.id);
  if (p.selectedBoneName) viewer.selectPoseBone(p.selectedBoneName);
}

// Live sync of gizmo edits back to per-person state + sidebar sliders.
// Throttled by the viewer so a fast drag doesn't melt the DOM. The slider
// inputs are tracked in ``_transformSliderRefs`` (populated by
// ``_buildTransformPanel``); we update their ``.value`` directly to avoid
// rebuilding the entire person list each frame.
const _transformSliderRefs = new Map();   // personId → {translate:[in*3], rotate:[in*3]}

viewer.setOnTransformChanged(() => {
  const xform = viewer.readActiveTransform();
  if (!xform) return;
  const p = state.persons.find((q) => q.id === xform.id);
  if (!p) return;
  p.transform.translate = xform.translate.map((v) => Math.round(v * 1000) / 1000);
  p.transform.rotate_deg = xform.rotate_deg.map((v) => Math.round(v * 10) / 10);
  // Push values into the visible slider inputs (if the panel is open).
  const refs = _transformSliderRefs.get(xform.id);
  if (refs) {
    for (let i = 0; i < 3; i++) {
      if (refs.translate?.[i]) {
        refs.translate[i].range.value = String(p.transform.translate[i].toFixed(2));
        refs.translate[i].num.value   = String(p.transform.translate[i].toFixed(2));
      }
      if (refs.rotate?.[i]) {
        refs.rotate[i].range.value = String(p.transform.rotate_deg[i].toFixed(1));
        refs.rotate[i].num.value   = String(p.transform.rotate_deg[i].toFixed(1));
      }
    }
  }
});

// ---------------------------------------------------------------------------
// Bbox overlay — image-space ↔ overlay-pixel mapping
// ---------------------------------------------------------------------------

function _imageDisplayBounds() {
  // When in "image" view, the displayed image is letterboxed inside the
  // viewport. We need the on-screen position/size of the image bitmap so
  // we can map pointer coords to image-pixel coords. In "3d" view we map
  // bboxes onto the small corner thumbnail (keeps them roughly visible).
  if (!state.inputImage) return null;
  const img = imageDisplayImg;
  if (!img || !img.naturalWidth) return null;
  const r = img.getBoundingClientRect();
  const vp = viewportEl.getBoundingClientRect();
  return {
    x: r.left - vp.left, y: r.top - vp.top,
    w: r.width, h: r.height,
    natW: state.inputImage.width, natH: state.inputImage.height,
  };
}

function _overlayPxToImagePx(px, py) {
  const b = _imageDisplayBounds();
  if (!b) return null;
  const tx = (px - b.x) / b.w;
  const ty = (py - b.y) / b.h;
  return [tx * b.natW, ty * b.natH];
}

function _imagePxToOverlayPx(ix, iy) {
  const b = _imageDisplayBounds();
  if (!b) return null;
  return [b.x + (ix / b.natW) * b.w, b.y + (iy / b.natH) * b.h];
}

function _bboxImageToOverlay(bbox) {
  const a = _imagePxToOverlayPx(bbox[0], bbox[1]);
  const c = _imagePxToOverlayPx(bbox[2], bbox[3]);
  if (!a || !c) return null;
  return { left: a[0], top: a[1], width: c[0] - a[0], height: c[1] - a[1] };
}

function rebuildBboxOverlay() {
  // Persist whenever the bbox layer is rebuilt — typically follows a
  // bbox add / delete or active-person switch.
  // Wipe all old rects (keep only the active-draw skeleton).
  for (const el of [...bboxOverlay.querySelectorAll(".bbox-rect, .bbox-pos-add, .bbox-neg, .bbox-obj, .bbox-obj-pos-add")]) el.remove();
  _refreshNegHint();
  if (!state.inputImage) return;
  // Bboxes are anchored to the displayed image — only render them while the
  // image is on screen.
  if (state.view !== "image") return;

  // Display policy:
  //   - Inactive persons: only their primary bbox, drawn faded (a
  //     clickable target so the user can switch active without losing
  //     their visual frame of reference). Additional / negative bboxes
  //     are hidden for non-active persons.
  //   - Active person: full set — primary, additional positives,
  //     negatives — at full opacity, so the user can edit / delete them.
  for (const p of state.persons) {
    if (!p.bbox) continue;
    const r = _bboxImageToOverlay(p.bbox);
    if (!r) continue;
    const isActive = p.id === state.activeId;

    const div = document.createElement("div");
    div.className = "bbox-rect" + (isActive ? " active" : "");
    if (!isActive) div.style.opacity = "0.35";
    Object.assign(div.style, {
      left: `${r.left}px`, top: `${r.top}px`,
      width: `${r.width}px`, height: `${r.height}px`,
    });
    const label = document.createElement("div");
    label.className = "pid-label";
    label.textContent = p.id;
    div.appendChild(label);

    // Delete + draw buttons only on the ACTIVE person's primary bbox —
    // otherwise the overlay gets visually noisy.
    if (isActive) {
      const del = document.createElement("button");
      del.className = "delete-btn";
      del.textContent = "×";
      del.style.pointerEvents = "auto";
      del.addEventListener("click", (ev) => {
        ev.stopPropagation();
        ev.preventDefault();
        removePerson(p.id);
      });
      div.appendChild(del);
    }

    div.addEventListener("click", () => setActive(p.id));
    bboxOverlay.appendChild(div);

    if (!isActive) continue;   // skip extras for non-active persons.

    // Additional positive bboxes — dashed blue outline.
    (p.additionalBboxes || []).forEach((ab, j) => {
      const ar = _bboxImageToOverlay(ab);
      if (!ar) return;
      const ad = document.createElement("div");
      ad.className = "bbox-pos-add active";
      Object.assign(ad.style, {
        left: `${ar.left}px`, top: `${ar.top}px`,
        width: `${ar.width}px`, height: `${ar.height}px`,
      });
      const albl = document.createElement("div");
      albl.className = "pid-label";
      albl.textContent = `${p.id} +${j + 1}`;
      const adel = document.createElement("button");
      adel.className = "delete-btn";
      adel.textContent = "×";
      adel.style.pointerEvents = "auto";
      adel.addEventListener("click", (ev) => {
        ev.stopPropagation();
        ev.preventDefault();
        removeAdditionalBbox(p.id, j);
      });
      ad.appendChild(albl);
      ad.appendChild(adel);
      ad.addEventListener("contextmenu", (ev) => ev.preventDefault());
      bboxOverlay.appendChild(ad);
    });

    // Negative bboxes — red dashed outline.
    (p.negativeBboxes || []).forEach((nb, j) => {
      const nr = _bboxImageToOverlay(nb);
      if (!nr) return;
      const nd = document.createElement("div");
      nd.className = "bbox-neg active";
      Object.assign(nd.style, {
        left: `${nr.left}px`, top: `${nr.top}px`,
        width: `${nr.width}px`, height: `${nr.height}px`,
      });
      const nlbl = document.createElement("div");
      nlbl.className = "neg-label";
      nlbl.textContent = `${p.id} − ${j + 1}`;
      const ndel = document.createElement("button");
      ndel.className = "delete-btn";
      ndel.textContent = "×";
      ndel.style.pointerEvents = "auto";
      ndel.addEventListener("click", (ev) => {
        ev.stopPropagation();
        ev.preventDefault();
        removeNegBbox(p.id, j);
      });
      nd.appendChild(nlbl);
      nd.appendChild(ndel);
      nd.addEventListener("contextmenu", (ev) => ev.preventDefault());
      bboxOverlay.appendChild(nd);
    });
  }

  // Object bboxes — rendered with the same primary / additional /
  // negative pattern as persons so the editing experience is uniform.
  // Yellow / amber for object positives, shared red for negatives.
  for (const o of state.objects) {
    if (!o.bbox) continue;
    const r = _bboxImageToOverlay(o.bbox);
    if (!r) continue;
    const isActive = o.id === state.activeObjectId;

    const div = document.createElement("div");
    div.className = "bbox-obj" + (isActive ? " active" : "");
    if (!isActive) div.style.opacity = "0.4";
    Object.assign(div.style, {
      left: `${r.left}px`, top: `${r.top}px`,
      width: `${r.width}px`, height: `${r.height}px`,
    });
    const label = document.createElement("div");
    label.className = "pid-label";
    label.textContent = o.id;
    div.appendChild(label);

    if (isActive) {
      const del = document.createElement("button");
      del.className = "delete-btn";
      del.textContent = "×";
      del.style.pointerEvents = "auto";
      del.addEventListener("click", (ev) => {
        ev.stopPropagation();
        ev.preventDefault();
        removeObject(o.id);
      });
      div.appendChild(del);
    }

    div.addEventListener("click", () => setActiveObject(o.id));
    bboxOverlay.appendChild(div);

    if (!isActive) continue;

    (o.additionalBboxes || []).forEach((ab, j) => {
      const ar = _bboxImageToOverlay(ab);
      if (!ar) return;
      const ad = document.createElement("div");
      ad.className = "bbox-obj-pos-add active";
      Object.assign(ad.style, {
        left: `${ar.left}px`, top: `${ar.top}px`,
        width: `${ar.width}px`, height: `${ar.height}px`,
      });
      const albl = document.createElement("div");
      albl.className = "pid-label";
      albl.textContent = `${o.id} +${j + 1}`;
      const adel = document.createElement("button");
      adel.className = "delete-btn";
      adel.textContent = "×";
      adel.style.pointerEvents = "auto";
      adel.addEventListener("click", (ev) => {
        ev.stopPropagation();
        ev.preventDefault();
        removeObjectAdditionalBbox(o.id, j);
      });
      ad.appendChild(albl);
      ad.appendChild(adel);
      ad.addEventListener("contextmenu", (ev) => ev.preventDefault());
      bboxOverlay.appendChild(ad);
    });

    (o.negativeBboxes || []).forEach((nb, j) => {
      const nr = _bboxImageToOverlay(nb);
      if (!nr) return;
      const nd = document.createElement("div");
      nd.className = "bbox-neg active";
      Object.assign(nd.style, {
        left: `${nr.left}px`, top: `${nr.top}px`,
        width: `${nr.width}px`, height: `${nr.height}px`,
      });
      const nlbl = document.createElement("div");
      nlbl.className = "neg-label";
      nlbl.textContent = `${o.id} − ${j + 1}`;
      const ndel = document.createElement("button");
      ndel.className = "delete-btn";
      ndel.textContent = "×";
      ndel.style.pointerEvents = "auto";
      ndel.addEventListener("click", (ev) => {
        ev.stopPropagation();
        ev.preventDefault();
        removeObjectNegBbox(o.id, j);
      });
      nd.appendChild(nlbl);
      nd.appendChild(ndel);
      nd.addEventListener("contextmenu", (ev) => ev.preventDefault());
      bboxOverlay.appendChild(nd);
    });
  }
}

function removeObjectAdditionalBbox(objectId, index) {
  const o = state.objects.find((q) => q.id === objectId);
  if (!o?.additionalBboxes) return;
  o.additionalBboxes.splice(index, 1);
  rebuildBboxOverlay();
  scheduleObjectRecrop(o.id);
}

function removeObjectNegBbox(objectId, index) {
  const o = state.objects.find((q) => q.id === objectId);
  if (!o?.negativeBboxes) return;
  o.negativeBboxes.splice(index, 1);
  rebuildBboxOverlay();
  scheduleObjectRecrop(o.id);
}

function _refreshNegHint() {
  if (!negBboxHint) return;
  const showHint = state.view === "image" && !state.drawMode &&
    (!!state.activeId || !!state.activeObjectId);
  negBboxHint.hidden = !showHint;
}

function removeNegBbox(personId, index) {
  const p = state.persons.find((q) => q.id === personId);
  if (!p?.negativeBboxes) return;
  p.negativeBboxes.splice(index, 1);
  rebuildBboxOverlay();
  scheduleSam2Preview();
}

function removeAdditionalBbox(personId, index) {
  const p = state.persons.find((q) => q.id === personId);
  if (!p?.additionalBboxes) return;
  p.additionalBboxes.splice(index, 1);
  rebuildBboxOverlay();
  scheduleSam2Preview();
}

// Drag drawing state. We split into a ``_pendingDrag`` (recorded at
// pointerdown but not yet committed) and a real ``_drag`` (only set
// once the pointer has moved past a small threshold). Splitting this
// way lets a click anywhere in the overlay — including ON existing
// bbox rectangles — keep working: a quick click bubbles to the
// underlying child's handler (× delete, switch active), while a real
// drag overrides and starts a new bbox.
let _drag = null;
let _pendingDrag = null;
const _DRAG_THRESHOLD_PX = 4;

// Right-click anywhere over the bbox overlay should never pop up the
// browser context menu — the right-click is reserved for negative-bbox
// drawing and we don't want a spurious menu interrupting it.
bboxOverlay.addEventListener("contextmenu", (ev) => ev.preventDefault());

// Pointer-events on the overlay are toggled by view: capture clicks for
// bbox drawing in image view, pass through to the canvas in 3D view so
// the TransformControls gizmo can receive its drag events. ``setView``
// updates this whenever the user switches tabs.
function _refreshBboxOverlayPointerEvents() {
  bboxOverlay.style.pointerEvents = (state.view === "image") ? "auto" : "none";
}
_refreshBboxOverlayPointerEvents();

bboxOverlay.addEventListener("pointerdown", (ev) => {
  if (!state.inputImage || state.view !== "image") return;

  // Buttons inside the overlay (× delete) get a free pass — never start
  // a drag from them so a single click reliably triggers the delete.
  const tag = ev.target?.tagName;
  if (tag === "BUTTON") return;

  // Decide what *kind* of bbox this drag would commit, based on
  // button + state. We DON'T commit yet; pointerdown only records
  // the start position and the intended mode in ``_pendingDrag``. The
  // actual drag is promoted in pointermove only after the pointer has
  // travelled past ``_DRAG_THRESHOLD_PX``, so a click without movement
  // bubbles through to the child's own click handler (setActive on
  // bbox body, etc.).
  let mode = null;
  if (ev.button === 2) {
    // Right click → negative bbox for whichever entity is active.
    // Mutually-exclusive: activeObject takes priority over activeId
    // (only one is ever set at a time, see setActive / setActiveObject).
    if (state.activeObjectId) mode = "object-negative";
    else if (state.activeId) mode = "negative";
    else return;
  } else if (ev.button === 0) {
    if (state.drawMode && state.drawType === "object") mode = "object";
    else if (state.drawMode) mode = "positive";
    else if (state.activeObjectId) mode = "object-positive-additional";
    else if (state.activeId) mode = "positive-additional";
    else return;
  } else {
    return;
  }

  const r = bboxOverlay.getBoundingClientRect();
  _pendingDrag = {
    mode,
    startX: ev.clientX - r.left,
    startY: ev.clientY - r.top,
    pointerId: ev.pointerId,
  };
  if (mode === "negative") {
    // Right-button needs preventDefault to avoid the browser context
    // menu before pointermove ever runs.
    ev.preventDefault();
  }
});

bboxOverlay.addEventListener("pointermove", (ev) => {
  // Promote a pending press to a real drag once the pointer moves past
  // the threshold. Until then, _drag stays null so a small twitch /
  // pure click still bubbles to child handlers.
  if (_pendingDrag && !_drag) {
    const r = bboxOverlay.getBoundingClientRect();
    const dx = (ev.clientX - r.left) - _pendingDrag.startX;
    const dy = (ev.clientY - r.top)  - _pendingDrag.startY;
    if (dx * dx + dy * dy < _DRAG_THRESHOLD_PX * _DRAG_THRESHOLD_PX) return;
    _drag = {
      mode: _pendingDrag.mode,
      startX: _pendingDrag.startX,
      startY: _pendingDrag.startY,
    };
    bboxRectActive.hidden = false;
    bboxRectActive.classList.toggle("neg", _drag.mode === "negative");
    bboxRectActive.style.left = `${_drag.startX}px`;
    bboxRectActive.style.top = `${_drag.startY}px`;
    bboxRectActive.style.width = `0px`;
    bboxRectActive.style.height = `0px`;
    try { bboxOverlay.setPointerCapture(_pendingDrag.pointerId); } catch (_e) {}
    _pendingDrag = null;
  }
  if (!_drag) return;
  const r = bboxOverlay.getBoundingClientRect();
  const x = ev.clientX - r.left;
  const y = ev.clientY - r.top;
  const left = Math.min(_drag.startX, x);
  const top  = Math.min(_drag.startY, y);
  const w    = Math.abs(x - _drag.startX);
  const h    = Math.abs(y - _drag.startY);
  bboxRectActive.style.left   = `${left}px`;
  bboxRectActive.style.top    = `${top}px`;
  bboxRectActive.style.width  = `${w}px`;
  bboxRectActive.style.height = `${h}px`;
});

function _endDrag(ev) {
  // Pointerup without crossing the drag threshold — discard the pending
  // intent so the child element's natural click handler can fire (×
  // delete button, click-to-activate on a faded bbox).
  if (_pendingDrag && !_drag) {
    _pendingDrag = null;
    return;
  }
  if (!_drag) return;
  try { bboxOverlay.releasePointerCapture(ev.pointerId); } catch (_e) {}
  const mode = _drag.mode;
  const r = bboxOverlay.getBoundingClientRect();
  const x = ev.clientX - r.left;
  const y = ev.clientY - r.top;
  const left = Math.min(_drag.startX, x);
  const top  = Math.min(_drag.startY, y);
  const right  = Math.max(_drag.startX, x);
  const bottom = Math.max(_drag.startY, y);
  _drag = null;
  bboxRectActive.hidden = true;
  bboxRectActive.classList.remove("neg");
  if ((right - left) < 6 || (bottom - top) < 6) {
    if (mode === "positive") {
      addHintEl.textContent = "矩形が小さすぎます — もう一度ドラッグしてください";
    }
    return;
  }
  const a = _overlayPxToImagePx(left, top);
  const b = _overlayPxToImagePx(right, bottom);
  if (!a || !b) {
    if (mode === "positive") {
      addHintEl.textContent = "画像の外側で離されました — 画像内でドラッグしてください";
    }
    return;
  }
  const bbox = [
    Math.max(0, Math.min(state.inputImage.width,  a[0])),
    Math.max(0, Math.min(state.inputImage.height, a[1])),
    Math.max(0, Math.min(state.inputImage.width,  b[0])),
    Math.max(0, Math.min(state.inputImage.height, b[1])),
  ];

  if (mode === "negative") {
    const p = state.persons.find((q) => q.id === state.activeId);
    if (!p) return;
    p.negativeBboxes = p.negativeBboxes || [];
    p.negativeBboxes.push(bbox);
    rebuildBboxOverlay();
    scheduleSam2Preview();
    return;
  }

  if (mode === "positive-additional") {
    const p = state.persons.find((q) => q.id === state.activeId);
    if (!p) return;
    p.additionalBboxes = p.additionalBboxes || [];
    p.additionalBboxes.push(bbox);
    rebuildBboxOverlay();
    scheduleSam2Preview();
    return;
  }

  if (mode === "object") {
    setDrawMode(false);
    _commitObjectFromBbox(bbox).catch((e) => {
      console.warn("object_crop failed:", e);
      addObjectHint.textContent = "切り抜き失敗: " + (e.message || e);
    });
    return;
  }

  if (mode === "object-positive-additional" || mode === "object-negative") {
    const o = state.objects.find((q) => q.id === state.activeObjectId);
    if (!o) return;
    if (mode === "object-positive-additional") {
      o.additionalBboxes = o.additionalBboxes || [];
      o.additionalBboxes.push(bbox);
    } else {
      o.negativeBboxes = o.negativeBboxes || [];
      o.negativeBboxes.push(bbox);
    }
    // Reflect the new rect on the image overlay immediately — the
    // recrop is async (debounced 350 ms) so without an explicit
    // rebuild here the user wouldn't see the dashed yellow / red bbox
    // appear right after the drag.
    rebuildBboxOverlay();
    rebuildObjectList();
    scheduleObjectRecrop(o.id);
    return;
  }

  // Positive (person bbox): assign to the pending person or add a new one.
  const pending = state.persons.find((p) => p.pendingBboxDraw);
  let addedId = null;
  if (pending) {
    pending.bbox = bbox;
    pending.pendingBboxDraw = false;
    addedId = pending.id;
  } else {
    const np = _newPerson();
    np.bbox = bbox;
    np.heightInputValue = displayFromMeters(state.unit.default_adult_height_m, state.unit.display_unit);
    np.heightMeters = state.unit.default_adult_height_m;
    state.persons.push(np);
    state.activeId = np.id;
    addedId = np.id;
  }
  setDrawMode(false);
  rebuildPersonList();
  rebuildBboxOverlay();
  refreshButtons();
  scheduleSam2Preview();
  if (addedId) {
    showAddedFlash(addedId);
  }
}
bboxOverlay.addEventListener("pointerup", _endDrag);
bboxOverlay.addEventListener("pointercancel", _endDrag);

function setDrawMode(on, type = "primary") {
  // drawMode=true is used for two flows:
  //   "primary" — drawing the primary bbox for a freshly added person
  //   "object"  — drawing a bbox to crop a 2D object overlay
  // After the drag commits the editor exits drawMode in both cases.
  state.drawMode = !!on;
  state.drawType = on ? type : "primary";
  document.body.classList.toggle("bbox-draw-mode", state.drawMode);
  if (on) {
    if (type === "object") {
      addHintEl.textContent = "画像上をドラッグして切り抜きたいオブジェクトの範囲を指定してください (Esc でキャンセル)";
    } else {
      addHintEl.textContent = "画像上をドラッグして人物の bbox を描いてください (Esc でキャンセル)";
    }
    if (state.view !== "image") setView("image");
  } else {
    addHintEl.textContent = "";
  }
  refreshFloatingAddBtn();
}

// Floating "+ 次の人物を追加" button shown over the image once at least
// one person has been bbox-completed. Gives a direct path to add the next
// person without scrolling back to the sidebar.
let _floatingAddBtn = null;
function refreshFloatingAddBtn() {
  const havePerson = state.persons.some((p) => p.bbox);
  const showFloating = havePerson && !state.drawMode && state.view === "image";
  if (showFloating) {
    if (!_floatingAddBtn) {
      _floatingAddBtn = document.createElement("button");
      _floatingAddBtn.id = "floating-add-btn";
      _floatingAddBtn.textContent = "+ 次の人物を追加";
      _floatingAddBtn.style.cssText = `
        position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
        z-index: 8; padding: 8px 16px; font-weight: 600;
        background: var(--accent); color: white; border: 1px solid var(--accent-2);
        border-radius: 6px; cursor: pointer; font-size: 13px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
      `;
      _floatingAddBtn.addEventListener("click", startAddPerson);
      viewportEl.appendChild(_floatingAddBtn);
    }
    _floatingAddBtn.style.display = "block";
  } else if (_floatingAddBtn) {
    _floatingAddBtn.style.display = "none";
  }
}

// Brief CSS pulse on the "+ Add Person" button + transient confirmation
// message after a bbox is committed.
function showAddedFlash(personId) {
  addPersonBtn.animate?.(
    [
      { boxShadow: "0 0 0 0 rgba(60,130,255,0.7)" },
      { boxShadow: "0 0 0 12px rgba(60,130,255,0)" },
    ],
    { duration: 700, iterations: 2 },
  );
  addHintEl.textContent = `✓ 人物 ${personId} を追加 — 続けて『${_floatingAddBtn ? "+ 次の人物を追加" : "+ Add Person"}』`;
  setTimeout(() => {
    if (addHintEl.textContent.startsWith("✓")) addHintEl.textContent = "";
  }, 3500);
}

// Shared "start adding a person" path — used by both the sidebar button
// and the floating viewport button.
function startAddPerson() {
  if (!state.inputImage) return;
  const np = _newPerson();
  np.pendingBboxDraw = true;
  np.heightInputValue = displayFromMeters(state.unit.default_adult_height_m, state.unit.display_unit);
  np.heightMeters = state.unit.default_adult_height_m;
  state.persons.push(np);
  state.activeId = np.id;
  setView("image");
  setDrawMode(true);
  rebuildPersonList();
  rebuildBboxOverlay();
  refreshButtons();
}

// ---------------------------------------------------------------------------
// Person list rendering
// ---------------------------------------------------------------------------

function rebuildPersonList() {
  // Persist after every visual rebuild — covers slider edits, height
  // changes, hand image drops, transform tweaks, and bbox add/delete.
  personListEl.innerHTML = "";
  // Slider DOM refs keyed by person id are about to be detached — clear
  // so the gizmo's transform-change handler doesn't write into stale
  // input elements no longer in the document.
  _transformSliderRefs.clear();
  if (state.persons.length === 0) {
    const m = document.createElement("div");
    m.className = "muted";
    m.textContent = "Add Person を押して人物の bbox を追加してください。";
    personListEl.appendChild(m);
  }

  // Quick-switch chip row — one click switches active focus to any
  // person without scrolling through their (potentially long) card.
  if (state.persons.length > 1) {
    const chipRow = document.createElement("div");
    chipRow.className = "person-quick-switch";
    for (const q of state.persons) {
      const chip = document.createElement("div");
      chip.className = "chip" + (q.id === state.activeId ? " active" : "");
      chip.textContent = q.id;
      chip.addEventListener("click", () => setActive(q.id));
      chipRow.appendChild(chip);
    }
    personListEl.appendChild(chipRow);
  }

  for (const p of state.persons) {
    const card = document.createElement("div");
    card.className = "person-card" + (p.id === state.activeId ? " active" : "");
    card.dataset.pid = p.id;

    // Head row
    const head = document.createElement("div");
    head.className = "head";
    const name = document.createElement("div");
    name.className = "pname";
    name.textContent = p.id;
    name.addEventListener("click", () => setActive(p.id));
    const actions = document.createElement("div");
    actions.className = "actions";
    const delBtn = document.createElement("button");
    delBtn.className = "btn btn-danger";
    delBtn.textContent = "削除";
    // pointerEvents: auto overrides the sidebar dim that ``body.bbox-draw-mode``
    // applies — without it, the user can't cancel a pending person while the
    // bbox-draw overlay is active. (The card-level click bubble is also
    // suppressed below to avoid a stale setActive on the deleted id.)
    delBtn.style.cssText = "padding: 2px 8px; pointer-events: auto;";
    delBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      ev.preventDefault();
      removePerson(p.id);
    });
    actions.appendChild(delBtn);
    head.appendChild(name);
    head.appendChild(actions);
    card.appendChild(head);

    // Bbox readout
    const bboxRO = document.createElement("div");
    bboxRO.className = "bbox-readout";
    if (p.bbox) {
      const [x1, y1, x2, y2] = p.bbox.map((v) => Math.round(v));
      const addCnt = (p.additionalBboxes?.length || 0);
      const negCnt = (p.negativeBboxes?.length || 0);
      const tags = [];
      if (addCnt > 0) tags.push(`+${addCnt} 正`);
      if (negCnt > 0) tags.push(`−${negCnt} 除外`);
      const tagStr = tags.length ? `  [${tags.join(", ")}]` : "";
      bboxRO.textContent = `bbox: (${x1},${y1})–(${x2},${y2})  ${x2 - x1}×${y2 - y1}px${tagStr}`;
    } else {
      bboxRO.textContent = "bbox 未指定 — Add Person 中にドラッグしてください";
    }
    card.appendChild(bboxRO);


    // Cropped preview thumbnail. Two flavours:
    //   1. SAM 2 masked crop (preferred) — populated by ``scheduleSam2Preview``
    //      a moment after the bbox is drawn. Shows what the inference path
    //      will actually consume (silhouette only, transparent background).
    //   2. Raw bbox crop fallback — synchronous canvas snip of the source
    //      image, shown immediately while SAM 2 is still running so the
    //      user gets instant feedback.
    if (p.bbox && state.inputImageElement) {
      const url = p.trimmedUrl || _ensureCroppedThumbnail(p);
      if (url) {
        const thumbWrap = document.createElement("div");
        thumbWrap.style.cssText = `
          display: flex; justify-content: center; align-items: center;
          background: #111; border: 1px solid var(--border); border-radius: 3px;
          padding: 4px; min-height: 60px; max-height: 140px; overflow: hidden;
          position: relative;
        `;
        const thumb = document.createElement("img");
        thumb.alt = `${p.id} trimmed preview`;
        thumb.style.cssText = "max-width: 100%; max-height: 130px; display: block; image-rendering: auto;";
        thumb.src = url;
        thumbWrap.appendChild(thumb);
        // Tiny badge so the user can tell which thumbnail flavour is on
        // screen (SAM 2 masked vs. raw bbox crop).
        const badge = document.createElement("div");
        badge.style.cssText = `
          position: absolute; top: 4px; right: 4px;
          padding: 1px 5px; font-size: 9px; line-height: 14px;
          background: rgba(0, 0, 0, 0.55); color: #cef;
          border-radius: 2px;
        `;
        badge.textContent = p.trimmedUrl ? "SAM2" : "bbox";
        thumbWrap.appendChild(badge);
        card.appendChild(thumbWrap);
      }
    }

    // Height row
    const hr = document.createElement("div");
    hr.className = "height-row";
    const hLabel = document.createElement("span");
    hLabel.textContent = "身長:";
    const hInput = document.createElement("input");
    hInput.type = "number";
    hInput.step = state.unit.display_unit === "cm" ? "0.5" : "0.05";
    hInput.placeholder = String(displayFromMeters(state.unit.default_adult_height_m, state.unit.display_unit));
    hInput.value = p.heightInputValue ?? "";
    hInput.addEventListener("input", () => {
      p.heightInputValue = hInput.value;
      const m = metersFromDisplay(hInput.value, state.unit.display_unit);
      if (m && m > 0.3 && m < 3.0) p.heightMeters = m;
      // Live re-render only if we already have a plus_job_id (post-inference).
      if (state.multiJobId) scheduleReRender();
    });
    const hUnit = document.createElement("span");
    hUnit.className = "unit-suffix";
    hUnit.textContent = unitSuffix();
    hr.appendChild(hLabel);
    hr.appendChild(hInput);
    hr.appendChild(hUnit);
    card.appendChild(hr);

    // Hand image rows
    for (const side of ["l", "r"]) {
      const row = document.createElement("div");
      row.className = "hand-row";
      const drop = document.createElement("label");
      drop.className = "hand-drop";
      drop.dataset.side = side;
      const fileInp = document.createElement("input");
      fileInp.type = "file";
      fileInp.accept = "image/*";
      const lbl = document.createElement("span");
      const img = document.createElement("img");
      img.alt = side === "l" ? "left hand" : "right hand";
      img.style.display = "none";
      const cur = side === "l" ? p.lhandImage : p.rhandImage;
      if (cur) {
        img.src = cur.dataUrl;
        img.style.display = "block";
        lbl.textContent = `${cur.width}×${cur.height}`;
      } else {
        lbl.textContent = side === "l" ? "Left hand image" : "Right hand image";
      }
      drop.appendChild(fileInp);
      drop.appendChild(lbl);
      drop.appendChild(img);
      const flipBtn = document.createElement("button");
      flipBtn.className = "flip-btn";
      flipBtn.textContent = "左右反転";
      flipBtn.disabled = !cur;
      const handleFile = async (f) => {
        if (!f) return;
        try {
          const cap = await decodeImage(f);
          if (side === "l") p.lhandImage = cap; else p.rhandImage = cap;
          rebuildPersonList();
        } catch (e) { console.warn("hand decode failed:", e); }
      };
      fileInp.addEventListener("change", (e) => handleFile(e.target.files?.[0]));
      ["dragenter", "dragover"].forEach((ev) =>
        drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragover"); })
      );
      ["dragleave", "drop"].forEach((ev) =>
        drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("dragover"); })
      );
      drop.addEventListener("drop", (e) => {
        e.preventDefault();
        handleFile(e.dataTransfer?.files?.[0]);
      });
      flipBtn.addEventListener("click", async () => {
        const cur2 = side === "l" ? p.lhandImage : p.rhandImage;
        if (!cur2) return;
        const flipped = await mirrorDataUrl(cur2.dataUrl);
        if (side === "l") p.lhandImage = flipped; else p.rhandImage = flipped;
        rebuildPersonList();
      });
      row.appendChild(drop);
      row.appendChild(flipBtn);
      card.appendChild(row);
    }

    // Edit panel (lean + IK + body preset) — only after inference, always
    // visible (no expand/collapse toggle to keep the UI flat like the
    // single Pose Editor's sidebar).
    if (p.settings) {
      card.appendChild(_buildEditPanel(p));
    }

    // Whole-card click switches active. Buttons / inputs inside that
    // need to NOT activate already call ``ev.stopPropagation()`` (削除,
    // + 正領域, − 除外領域, etc.); everything else just promotes to
    // active, which is idempotent and harmless.
    if (p.id !== state.activeId) {
      card.addEventListener("click", () => setActive(p.id));
    }
    personListEl.appendChild(card);
  }
}

// Build the per-person edit panel — flat layout matching the single
// Pose Editor's sidebar:
//   ポーズ補正: lean slider + "ポーズ調整(回転・移動)" button (enters IK mode)
//   Body Preset: preset Load + body preset JSON drop
function _buildEditPanel(p) {
  const wrap = document.createElement("div");
  wrap.style.cssText = `
    margin-top: 6px; padding: 8px; background: #232323;
    border: 1px solid #383838; border-radius: 4px;
    display: flex; flex-direction: column; gap: 8px;
  `;

  // ===== ポーズ補正 section =====
  const poseHeader = document.createElement("div");
  poseHeader.style.cssText = "font-size: 11px; font-weight: 600; color: var(--muted); margin-top: 2px;";
  poseHeader.textContent = "ポーズ補正";
  wrap.appendChild(poseHeader);

  // Lean correction slider
  const leanRow = document.createElement("div");
  leanRow.className = "row";
  const leanLabel = document.createElement("span");
  leanLabel.style.cssText = "font-size: 11px; color: var(--muted); min-width: 90px;";
  leanLabel.textContent = "前かがみ補正:";
  const leanRange = document.createElement("input");
  leanRange.type = "range";
  leanRange.min = "0"; leanRange.max = "1"; leanRange.step = "0.01";
  leanRange.style.flex = "1 1 auto";
  leanRange.value = String(p.settings?.pose_adjust?.lean_correction ?? 0);
  const leanNum = document.createElement("input");
  leanNum.type = "number";
  leanNum.min = "0"; leanNum.max = "1"; leanNum.step = "0.01";
  leanNum.style.cssText = "width: 60px; background: #1a1a1a; border: 1px solid var(--border); color: var(--text); padding: 2px 4px; border-radius: 3px;";
  leanNum.value = leanRange.value;
  const leanCommit = (v) => {
    const x = Math.max(0, Math.min(1, parseFloat(v) || 0));
    leanRange.value = String(x);
    leanNum.value = String(x);
    if (!p.settings) p.settings = _emptySettings();
    p.settings.pose_adjust = p.settings.pose_adjust || {};
    p.settings.pose_adjust.lean_correction = x;
    scheduleReRender();
  };
  leanRange.addEventListener("input", () => leanCommit(leanRange.value));
  leanNum.addEventListener("input", () => leanCommit(leanNum.value));
  leanRow.appendChild(leanLabel);
  leanRow.appendChild(leanRange);
  leanRow.appendChild(leanNum);
  wrap.appendChild(leanRow);

  // "ポーズ調整(回転・移動)" — toggles IK mode for the active person.
  // While the mode is on, the 3D viewport gets IK handles for the hands
  // and feet (drag → solves shoulder/elbow / hip/knee), and the sidebar
  // shows the ported single-editor bone-rotation panel for fine-grained
  // X/Y/Z tweaks per bone, plus reset buttons.
  const poseAdjActive = _ikMode && state.activeId === p.id;
  const poseAdjBtn = document.createElement("button");
  poseAdjBtn.className = "btn" + (poseAdjActive ? " btn-primary" : "");
  poseAdjBtn.style.padding = "5px 10px";
  poseAdjBtn.textContent = poseAdjActive
    ? "ポーズ調整を終了"
    : "ポーズ調整(回転・移動)";
  poseAdjBtn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    setActive(p.id);
    setIkMode(!poseAdjActive);
  });
  wrap.appendChild(poseAdjBtn);

  // Bone-rotation editor — ported from the single Pose Editor's
  // ``pose-edit-panel``. Visible only while pose-adjust (IK) mode is on
  // for this person, mirroring the single editor where the panel is
  // tucked behind the "ポーズ調整(回転)" button.
  if (poseAdjActive) {
    wrap.appendChild(_buildPoseEditPanel(p));
  }

  // ===== Per-person export (FBX / BVH) =====
  // Sits directly under the "ポーズ補正" area so the user can grab a
  // rigged file for the person whose pose they just tweaked. The Blender
  // path comes from the ComfyUI node's ``blender_exe`` widget (forwarded
  // via the iframe URL); writing to ``ComfyUI/output/`` is handled by
  // the /sam3d/api/plus/export_{fbx,bvh} endpoints.
  wrap.appendChild(_buildPerPersonExportRow(p));

  // ===== Body Preset section =====
  const bodyPresetHeader = document.createElement("div");
  bodyPresetHeader.style.cssText = "font-size: 11px; font-weight: 600; color: var(--muted); margin-top: 6px;";
  bodyPresetHeader.textContent = "Body Preset";
  wrap.appendChild(bodyPresetHeader);

  // Preset row — dropdown + Load button
  const presetRow = document.createElement("div");
  presetRow.className = "row";
  const presetLabel = document.createElement("span");
  presetLabel.style.cssText = "font-size: 11px; color: var(--muted); flex: 0 0 auto;";
  presetLabel.textContent = "Preset:";
  const presetSelect = document.createElement("select");
  presetSelect.style.cssText = `
    flex: 1 1 auto; background: #1a1a1a; color: var(--text);
    border: 1px solid var(--border); border-radius: 3px; padding: 3px;
    font-size: 11px;
  `;
  const opt0 = document.createElement("option");
  opt0.value = ""; opt0.textContent = "(preset を選択)";
  presetSelect.appendChild(opt0);
  for (const name of _presetNames) {
    const o = document.createElement("option");
    o.value = name; o.textContent = name;
    presetSelect.appendChild(o);
  }
  const presetLoadBtn = document.createElement("button");
  presetLoadBtn.className = "btn";
  presetLoadBtn.textContent = "Load";
  presetLoadBtn.style.padding = "3px 10px";
  presetLoadBtn.addEventListener("click", async (ev) => {
    ev.stopPropagation();
    const name = presetSelect.value;
    if (!name) return;
    try {
      const r = await fetch(`/sam3d/api/preset/${encodeURIComponent(name)}`);
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
      _applyBodyPresetJson(p, j);
      runInfoEl.textContent = `${p.id}: preset "${name}" を適用`;
    } catch (e) {
      console.warn("preset load failed:", e);
      alert("プリセットのロードに失敗しました: " + e.message);
    }
  });
  presetRow.appendChild(presetLabel);
  presetRow.appendChild(presetSelect);
  presetRow.appendChild(presetLoadBtn);
  wrap.appendChild(presetRow);

  // Body Preset JSON file drop — applies body_params/bone_lengths/blendshapes
  const bodyPresetDrop = document.createElement("label");
  bodyPresetDrop.className = "file-drop";
  bodyPresetDrop.style.cssText = "padding: 6px; font-size: 11px; cursor: pointer;";
  const bodyPresetInput = document.createElement("input");
  bodyPresetInput.type = "file";
  bodyPresetInput.accept = ".json,application/json";
  const bodyPresetLabel = document.createElement("span");
  bodyPresetLabel.textContent = "ボディプリセット JSON をアップロード (任意)";
  bodyPresetDrop.appendChild(bodyPresetInput);
  bodyPresetDrop.appendChild(bodyPresetLabel);
  const handleBodyPresetFile = async (file) => {
    if (!file) return;
    try {
      const text = await file.text();
      const j = JSON.parse(text);
      _applyBodyPresetJson(p, j);
      bodyPresetLabel.textContent = `${file.name} を適用`;
      runInfoEl.textContent = `${p.id}: ${file.name} を適用`;
    } catch (e) {
      console.warn("body preset json load failed:", e);
      alert("ボディプリセット JSON の読込に失敗: " + e.message);
    }
  };
  bodyPresetInput.addEventListener("change", (ev) => {
    ev.stopPropagation();
    handleBodyPresetFile(ev.target.files?.[0]);
  });
  ["dragenter", "dragover"].forEach((evname) =>
    bodyPresetDrop.addEventListener(evname, (ev) => {
      ev.preventDefault(); bodyPresetDrop.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evname) =>
    bodyPresetDrop.addEventListener(evname, (ev) => {
      ev.preventDefault(); bodyPresetDrop.classList.remove("dragover");
    })
  );
  bodyPresetDrop.addEventListener("drop", (ev) => {
    ev.preventDefault();
    handleBodyPresetFile(ev.dataTransfer?.files?.[0]);
  });
  wrap.appendChild(bodyPresetDrop);

  return wrap;
}

// ---------------------------------------------------------------------------
// Per-person FBX / BVH export row.
//
// Each button POSTs to /sam3d/api/plus/export_{fbx,bvh} which spawns
// Blender as a subprocess and writes the rigged file to
// ``ComfyUI/output/sam3d_pose_plus_<person_id>_<timestamp>.<ext>``. While
// the request is in flight the buttons disable themselves and the status
// line shows progress; on completion it shows the produced filename.
// ---------------------------------------------------------------------------
function _buildPerPersonExportRow(p) {
  const wrap = document.createElement("div");
  wrap.style.cssText =
    "margin-top: 4px; display: flex; flex-direction: column; gap: 4px;";

  const row = document.createElement("div");
  row.className = "row";
  row.style.gap = "6px";

  const fbxBtn = document.createElement("button");
  fbxBtn.className = "btn";
  fbxBtn.style.padding = "5px 10px";
  fbxBtn.textContent = "FBX をダウンロード";
  fbxBtn.title = "この人物の rigged FBX をブラウザにダウンロードする (ComfyUI/output/ にもコピーが残ります)";

  const bvhBtn = document.createElement("button");
  bvhBtn.className = "btn";
  bvhBtn.style.padding = "5px 10px";
  bvhBtn.textContent = "BVH をダウンロード";
  bvhBtn.title = "この人物の rigged BVH をブラウザにダウンロードする (ComfyUI/output/ にもコピーが残ります)";

  row.appendChild(fbxBtn);
  row.appendChild(bvhBtn);
  wrap.appendChild(row);

  const info = document.createElement("div");
  info.className = "muted";
  info.style.cssText = "font-size: 10px; min-height: 12px;";
  wrap.appendChild(info);

  // Disabled until inference produced a multi job id (settings / skeleton
  // are derived from the cached session).
  const ready = !!state.multiJobId && !!p.settings;
  fbxBtn.disabled = !ready;
  bvhBtn.disabled = !ready;
  if (!ready) {
    info.textContent = "推論実行後に出力できます";
  }

  async function runExport(fmt) {
    if (!state.multiJobId) {
      alert("先に『推論実行 / Run Inference』を押してください。");
      return;
    }
    if (!_blenderExe) {
      alert(
        "Blender のパスが設定されていません。\n"
        + "ComfyUI ノードの blender_exe フィールドに blender.exe のパスを"
        + "入力してから再度お試しください。",
      );
      return;
    }
    fbxBtn.disabled = true;
    bvhBtn.disabled = true;
    const label = fmt.toUpperCase();
    info.textContent = `${label} を出力中… (Blender 実行中)`;
    try {
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      const filename = `sam3d_pose_plus_${p.id}_${stamp}.${fmt}`;
      const r = await fetch(`/sam3d/api/plus/export_${fmt}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          plus_job_id:     state.multiJobId,
          person_id:       p.id,
          blender_exe:     _blenderExe,
          output_filename: filename,
        }),
      });
      if (!r.ok) {
        // Error path — backend returned JSON, not the binary file.
        const j = await r.json().catch(() => ({}));
        throw new Error(j.error || `HTTP ${r.status}`);
      }
      // Success path — backend streamed the produced file. Pull the
      // bytes into a Blob and synthesise an <a download> click so the
      // browser pops its native save dialog.
      const blob = await r.blob();
      const serverFilename = r.headers.get("X-Sam3d-Output-Filename") || filename;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = serverFilename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      // Revoke after the click handler had a chance to start the
      // download; small delay covers slow browsers.
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      info.textContent = `${label} ダウンロード済: ${serverFilename}`;
    } catch (e) {
      console.warn(`[pose_editor_plus] ${fmt} export failed:`, e);
      info.textContent = `${label} 出力に失敗: ${e.message || e}`;
      alert(`${label} 出力に失敗しました:\n${e.message || e}`);
    } finally {
      fbxBtn.disabled = false;
      bvhBtn.disabled = false;
    }
  }

  fbxBtn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    setActive(p.id);
    runExport("fbx");
  });
  bvhBtn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    setActive(p.id);
    runExport("bvh");
  });

  return wrap;
}

// Apply a parsed body preset settings dict (preset or user-uploaded JSON)
// to the person's slider state. ``pose_adjust`` is preserved — this
// touches only body-preset fields (body_params / bone_lengths /
// blendshapes).
function _applyBodyPresetJson(p, j) {
  if (!j || typeof j !== "object") return;
  p.settings = p.settings || _emptySettings();
  if (j.body_params)  p.settings.body_params  = j.body_params;
  if (j.bone_lengths) p.settings.bone_lengths = j.bone_lengths;
  if (j.blendshapes)  p.settings.blendshapes  = j.blendshapes;
  scheduleReRender();
}

function _emptySettings() {
  return {
    body_params: {}, bone_lengths: {}, blendshapes: {},
    pose_adjust: { lean_correction: 0, rotation_overrides: {} },
  };
}

// Whole-body translate (X/Y/Z meters) + rotate (X/Y/Z degrees) sub-panel.
// Sliders drive the per-person Three.js anchor directly (no backend round-
// trip) and register their DOM refs in ``_transformSliderRefs`` so the
// gizmo can push values back into them without a full list rebuild.
function _buildTransformPanel(p) {
  const panel = document.createElement("div");
  panel.style.cssText = `
    padding: 6px; background: #1a1a1a; border: 1px solid var(--border);
    border-radius: 3px; display: flex; flex-direction: column; gap: 6px;
  `;

  if (!p.transform) {
    p.transform = { translate: [0, 0, 0], rotate_deg: [0, 0, 0] };
  }

  // Per-person slider ref bucket — populated as we build each row.
  const refs = { translate: [null, null, null], rotate: [null, null, null] };
  _transformSliderRefs.set(p.id, refs);

  function _slider(labelText, group, axis, min, max, step, decimals) {
    const row = document.createElement("div");
    row.className = "row";
    const lbl = document.createElement("span");
    lbl.style.cssText = "font-size: 11px; color: var(--muted); min-width: 70px;";
    lbl.textContent = labelText;
    const rng = document.createElement("input");
    rng.type = "range";
    rng.min = String(min); rng.max = String(max); rng.step = String(step);
    rng.style.flex = "1 1 auto";
    const cur = group === "translate" ? p.transform.translate[axis] : p.transform.rotate_deg[axis];
    rng.value = String(cur.toFixed(decimals));
    const num = document.createElement("input");
    num.type = "number";
    num.min = String(min); num.max = String(max); num.step = String(step);
    num.style.cssText = "width: 64px; background: #1a1a1a; border: 1px solid var(--border); color: var(--text); padding: 2px 4px; border-radius: 3px;";
    num.value = rng.value;
    const commit = (v) => {
      const x = Math.max(min, Math.min(max, parseFloat(v) || 0));
      rng.value = String(x.toFixed(decimals));
      num.value = String(x.toFixed(decimals));
      if (group === "translate") {
        p.transform.translate[axis] = x;
      } else {
        p.transform.rotate_deg[axis] = x;
      }
      // Frontend-only: push the new transform straight into the anchor.
      // No backend re-render — geometry doesn't change for transform edits.
      viewer.writeAnchorTransform(p.id, p.transform.translate, p.transform.rotate_deg);
    };
    rng.addEventListener("input", () => commit(rng.value));
    num.addEventListener("input", () => commit(num.value));
    row.appendChild(lbl); row.appendChild(rng); row.appendChild(num);
    if (group === "translate") refs.translate[axis] = { range: rng, num };
    else                       refs.rotate[axis]    = { range: rng, num };
    return row;
  }

  const tHeader = document.createElement("div");
  tHeader.className = "muted";
  tHeader.style.fontSize = "11px";
  tHeader.textContent = "位置 (m)";
  panel.appendChild(tHeader);
  panel.appendChild(_slider("X (横)",   "translate", 0, -2, 2, 0.01, 2));
  panel.appendChild(_slider("Y (高さ)", "translate", 1, -2, 2, 0.01, 2));
  panel.appendChild(_slider("Z (奥行)", "translate", 2, -2, 2, 0.01, 2));

  const rHeader = document.createElement("div");
  rHeader.className = "muted";
  rHeader.style.fontSize = "11px";
  rHeader.style.marginTop = "4px";
  rHeader.textContent = "回転 (°) — 腰中心";
  panel.appendChild(rHeader);
  panel.appendChild(_slider("X (前後)",  "rotate", 0, -180, 180, 0.5, 1));
  panel.appendChild(_slider("Y (左右)",  "rotate", 1, -180, 180, 0.5, 1));
  panel.appendChild(_slider("Z (傾き)",  "rotate", 2, -180, 180, 0.5, 1));

  const resetRow = document.createElement("div");
  resetRow.className = "row";
  const resetBtn = document.createElement("button");
  resetBtn.className = "btn btn-danger";
  resetBtn.style.padding = "3px 8px";
  resetBtn.textContent = "位置/回転をリセット";
  resetBtn.addEventListener("click", () => {
    p.transform = { translate: [0, 0, 0], rotate_deg: [0, 0, 0] };
    viewer.writeAnchorTransform(p.id, p.transform.translate, p.transform.rotate_deg);
    rebuildPersonList();
  });
  resetRow.appendChild(resetBtn);
  panel.appendChild(resetRow);

  return panel;
}

function _buildPoseEditPanel(p) {
  const panel = document.createElement("div");
  panel.style.cssText = `
    padding: 6px; background: #1a1a1a; border: 1px solid var(--border);
    border-radius: 3px; display: flex; flex-direction: column; gap: 6px;
  `;

  if (!p.skeleton || !p.skeleton.bones?.length) {
    const m = document.createElement("div");
    m.className = "muted";
    m.textContent = "推論後にボーン情報が利用できます。";
    panel.appendChild(m);
    return panel;
  }

  // Bone selector
  const boneRow = document.createElement("div");
  boneRow.className = "row";
  const boneLabel = document.createElement("span");
  boneLabel.style.cssText = "font-size: 11px; color: var(--muted); min-width: 60px;";
  boneLabel.textContent = "ボーン:";
  const boneSelect = document.createElement("select");
  boneSelect.style.cssText = `
    flex: 1 1 auto; background: #1a1a1a; color: var(--text);
    border: 1px solid var(--border); border-radius: 3px; padding: 3px;
    font-size: 11px;
  `;
  for (const b of p.skeleton.bones) {
    const o = document.createElement("option");
    o.value = b.name;
    const hasOverride = !!p.settings?.pose_adjust?.rotation_overrides?.[String(b.joint_id)];
    o.textContent = (hasOverride ? "● " : "  ") + b.name;
    boneSelect.appendChild(o);
  }
  boneSelect.value = p.selectedBoneName || p.skeleton.bones[0].name;
  boneRow.appendChild(boneLabel);
  boneRow.appendChild(boneSelect);
  panel.appendChild(boneRow);

  // X/Y/Z sliders
  const overrides = p.settings?.pose_adjust?.rotation_overrides || {};
  function _readEulerForBone(name) {
    const bone = p.skeleton.bones.find((b) => b.name === name);
    if (!bone) return [0, 0, 0];
    const cur = overrides[String(bone.joint_id)];
    if (Array.isArray(cur) && cur.length === 3) return cur.map((v) => v * 180 / Math.PI);
    return [0, 0, 0];
  }
  function _writeEulerForBone(name, eulerDeg) {
    const bone = p.skeleton.bones.find((b) => b.name === name);
    if (!bone) return;
    if (!p.settings) p.settings = _emptySettings();
    p.settings.pose_adjust = p.settings.pose_adjust || {};
    p.settings.pose_adjust.rotation_overrides = p.settings.pose_adjust.rotation_overrides || {};
    const ov = p.settings.pose_adjust.rotation_overrides;
    const radians = eulerDeg.map((d) => d * Math.PI / 180);
    if (radians.every((v) => Math.abs(v) < 1e-8)) {
      delete ov[String(bone.joint_id)];
    } else {
      ov[String(bone.joint_id)] = radians;
    }
  }

  const axes = [
    { label: "X 軸 (°)", key: 0 },
    { label: "Y 軸 (°)", key: 1 },
    { label: "Z 軸 (°)", key: 2 },
  ];
  const ranges = [];
  const nums = [];
  for (const ax of axes) {
    const row = document.createElement("div");
    row.className = "row";
    const lbl = document.createElement("span");
    lbl.style.cssText = "font-size: 11px; color: var(--muted); min-width: 60px;";
    lbl.textContent = ax.label;
    const rng = document.createElement("input");
    rng.type = "range";
    rng.min = "-180"; rng.max = "180"; rng.step = "0.5";
    rng.style.flex = "1 1 auto";
    const num = document.createElement("input");
    num.type = "number";
    num.min = "-180"; num.max = "180"; num.step = "0.5";
    num.style.cssText = "width: 60px; background: #1a1a1a; border: 1px solid var(--border); color: var(--text); padding: 2px 4px; border-radius: 3px;";
    row.appendChild(lbl); row.appendChild(rng); row.appendChild(num);
    panel.appendChild(row);
    ranges.push(rng);
    nums.push(num);
  }
  function syncSliders() {
    const cur = _readEulerForBone(boneSelect.value);
    for (let i = 0; i < 3; i++) {
      ranges[i].value = String(cur[i].toFixed(1));
      nums[i].value = String(cur[i].toFixed(1));
    }
  }
  function commit() {
    const eulerDeg = ranges.map((r) => parseFloat(r.value) || 0);
    _writeEulerForBone(boneSelect.value, eulerDeg);
    // Push the slider value into the viewer's pose-edit overlay too so
    // the bone handle moves in real time (single Pose Editor parity).
    if (_ikMode && viewer.isPoseEditActive()) {
      const r = eulerDeg.map((d) => d * Math.PI / 180);
      viewer.setPoseBoneLocalEuler(boneSelect.value, r[0], r[1], r[2]);
    }
    triggerReRender();
  }
  for (let i = 0; i < 3; i++) {
    ranges[i].addEventListener("input", () => {
      nums[i].value = ranges[i].value;
      commit();
    });
    nums[i].addEventListener("input", () => {
      ranges[i].value = nums[i].value;
      commit();
    });
  }
  boneSelect.addEventListener("change", () => {
    p.selectedBoneName = boneSelect.value;
    if (_ikMode && viewer.isPoseEditActive()) {
      viewer.selectPoseBone(boneSelect.value);
    }
    syncSliders();
  });
  syncSliders();

  // Reset buttons
  const resetRow = document.createElement("div");
  resetRow.className = "row";
  const resetBoneBtn = document.createElement("button");
  resetBoneBtn.className = "btn";
  resetBoneBtn.style.padding = "3px 8px";
  resetBoneBtn.textContent = "このボーンをリセット";
  resetBoneBtn.addEventListener("click", () => {
    _writeEulerForBone(boneSelect.value, [0, 0, 0]);
    if (_ikMode && viewer.isPoseEditActive()) {
      viewer.resetPoseBone(boneSelect.value);
    }
    syncSliders();
    triggerReRender();
  });
  const resetAllBtn = document.createElement("button");
  resetAllBtn.className = "btn btn-danger";
  resetAllBtn.style.padding = "3px 8px";
  resetAllBtn.textContent = "全ボーンをリセット";
  resetAllBtn.addEventListener("click", () => {
    if (!p.settings) p.settings = _emptySettings();
    p.settings.pose_adjust = p.settings.pose_adjust || {};
    p.settings.pose_adjust.rotation_overrides = {};
    if (_ikMode && viewer.isPoseEditActive()) {
      viewer.resetAllPoseBones();
    }
    syncSliders();
    triggerReRender();
  });
  resetRow.appendChild(resetBoneBtn);
  resetRow.appendChild(resetAllBtn);
  panel.appendChild(resetRow);

  return panel;
}

// Synchronous bbox crop using a pre-decoded HTMLImageElement. Avoids the
// race between async-Image-load and rebuildPersonList re-running before
// the resolved promise can find a still-attached <img> in the DOM.
function _cropSync(bbox, maxSize) {
  const im = state.inputImageElement;
  if (!im || !bbox || bbox.length !== 4) return null;
  if (!im.naturalWidth || !im.naturalHeight) return null;
  const [x1, y1, x2, y2] = bbox.map((v) => Math.max(0, Math.round(v)));
  const cw = Math.max(1, Math.min(im.naturalWidth - x1, x2 - x1));
  const ch = Math.max(1, Math.min(im.naturalHeight - y1, y2 - y1));
  if (cw <= 0 || ch <= 0) return null;
  const longSide = Math.max(cw, ch);
  const scale = longSide > maxSize ? maxSize / longSide : 1;
  const ow = Math.max(1, Math.round(cw * scale));
  const oh = Math.max(1, Math.round(ch * scale));
  const off = document.createElement("canvas");
  off.width = ow; off.height = oh;
  try {
    off.getContext("2d").drawImage(im, x1, y1, cw, ch, 0, 0, ow, oh);
    return off.toDataURL("image/png");
  } catch (e) {
    console.warn("[pose_editor_plus] crop failed:", e);
    return null;
  }
}

function _ensureCroppedThumbnail(p) {
  if (!p.bbox || !state.inputImageElement) return null;
  const key = p.bbox.map((v) => Math.round(v)).join(",");
  if (p._croppedKey === key && p._croppedUrl) return p._croppedUrl;
  const url = _cropSync(p.bbox, 160);
  if (url) {
    p._croppedKey = key;
    p._croppedUrl = url;
  }
  return url;
}

async function mirrorDataUrl(dataUrl) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => {
      const off = document.createElement("canvas");
      off.width = im.naturalWidth; off.height = im.naturalHeight;
      const ctx = off.getContext("2d");
      ctx.translate(off.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(im, 0, 0);
      const url = off.toDataURL("image/png");
      resolve({ dataUrl: url, width: off.width, height: off.height, blob: dataUrlToBlob(url) });
    };
    im.onerror = () => reject(new Error("mirror decode failed"));
    im.src = dataUrl;
  });
}

function setActive(id) {
  // Skip the rebuild when nothing actually changes — keeps the captured
  // pointer alive when the click handler fires inside an existing drag.
  if (state.activeId === id && !state.activeObjectId) return;
  state.activeId = id;
  // Mutual exclusion: editing a person deactivates the current object so
  // mouse drags target the person's bboxes, not the object's.
  state.activeObjectId = null;
  rebuildPersonList();
  rebuildBboxOverlay();
  rebuildObjectList();
  rebuildObjectOverlay();
  _refreshNegHint();
  _refreshViewportToolbar();
  // Attach the gizmo to the newly active person (if a render exists).
  if (viewer.getAnchorCount() > 0) {
    if (_ikMode) {
      // IK mode follows the active person — rebuild the pose-edit
      // overlay against that person's skeleton.
      refreshPoseEditForActive();
    } else {
      viewer.setActiveAnchor(id);
    }
  }
}

function removePerson(id) {
  console.log("[pose_editor_plus] removePerson", id, "before:", state.persons.map((p) => p.id));
  const idx = state.persons.findIndex((p) => p.id === id);
  if (idx < 0) {
    console.warn("[pose_editor_plus] removePerson: id not found", id);
    return;
  }

  // Compute where the active person will land in the post-splice + post-
  // renumber array (or -1 if the active person was the one being deleted).
  let activeIdxAfter = -1;
  if (state.activeId && state.activeId !== id) {
    const oldActiveIdx = state.persons.findIndex((p) => p.id === state.activeId);
    if (oldActiveIdx > idx)        activeIdxAfter = oldActiveIdx - 1;
    else if (oldActiveIdx >= 0)    activeIdxAfter = oldActiveIdx;
  }

  // Snapshot the kept old IDs (in their post-splice order) BEFORE we
  // renumber so the backend can re-key its cached slots to match.
  const keepOldIds = state.persons
    .filter((p) => p.id !== id)
    .map((p) => p.id);

  state.persons.splice(idx, 1);

  // Renumber: ``p0, p1, p2 ...`` reflects the current array order so new
  // persons added after a delete don't skip over the gap. The shared
  // ``_personCounter`` is reset to the live length.
  state.persons.forEach((p, i) => { p.id = `p${i}`; });
  _personCounter = state.persons.length;

  // Rebind active id against the renumbered array.
  if (activeIdxAfter >= 0 && activeIdxAfter < state.persons.length) {
    state.activeId = state.persons[activeIdxAfter].id;
  } else {
    state.activeId = state.persons[0]?.id || null;
  }
  console.log("[pose_editor_plus] removePerson after:", state.persons.map((p) => p.id));

  // Cancel any in-progress drag for this person.
  if (state.drawMode) {
    setDrawMode(false);
    bboxRectActive.hidden = true;
    _drag = null;
  }

  // If we have a cached inference, reconcile the backend session in place
  // so the survivors stay rendered while the deleted slot disappears.
  // Falling back to a full session drop only when there are no people
  // left or the reconcile API is unavailable keeps the bug-fix path
  // minimal: deleting one person no longer wipes the others from 3D.
  if (state.multiJobId) {
    const jobId = state.multiJobId;
    if (state.persons.length === 0) {
      state.multiJobId = null;
      fetch("/sam3d/api/plus/drop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plus_job_id: jobId }),
      }).catch(() => {});
      try { viewer.clearAnchors(); } catch (e) { console.warn("clearAnchors failed:", e); }
      overlayEl.hidden = false;
    } else {
      fetch("/sam3d/api/plus/reconcile_persons", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plus_job_id: jobId, keep_old_ids: keepOldIds }),
      })
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          // Backend slots renamed — kick a render so the merged OBJ
          // drops the deleted person and survivors stay on screen.
          scheduleReRender();
        })
        .catch((e) => {
          console.warn("[pose_editor_plus] reconcile failed; dropping session:", e);
          state.multiJobId = null;
          fetch("/sam3d/api/plus/drop", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ plus_job_id: jobId }),
          }).catch(() => {});
          try { viewer.clearAnchors(); } catch (_e) {}
          overlayEl.hidden = false;
          rebuildPersonList();
        });
    }
  }
  try {
    rebuildPersonList();
    rebuildBboxOverlay();
    refreshButtons();
    refreshFloatingAddBtn();
  } catch (e) {
    console.error("[pose_editor_plus] rebuild after removePerson failed:", e);
  }
  // Removing a person changes the negative-prompt set for the survivors —
  // re-run preview so their masks reflect the smaller crowd.
  if (state.persons.some((p) => p.bbox)) {
    scheduleSam2Preview();
  }
}

function refreshButtons() {
  const haveImage = !!state.inputImage;
  const havePersons = state.persons.length > 0;
  const allBboxed = havePersons && state.persons.every((p) => p.bbox);
  addPersonBtn.disabled = !haveImage || state.drawMode || state.inferenceRunning;
  addObjectBtn.disabled = !haveImage || state.drawMode || state.inferenceRunning;
  runBtn.disabled = !haveImage || !allBboxed || state.inferenceRunning;
  confirmBtn.disabled = !state.multiJobId || state.inferenceRunning;
  _refreshViewportToolbar();
}

// The viewport's right-side toolbar (translate / rotate gizmo + their
// resets, image-preview toggle) only operates on the active body anchor
// in 3D view; hide it on the image tab, when no person is selected, or
// while pose-edit (IK) is active — in IK mode the gizmo is repurposed
// for bones, so the body-anchor controls would just confuse the user.
function _refreshViewportToolbar() {
  const tb = document.querySelector(".viewport-toolbar");
  if (!tb) return;
  const show = state.view === "3d" && !!state.activeId && !_ikMode;
  tb.style.display = show ? "flex" : "none";
}

// ---------------------------------------------------------------------------
// File upload
// ---------------------------------------------------------------------------

async function setInputImage(file) {
  if (!file) return;
  try {
    const cap = await decodeImage(file);
    state.inputImage = cap;
    state.inputBlob = cap.blob;
    // Pre-decode an HTMLImageElement so per-person thumbnail crops can
    // run synchronously inside rebuildPersonList without an async race.
    const im = new Image();
    await new Promise((resolve, reject) => {
      im.onload = () => resolve();
      im.onerror = () => reject(new Error("input image decode failed"));
      im.src = cap.dataUrl;
    });
    state.inputImageElement = im;

    fileLabelEl.textContent = `${cap.width}×${cap.height}`;
    previewEl.src = cap.dataUrl;
    previewEl.hidden = false;
    imageInfoEl.textContent = `${file.name || "image"} — ${cap.width}×${cap.height}`;
    imageDisplayImg.src = cap.dataUrl;
    if (state.view === "image") setView("image");  // refresh layout
    refreshButtons();
    // Reset persons / multi job / objects — a new image invalidates everything.
    state.persons = [];
    state.activeId = null;
    state.multiJobId = null;
    state.objects = [];
    state.activeObjectId = null;
    viewer.clearAnchors();
    viewer.resetCohortLayoutCache();
    viewer.resetFramedFingerprint();
    overlayEl.hidden = false;
    rebuildPersonList();
    rebuildBboxOverlay();
    rebuildObjectList();
    rebuildObjectOverlay();
  } catch (e) {
    console.warn("image load failed:", e);
    alert("画像の読み込みに失敗しました: " + e.message);
  }
}

fileInputEl.addEventListener("change", (e) => setInputImage(e.target.files?.[0]));
["dragenter", "dragover"].forEach((ev) =>
  fileDropEl.addEventListener(ev, (e) => { e.preventDefault(); fileDropEl.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((ev) =>
  fileDropEl.addEventListener(ev, (e) => { e.preventDefault(); fileDropEl.classList.remove("dragover"); })
);
fileDropEl.addEventListener("drop", (e) => {
  e.preventDefault();
  setInputImage(e.dataTransfer?.files?.[0]);
});

// ---------------------------------------------------------------------------
// Add Person
// ---------------------------------------------------------------------------

addPersonBtn.addEventListener("click", startAddPerson);

addObjectBtn.addEventListener("click", () => {
  if (!state.inputImage) return;
  setDrawMode(true, "object");
  addObjectHint.textContent = "画像上をドラッグして切り抜き範囲を指定";
});

// ---------------------------------------------------------------------------
// Range select (capture region for the composite pose_image)
// ---------------------------------------------------------------------------

function _applyRangeRectVisual() {
  if (state.captureRange) {
    rangeRectEl.hidden = false;
    rangeRectEl.style.left   = `${state.captureRange.x}px`;
    rangeRectEl.style.top    = `${state.captureRange.y}px`;
    rangeRectEl.style.width  = `${state.captureRange.w}px`;
    rangeRectEl.style.height = `${state.captureRange.h}px`;
  } else {
    rangeRectEl.hidden = true;
  }
}

function _refreshRangeButtons() {
  rangeSelectBtn.textContent = state.rangeMode ? "決定" : "画像範囲";
  rangeSelectBtn.classList.toggle("active", state.rangeMode);
  // Reset is shown whenever a rect exists OR while in mode (so the user
  // can scrap a half-drawn rect and re-draw without leaving the mode).
  rangeResetBtn.hidden = !(state.captureRange || state.rangeMode);
}

function setRangeMode(on) {
  state.rangeMode = !!on;
  document.body.classList.toggle("range-select-mode", state.rangeMode);
  if (state.rangeMode && state.view !== "3d") setView("3d");
  _refreshRangeButtons();
}

rangeSelectBtn.addEventListener("click", () => setRangeMode(!state.rangeMode));
rangeResetBtn.addEventListener("click", () => {
  state.captureRange = null;
  _applyRangeRectVisual();
  _refreshRangeButtons();
});

let _rangeDrag = null;
rangeOverlayEl.addEventListener("pointerdown", (ev) => {
  if (!state.rangeMode || ev.button !== 0) return;
  const r = rangeOverlayEl.getBoundingClientRect();
  _rangeDrag = {
    pointerId: ev.pointerId,
    startX: ev.clientX - r.left,
    startY: ev.clientY - r.top,
  };
  rangeRectEl.hidden = false;
  rangeRectEl.style.left = `${_rangeDrag.startX}px`;
  rangeRectEl.style.top  = `${_rangeDrag.startY}px`;
  rangeRectEl.style.width = "0px";
  rangeRectEl.style.height = "0px";
  try { rangeOverlayEl.setPointerCapture(ev.pointerId); } catch (_e) {}
});
rangeOverlayEl.addEventListener("pointermove", (ev) => {
  if (!_rangeDrag) return;
  const r = rangeOverlayEl.getBoundingClientRect();
  const x = ev.clientX - r.left;
  const y = ev.clientY - r.top;
  const left = Math.min(_rangeDrag.startX, x);
  const top  = Math.min(_rangeDrag.startY, y);
  const w    = Math.abs(x - _rangeDrag.startX);
  const h    = Math.abs(y - _rangeDrag.startY);
  rangeRectEl.style.left = `${left}px`;
  rangeRectEl.style.top  = `${top}px`;
  rangeRectEl.style.width = `${w}px`;
  rangeRectEl.style.height = `${h}px`;
});
const _endRangeDrag = (ev) => {
  if (!_rangeDrag) return;
  try { rangeOverlayEl.releasePointerCapture(_rangeDrag.pointerId); } catch (_e) {}
  const r = rangeOverlayEl.getBoundingClientRect();
  const x = ev.clientX - r.left;
  const y = ev.clientY - r.top;
  const left = Math.min(_rangeDrag.startX, x);
  const top  = Math.min(_rangeDrag.startY, y);
  const w    = Math.abs(x - _rangeDrag.startX);
  const h    = Math.abs(y - _rangeDrag.startY);
  _rangeDrag = null;
  if (w < 8 || h < 8) {
    rangeRectEl.hidden = !state.captureRange;
    return;
  }
  state.captureRange = { x: left, y: top, w, h };
  _applyRangeRectVisual();
  _refreshRangeButtons();
};
rangeOverlayEl.addEventListener("pointerup",     _endRangeDrag);
rangeOverlayEl.addEventListener("pointercancel", _endRangeDrag);

// ---------------------------------------------------------------------------
// Background color
// ---------------------------------------------------------------------------

function setBgColor(hex) {
  // ``state.bgColor`` only feeds the capture path (composite pose_image).
  // The live 3D viewer keeps its default neutral grey so the user can
  // edit comfortably regardless of the chosen output background.
  state.bgColor = hex;
  if (bgColorSwatch) bgColorSwatch.style.background = hex;
  if (bgColorInput && bgColorInput.value !== hex) bgColorInput.value = hex;
}
bgColorBtn.addEventListener("click", (ev) => {
  if (ev.target === bgColorInput) return;
  bgColorInput.click();
});
bgColorInput.addEventListener("input",  () => setBgColor(bgColorInput.value));
bgColorInput.addEventListener("change", () => setBgColor(bgColorInput.value));

// Allow Esc to cancel pending bbox draw.
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && state.drawMode) {
    const pending = state.persons.find((p) => p.pendingBboxDraw);
    if (pending) state.persons = state.persons.filter((p) => p !== pending);
    setDrawMode(false);
    rebuildPersonList();
    rebuildBboxOverlay();
    refreshButtons();
  }
});

// ---------------------------------------------------------------------------
// Run inference
// ---------------------------------------------------------------------------

runBtn.addEventListener("click", async () => {
  if (!state.inputImage || state.persons.length === 0) return;
  state.inferenceRunning = true;
  runInfoEl.textContent = "推論中…";
  refreshButtons();
  setStatus("推論実行中…");
  try {
    const fd = new FormData();
    fd.append("image", state.inputBlob, "input.png");
    const personsPayload = state.persons.map((p) => ({
      id: p.id,
      bbox_xyxy: p.bbox,
      height_m: p.heightMeters,
      additional_bboxes: p.additionalBboxes || [],
      negative_bboxes: p.negativeBboxes || [],
    }));
    fd.append("payload", JSON.stringify({
      persons: personsPayload, inference_type: "full",
    }));
    for (const p of state.persons) {
      if (p.lhandImage?.blob) fd.append(`lhand_${p.id}`, p.lhandImage.blob, `lhand_${p.id}.png`);
      if (p.rhandImage?.blob) fd.append(`rhand_${p.id}`, p.rhandImage.blob, `rhand_${p.id}.png`);
    }
    const res = await fetch("/sam3d/api/plus/process", { method: "POST", body: fd });
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || `HTTP ${res.status}`);
    state.multiJobId = j.plus_job_id;
    overlayEl.hidden = true;
    // Fresh inference produces fresh hip_world placements; throw away
    // the previous cohort's layout cache so the new render starts from
    // a freshly-computed cohort centring + ground snap.
    viewer.resetCohortLayoutCache();
    viewer.resetFramedFingerprint();
    // Persist per-person settings + skeleton from the inference response so
    // the edit panel can render lean / bone-rotation controls without an
    // extra round-trip.
    for (const pp of j.per_person) {
      const slot = state.persons.find((q) => q.id === pp.id);
      if (!slot) continue;
      slot.settings = pp.settings || _emptySettings();
      slot.skeleton = pp.humanoid_skeleton || null;
      // ``transform`` is now frontend-owned state — fresh inference resets
      // it to identity so a previous run's offsets don't carry over to a
      // re-cropped session. The hip_world used for the gizmo's pivot
      // comes from the response's per_person[i].hip_world.
      if (!slot.transform) {
        slot.transform = { translate: [0, 0, 0], rotate_deg: [0, 0, 0] };
      } else {
        slot.transform = { translate: [0, 0, 0], rotate_deg: [0, 0, 0] };
      }
      slot.expanded = false;
      slot.poseEditOpen = false;
      slot.transformOpen = false;
    }
    await viewer.loadObjPerPerson(j.obj_url, j.per_person, state.persons, state.activeId);
    if (state.activeId) viewer.setActiveAnchor(state.activeId);
    runInfoEl.textContent = `OK (${j.per_person.length} 人, ${j.elapsed_sec}s)`;
    setView("3d");
    setStatus("推論完了");
    rebuildPersonList();
  } catch (e) {
    console.error("multi_process failed:", e);
    runInfoEl.textContent = "推論失敗: " + e.message;
    setStatus("推論失敗");
  } finally {
    state.inferenceRunning = false;
    refreshButtons();
  }
});

// ---------------------------------------------------------------------------
// Per-person re-render (height changes after inference)
// ---------------------------------------------------------------------------

let _reRenderTimer = null;
function scheduleReRender() {
  clearTimeout(_reRenderTimer);
  _reRenderTimer = setTimeout(triggerReRender, 220);
}

// "Trigger" semantics — used during continuous IK drag where a 220 ms
// debounce would defer every render until the user lets go. Mirrors the
// single Pose Editor's ``triggerRender``: fire immediately when no
// fetch is outstanding; raise a "dirty" flag when one is, and re-fire
// once the in-flight request settles. The result is roughly one render
// per backend round-trip (~5 fps on a typical machine), which keeps the
// mesh visibly tracking the IK handle in real time.
let _reRenderInFlight = false;
let _reRenderDirty    = false;
function triggerReRender() {
  clearTimeout(_reRenderTimer);
  _reRenderTimer = null;
  if (_reRenderInFlight) {
    _reRenderDirty = true;
    return;
  }
  reRender();
}

async function reRender() {
  if (!state.multiJobId) return;
  _reRenderInFlight = true;
  _reRenderDirty    = false;
  const per_person_settings = {};
  for (const p of state.persons) {
    // ``transform`` is purely client-side now (anchored Object3D wrapping
    // the per-person mesh) — backend doesn't see it on /multi_render.
    per_person_settings[p.id] = {
      height_m: p.heightMeters,
      body_params:  p.settings?.body_params  || {},
      bone_lengths: p.settings?.bone_lengths || {},
      blendshapes:  p.settings?.blendshapes  || {},
      pose_adjust:  p.settings?.pose_adjust  || { lean_correction: 0, rotation_overrides: {} },
    };
  }
  try {
    const res = await fetch("/sam3d/api/plus/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        plus_job_id: state.multiJobId, per_person_settings,
      }),
    });
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || `HTTP ${res.status}`);
    for (const pp of j.per_person) {
      const slot = state.persons.find((q) => q.id === pp.id);
      if (!slot) continue;
      slot.skeleton = pp.humanoid_skeleton || slot.skeleton;
    }
    // Re-build per-person anchors against the new mesh; existing user
    // transforms are re-applied so a height-change drag doesn't snap a
    // posed body back to identity.
    await viewer.loadObjPerPerson(j.obj_url, j.per_person, state.persons, state.activeId);
    if (state.activeId) {
      if (_ikMode) {
        // The pose-edit overlay was carried across the anchor rebuild
        // (see ``loadObjPerPerson``'s ``carriedPoseRoot`` branch) so the
        // user's IK manipulations persist. Just resync the gizmo target
        // to the selected bone if no drag is in progress.
        viewer.syncIkTargetToSelected();
      } else {
        viewer.setActiveAnchor(state.activeId);
      }
    }
  } catch (e) {
    console.warn("multi_render failed:", e);
    runInfoEl.textContent = "再描画失敗: " + e.message;
  } finally {
    _reRenderInFlight = false;
    if (_reRenderDirty) {
      // Pick up any edits that arrived while the previous fetch was in
      // flight — needed for live IK drag where the gizmo keeps emitting
      // change events faster than the backend can render.
      _reRenderDirty = false;
      reRender();
    } else if (_poseRebuildAfterRender) {
      // The drag ended and the queue has fully drained. The skeleton
      // we just received is the freshest one possible; rebuild the
      // pose-edit overlay against it so IK bones snap exactly onto
      // the rendered mesh's joint positions.
      _poseRebuildAfterRender = false;
      _rebuildPoseOverlayFromCurrent();
    }
  }
}

// ---------------------------------------------------------------------------
// Confirm-and-close
// ---------------------------------------------------------------------------

async function captureAllRenders() {
  // Composite — apply the user-picked background colour. The hex is
  // converted to a 24-bit integer for THREE.WebGLRenderer.setClearColor.
  const bgInt = parseInt((state.bgColor || "#2a2c30").slice(1), 16);
  const compositeBase = viewer.captureComposite(bgInt);
  // Overlay any objects on top of the composite (only — solo per-person
  // captures stay clean silhouettes).
  let composite = await _compositeObjects(compositeBase);
  if (state.captureRange) composite = await _cropCapture(composite, state.captureRange);

  // Per-person solo captures with transparent background.
  const perPerson = [];
  const N = viewer.getAnchorCount();
  for (let i = 0; i < N; i++) {
    viewer.setSoloVisibility(i);
    await new Promise(requestAnimationFrame);
    let cap = viewer.captureComposite(null);   // null → transparent
    if (state.captureRange) cap = await _cropCapture(cap, state.captureRange);
    perPerson.push(cap.dataUrl);
  }
  viewer.setSoloVisibility(null);
  return { composite, perPerson };
}

// Crop a captured PNG to ``rect`` (in viewport CSS pixels). The capture
// is at canvas backing-store resolution so we multiply by devicePixelRatio
// to land on the correct sub-rect.
async function _cropCapture(cap, rect) {
  const baseImg = await new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("crop base decode failed"));
    im.src = cap.dataUrl;
  });
  const dpr = window.devicePixelRatio || 1;
  const sx = Math.max(0, Math.round(rect.x * dpr));
  const sy = Math.max(0, Math.round(rect.y * dpr));
  const sw = Math.max(1, Math.min(cap.width  - sx, Math.round(rect.w * dpr)));
  const sh = Math.max(1, Math.min(cap.height - sy, Math.round(rect.h * dpr)));
  const off = document.createElement("canvas");
  off.width = sw; off.height = sh;
  const ctx = off.getContext("2d");
  ctx.drawImage(baseImg, sx, sy, sw, sh, 0, 0, sw, sh);
  return { dataUrl: off.toDataURL("image/png"), width: sw, height: sh };
}

// Draw the active object overlay on top of a Three.js capture. The capture
// gives us the 3D render at backing-store resolution (dpr-scaled), and
// each object's posX/posY are normalised viewport coords so we just
// multiply by the capture size to land at the right pixels.
async function _compositeObjects(cap) {
  if (!state.objects.length) return cap;
  const baseImg = await new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("composite base decode failed"));
    im.src = cap.dataUrl;
  });
  const off = document.createElement("canvas");
  off.width = cap.width; off.height = cap.height;
  const ctx = off.getContext("2d");
  ctx.drawImage(baseImg, 0, 0);

  const dpr = window.devicePixelRatio || 1;
  for (const o of state.objects) {
    if (!o.dataUrl) continue;
    let img;
    try {
      img = await new Promise((resolve, reject) => {
        const im = new Image();
        im.onload = () => resolve(im);
        im.onerror = () => reject(new Error(`object ${o.id} decode failed`));
        im.src = o.dataUrl;
      });
    } catch (e) {
      console.warn(e);
      continue;
    }
    const cx = o.posX * cap.width;
    const cy = o.posY * cap.height;
    // CSS pixels of the sprite; multiply by dpr for backing-store size.
    const w = o.naturalWidth * o.scale * dpr;
    const h = o.naturalHeight * o.scale * dpr;
    ctx.save();
    ctx.globalAlpha = Math.max(0, Math.min(1, o.opacity ?? 1));
    ctx.translate(cx, cy);
    ctx.rotate(o.rotationDeg * Math.PI / 180);
    ctx.drawImage(img, -w / 2, -h / 2, w, h);
    ctx.restore();
  }
  return { dataUrl: off.toDataURL("image/png"), width: cap.width, height: cap.height };
}

confirmBtn.addEventListener("click", async () => {
  if (!state.multiJobId) return;
  try {
    const { composite, perPerson } = await captureAllRenders();
    const handLs = state.persons.map((p) => p.lhandImage?.dataUrl || "");
    const handRs = state.persons.map((p) => p.rhandImage?.dataUrl || "");
    const target = window.parent && window.parent !== window ? window.parent : window.opener;
    if (!target) {
      alert("親ウィンドウが解決できませんでした (popup-blocker?)");
      return;
    }
    target.postMessage({
      type: CONFIRM_MSG,
      node_id: NODE_ID,
      pose_image:    composite.dataUrl,
      pose_images:   JSON.stringify(perPerson),
      input_image:   state.inputImage.dataUrl,
      hand_l_images: JSON.stringify(handLs),
      hand_r_images: JSON.stringify(handRs),
    }, window.location.origin);
  } catch (e) {
    console.error("confirm failed:", e);
    alert("確定に失敗しました: " + e.message);
  }
});

document.addEventListener("keydown", (ev) => {
  if ((ev.ctrlKey || ev.metaKey) && ev.key === "Enter") {
    ev.preventDefault();
    confirmBtn.click();
  }
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

(async function boot() {
  setStatus("loading units / presets…");
  try {
    const r = await fetch("/sam3d/api/units");
    if (r.ok) state.unit = await r.json();
  } catch (e) { console.warn("units fetch failed:", e); }
  try {
    await _loadPresetList();
  } catch (e) { console.warn("preset list load failed:", e); }

  setStatus("ready");

  // Always start with a clean slate — the editor no longer remembers
  // previous sessions. The setup below must always run, even if individual
  // steps fail, so the user is never left with a half-initialized UI.
  try { viewer.resize(); } catch (e) { console.warn("viewer.resize failed:", e); }
  try { setView("image"); } catch (e) { console.warn("setView failed:", e); }
  try { rebuildPersonList(); } catch (e) { console.warn("rebuildPersonList failed:", e); }
  try { rebuildBboxOverlay(); } catch (e) { console.warn("rebuildBboxOverlay failed:", e); }
  try { rebuildObjectList(); } catch (e) { console.warn("rebuildObjectList failed:", e); }
  try { rebuildObjectOverlay(); } catch (e) { console.warn("rebuildObjectOverlay failed:", e); }
  try { refreshButtons(); } catch (e) { console.warn("refreshButtons failed:", e); }
  try { refreshFloatingAddBtn(); } catch (e) { console.warn("refreshFloatingAddBtn failed:", e); }
  // Sync the swatch + native picker UI with the default bg colour.
  try {
    if (bgColorInput) bgColorInput.value = state.bgColor;
    if (bgColorSwatch) bgColorSwatch.style.background = state.bgColor;
  } catch (e) { console.warn("bg color init failed:", e); }
  try { _applyRangeRectVisual(); } catch (e) { console.warn("range visual init failed:", e); }
  try { _refreshRangeButtons(); } catch (e) { console.warn("range buttons init failed:", e); }
})();

// Re-layout overlay sprites on viewport resize (their positions are
// normalised, but the absolute pixel offsets need recomputing).
window.addEventListener("resize", () => {
  rebuildObjectOverlay();
});

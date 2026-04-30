// AUTO-GENERATED from E:/SAM3DBody_utills/web/static/app.js by
// nodes/editor/__init__.py setup; rewrites /api/* -> /sam3d/api/*
// and /tmp/* -> /sam3d/editor/tmp/*. Editor-only entry-point lives in
// editor_init.js, which loads this module and wires up confirm-and-close.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const healthEl = $("health");
const fileInput = $("file-input");
const fileDrop = $("file-drop");
const fileLabel = $("file-label");
const previewEl = $("preview");
const runBtn = $("run-btn");
const runInfo = $("run-info");
const overlay = $("viewport-overlay");
// Segmentation params now live in config.ini [segmentation]; no longer in UI.
const bodyPresetSection = $("body-preset-section");
const presetSelect = $("preset-select");
const loadPresetBtn = $("load-preset-btn");
const resetBtn = $("reset-btn");
const exportFbxBtn = $("export-fbx-btn");
const exportBvhBtn = $("export-bvh-btn");
const fbxInfo = $("fbx-info");
const bvhInfo = $("bvh-info");
const renderInfo = $("render-info");
const bodyPresetJsonInput = $("body-preset-json-input");
const bodyPresetJsonDrop = $("body-preset-json-drop");
const bodyPresetJsonLabel = $("body-preset-json-label");
const bodyPresetSourceInfo = $("body-preset-source-info");
const bodyParamsEl = $("body-params-sliders");
const boneLengthsEl = $("bone-lengths-sliders");
const blendshapesEl = $("blendshapes-sliders");

// ---------------------------------------------------------------------------
// Tab nav
// ---------------------------------------------------------------------------

const TABS = ["image", "video", "make", "admin"];
const MAKE_JOB_ID = "make";

const featureFlags = { preset_pack_admin: false, debug: false };

function activateTab(name) {
  if (!TABS.includes(name)) return;
  // Rotation editor locks the tab — silently refuse switches until the user
  // exits. Clicks on the disabled tab buttons are already blocked by the
  // pointer-events rule, but programmatic calls (localStorage restore etc.)
  // can still land here.
  if (poseEditMode) return;
  // Same defense while pose / motion inference is in flight.
  if (inferenceLocked) return;
  // Honor runtime feature flags: if Preset Admin is disabled in config.ini,
  // bounce away silently when someone tries to activate it.
  if (name === "admin" && !featureFlags.preset_pack_admin) {
    name = "image";
  }
  document.querySelectorAll("[data-tab-panel]").forEach((el) => {
    el.hidden = el.dataset.tabPanel !== name;
  });
  // The Body Preset panel (preset dropdown + JSON upload, plus sliders on
  // Make tab) is always available on image/video/make tabs — users can
  // stage a preset / upload a custom body preset before running inference.
  bodyPresetUnlocked = true;
  document.querySelectorAll("[data-tab-show]").forEach((el) => {
    const allowed = el.dataset.tabShow.split(/\s+/).filter(Boolean);
    el.hidden = !allowed.includes(name);
  });
  // `data-tab-hide="x y"` hides the element only on those tabs. Useful for
  // entries that live inside a shared section (e.g. Export FBX button in the
  // Body Preset panel) but should disappear on one tab.
  document.querySelectorAll("[data-tab-hide]").forEach((el) => {
    const hidden = el.dataset.tabHide.split(/\s+/).filter(Boolean);
    el.hidden = hidden.includes(name);
  });
  document.querySelectorAll(".tab-nav button[data-tab]").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  try { localStorage.setItem("body3d.tab", name); } catch (_e) {}

  // Swap the viewport's cached mesh/animation/camera state to match the tab.
  viewer.switchTab(name);

  // Rebind settings + Body Preset panel UI to the active tab's slot so each
  // tab keeps its own body_params / bone_lengths / blendshapes / preset
  // selection. Admin tab has no body-preset state — leave `settings` alone.
  if (tabSettings[name]) {
    settings = tabSettings[name];
    refreshSlidersFromSettings();
    refreshBodyPresetPanelFromState();
  }

  // Rebind currentJobId so /api/render targets the right session. Image /
  // video fall back to MAKE_JOB_ID until inference produces a tab-specific
  // session — this lets the viewport show the MHR neutral body (with the
  // tab's current body preset settings) before the user runs anything.
  if (name in tabJobIds) {
    currentJobId = tabJobIds[name] || MAKE_JOB_ID;
  }

  // First-visit render: if the viewer has no cached mesh for this tab,
  // kick off a render so the viewport isn't empty on arrival. Subsequent
  // visits rely on viewer.switchTab() having restored the cached state.
  if ((name === "image" || name === "video" || name === "make")
      && !viewer.hasCachedMesh(name)) {
    scheduleRender();
  }
}

function initTabs() {
  document.querySelectorAll(".tab-nav button[data-tab]").forEach((b) => {
    b.addEventListener("click", () => activateTab(b.dataset.tab));
  });
  let initial = "image";
  try { initial = localStorage.getItem("body3d.tab") || "image"; } catch (_e) {}
  activateTab(initial);
}

// Track whether a pose session exists so the Body Preset panel only appears
// after the user has run /api/process at least once (matches prior behaviour).
let bodyPresetUnlocked = false;
let renderAvailable = false;  // Save PNG is disabled until the first mesh is drawn.

const viewer = initViewer();
let selectedFile = null;
let currentJobId = null;
// Per-tab job id. Image tab gets its id from /api/process, make tab uses a
// fixed neutral-body session key. Video tab has no OBJ-render job (it
// builds FBX animations via currentMotionId instead), so its entry stays
// null — scheduleRender short-circuits when on the video tab without motion.
// Kept in sync with `currentJobId` on every tab switch.
const tabJobIds = { image: null, video: null, make: MAKE_JOB_ID };
let schema = null;
// Per-tab body-preset shape state. Image / video / make tabs each own an
// independent `settings` snapshot and Body Preset panel UI state (selected
// preset, source label, JSON upload label). `settings` is rebound on tab
// switch to point at the active tab's slot so that existing readers
// (render calls, slider handlers) work unchanged. The Body Preset panel
// (preset dropdown + JSON upload) itself is shared DOM across tabs — we
// just swap the values it displays when the tab changes.
// ``lean_correction`` starts at 0.5 (the calibrated "いい感じ" value) so a
// fresh tab already straightens a forward-leaning subject without the user
// having to touch the slider.
const LEAN_CORRECTION_DEFAULT = 0.0;

function _freshSettings() {
  return {
    body_params: {},
    bone_lengths: {},
    blendshapes: {},
    pose_adjust: { lean_correction: LEAN_CORRECTION_DEFAULT },
  };
}
function _freshBodyPresetState() {
  return { preset: "", sourceText: "", jsonLabel: null };
}
const tabSettings = {
  image: _freshSettings(),
  video: _freshSettings(),
  make:  _freshSettings(),
};
const tabBodyPresetState = {
  image: _freshBodyPresetState(),
  video: _freshBodyPresetState(),
  make:  _freshBodyPresetState(),
};
let settings = tabSettings.image;

function initViewer() {
  const canvas = $("three-canvas");
  // `preserveDrawingBuffer` is required for `canvas.toBlob()` / `toDataURL()`
  // to return the rendered image instead of a blank buffer (WebGL normally
  // clears the back buffer after each present).
  const renderer = new THREE.WebGLRenderer({
    canvas, antialias: true, preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x22242a, 1);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  camera.position.set(0, 1.0, 3.0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const dir = new THREE.DirectionalLight(0xffffff, 1.2);
  dir.position.set(3, 5, 2);
  scene.add(dir);
  const fill = new THREE.DirectionalLight(0xaaccff, 0.3);
  fill.position.set(-3, 2, -4);
  scene.add(fill);

  const grid = new THREE.GridHelper(4, 20, 0x555555, 0x333333);
  scene.add(grid);

  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 0.9, 0);
  controls.update();

  let meshGroup = null;
  // Inner OBJ root mesh inside meshGroup wrapper. We use a wrapper so the
  // body-anchor gizmo (and rotation pivot) lives at the body's visual
  // centre (matches the Pose Editor + behaviour) — the inner mesh is
  // shifted by ``-_meshLocalCenter`` so its bbox centre sits at the
  // wrapper's local origin, while the wrapper itself is parked at the
  // visual centre in scene-world coords.
  let _meshInner = null;
  const _meshLocalCenter = new THREE.Vector3();
  // Framing done once per job_id to avoid camera jump on every slider edit.
  let framedJob = null;
  // Per-job mesh fit (height-scale + center / ground offset) computed on
  // first render and cached. Subsequent renders for the same job reuse it
  // so IK / sliders that change the bbox don't translate the body around
  // the viewport — the user expects the body to stay anchored.
  let _meshFitJob = null;
  let _meshFitScale = 1.0;
  const _meshFitOffset = new THREE.Vector3();   // immutable fit anchor
  // Hips drag translation, accumulated by the user. Lives separately from
  // _meshFitOffset so "Reset all bones" can wipe it without losing the
  // first-load centering. Final mesh position = fit anchor + hips offset.
  const _hipsOffset = new THREE.Vector3();
  // Manual body-anchor transform — driven by the translate / rotate gizmo
  // bound to meshGroup whenever pose-edit (IK) mode is OFF. Applied on top
  // of fit + hips offsets so:
  //   meshGroup.position = _meshFitOffset + _hipsOffset + _bodyXform.translate
  //   meshGroup.rotation =                                _bodyXform.rotate
  // Survives mesh re-renders the same way fit / hips offsets do.
  const _bodyXform = {
    translate: new THREE.Vector3(),
    rotate:    new THREE.Euler(0, 0, 0, "XYZ"),
  };
  let _onBodyTransformChanged = null;
  // Animation bookkeeping (used when the viewer shows an animated FBX clip
  // produced by the video pipeline).
  let mixer = null;
  let animClip = null;
  const clock = new THREE.Clock();

  // Per-tab viewport cache so that switching between image / video / make
  // restores the mesh, animation and camera state the user last saw. Admin
  // tab has no viewport state and is skipped by switchTab().
  const tabStates = { image: null, video: null, make: null };
  let activeTabKey = "image";

  function _captureState() {
    return {
      meshGroup,
      mixer,
      animClip,
      framedJob,
      cameraPos: camera.position.clone(),
      target: controls.target.clone(),
    };
  }

  function _clearActive() {
    if (meshGroup && meshGroup.parent === scene) scene.remove(meshGroup);
    meshGroup = null;
    mixer = null;
    animClip = null;
    _dirty = true;
  }

  function _applyState(s) {
    if (!s) return;
    if (s.meshGroup) {
      if (s.meshGroup.parent !== scene) scene.add(s.meshGroup);
      meshGroup = s.meshGroup;
    }
    mixer = s.mixer || null;
    animClip = s.animClip || null;
    framedJob = s.framedJob || null;
    if (s.cameraPos) camera.position.copy(s.cameraPos);
    if (s.target) controls.target.copy(s.target);
    controls.update();
    _dirty = true;
  }

  function switchTab(tabKey) {
    if (!(tabKey in tabStates)) return;  // admin etc. — keep viewer untouched.
    if (tabKey === activeTabKey) return;
    tabStates[activeTabKey] = _captureState();
    _clearActive();
    activeTabKey = tabKey;
    _applyState(tabStates[tabKey]);
  }

  function hasCachedMesh(tabKey) {
    return !!(tabStates[tabKey] && tabStates[tabKey].meshGroup);
  }

  // Dirty-flag render loop. Re-rendering a static scene every frame burns
  // GPU / battery for no visible change, so `renderer.render` is gated on
  // a `_dirty` bit flipped by OrbitControls events, mesh swaps, camera
  // restores, and — every frame — an active AnimationMixer.
  let _dirty = true;
  const markDirty = () => { _dirty = true; };
  controls.addEventListener("change", markDirty);

  function resize() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    markDirty();
  }
  window.addEventListener("resize", resize);
  resize();

  (function tick() {
    const dt = clock.getDelta();
    if (mixer) {
      mixer.update(dt);
      _dirty = true;
    }
    controls.update();
    if (_dirty) {
      renderer.render(scene, camera);
      _dirty = false;
    }
    requestAnimationFrame(tick);
  })();

  function _disposeMesh() {
    if (mixer) {
      mixer.stopAllAction();
      mixer.uncacheRoot(meshGroup);
      mixer = null;
    }
    animClip = null;
    if (meshGroup) {
      scene.remove(meshGroup);
      meshGroup.traverse?.((o) => {
        if (o.isMesh) {
          o.geometry?.dispose?.();
          if (Array.isArray(o.material)) {
            o.material.forEach((m) => m?.dispose?.());
          } else {
            o.material?.dispose?.();
          }
        }
      });
    }
  }

  async function loadObj(url, jobId) {
    const loader = new OBJLoader();
    const root = await new Promise((resolve, reject) => {
      loader.load(url, resolve, undefined, reject);
    });
    root.traverse((obj) => {
      if (obj.isMesh) {
        obj.material = new THREE.MeshStandardMaterial({
          color: 0xbbcddd, roughness: 0.75, metalness: 0.05, flatShading: false,
        });
      }
    });

    // Preserve the pose-edit overlay across mesh reloads — _disposeMesh
    // would otherwise traverse the old meshGroup and dispose our handle /
    // line geometries. Detach now, re-attach to the new meshGroup below.
    const carriedPoseRoot = (poseEdit.active && poseEdit.root) ? poseEdit.root : null;
    if (carriedPoseRoot && carriedPoseRoot.parent) {
      carriedPoseRoot.parent.remove(carriedPoseRoot);
    }

    _disposeMesh();

    // Wrap the OBJ root in a body-anchor so the right-toolbar gizmo
    // (translate / rotate) pivots around the body's visual centre.
    // The wrapper holds the height-scale + world placement; the inner
    // mesh is shifted by ``-_meshLocalCenter`` so its bbox centre sits
    // at the wrapper's local origin (mirrors Pose Editor +).
    const wrapper = new THREE.Group();
    wrapper.name = "_body_anchor";
    wrapper.add(root);
    meshGroup = wrapper;
    _meshInner = root;

    // First render of this job → measure & cache the height-fit / center /
    // ground offset. Same job re-renders → reuse cached values so IK or
    // slider edits that change the bbox don't slide the body sideways
    // (re-centring on every frame is what makes the body "drift" during
    // an arm IK drag). Hips drag offset (if any) is added on top.
    const targetHeight = 1.7;
    if (jobId !== _meshFitJob) {
      // Measure the unscaled bbox (wrapper at scale 1, mesh at default).
      const box = new THREE.Box3().setFromObject(root);
      const size = new THREE.Vector3();
      box.getSize(size);
      const scale = targetHeight / Math.max(size.y, 1e-3);
      const center = new THREE.Vector3();
      box.getCenter(center);

      _meshFitJob = jobId;
      _meshFitScale = scale;
      _meshLocalCenter.copy(center);
      // After applying scale on the wrapper, the body height is
      // ``size.y * scale``. Half-height is the wrapper's world Y so
      // feet land at y=0.
      _meshFitOffset.set(0, (size.y * scale) / 2, 0);
      // New job → wipe any prior Hips translation so the new body
      // doesn't inherit the previous body's drag offset. Same goes for
      // the manual body-anchor transform — a fresh inference starts at
      // identity so the user sees the freshly-centered cohort, not a
      // translated one from the previous photo.
      _hipsOffset.set(0, 0, 0);
      _bodyXform.translate.set(0, 0, 0);
      _bodyXform.rotate.set(0, 0, 0);
    }
    wrapper.scale.setScalar(_meshFitScale);
    root.position.copy(_meshLocalCenter).multiplyScalar(-1);
    wrapper.position.copy(_meshFitOffset)
      .add(_hipsOffset)
      .add(_bodyXform.translate);
    wrapper.rotation.copy(_bodyXform.rotate);

    scene.add(wrapper);

    // Re-attach the pose-edit overlay so it inherits the new meshGroup's
    // height-fit + centring transforms. Bone handles also need their local
    // scale recomputed against the new meshGroup.scale so their visible
    // size in world units stays stable across re-renders. Finger handles
    // keep their 1/8 ratio so they don't swell when the mesh re-fits.
    if (carriedPoseRoot) {
      meshGroup.add(carriedPoseRoot);
      const meshLocalScale = Math.max(meshGroup.scale.x, 1e-6);
      const baseWorldScale = (poseEdit.baseHandleScale || 0.03) * (poseEdit.lastMeshScale || 1.0);
      const newHandleLocalScale = baseWorldScale / meshLocalScale;
      poseEdit.baseHandleScale = newHandleLocalScale;
      poseEdit.lastMeshScale = meshLocalScale;
      for (const bone of poseEdit.bones) {
        const perBoneScale = _isFingerBoneName(bone.name)
          ? newHandleLocalScale * _FINGER_HANDLE_SCALE_RATIO
          : newHandleLocalScale;
        bone.handle.scale.setScalar(perBoneScale);
      }
      // Bone handle world positions just changed (mesh re-fitted to the
      // new height). Sync the IK target to follow the selected bone so
      // the translate gizmo doesn't end up floating in space.
      //
      // BUT skip when an IK drag (or Hips translate) is in flight —
      // re-syncing mid-drag would snap the target back to the bone, then
      // the next mouse move would jump again. The result is visible
      // jitter, most obvious on finger bones / small drags.
      if (poseEdit.selected && !_ikDrag.active && !_hipsDrag.active) {
        meshGroup.updateMatrixWorld(true);
        poseEdit.selected.handle.getWorldPosition(_ikTarget.position);
      }
    }

    if (framedJob !== jobId) {
      controls.target.set(0, targetHeight * 0.5, 0);
      camera.position.set(0, targetHeight * 0.6, targetHeight * 2.2);
      controls.update();
      framedJob = jobId;
    }
    // Re-bind the body-anchor gizmo to the freshly-built meshGroup so
    // the right-toolbar translate / rotate buttons keep working across
    // re-renders. Skip while pose-edit is active — IK owns the gizmo.
    if (!poseEdit.active) {
      tControls.enabled = true;
      tControls.attach(meshGroup);
      _tHelper.visible = true;
    }
    _dirty = true;
  }

  async function loadFbxAnimated(url, jobId) {
    const loader = new FBXLoader();
    const root = await new Promise((resolve, reject) => {
      loader.load(url, resolve, undefined, reject);
    });

    // FBX default material is often too dark; reassign a matte look so the
    // animated body reads well against the neutral viewport background.
    root.traverse((obj) => {
      if (obj.isMesh || obj.isSkinnedMesh) {
        obj.material = new THREE.MeshStandardMaterial({
          color: 0xbbcddd, roughness: 0.75, metalness: 0.05, flatShading: false,
        });
        obj.castShadow = false;
        obj.receiveShadow = false;
      }
    });

    _disposeMesh();
    meshGroup = root;

    // Fit roughly the same size + centering we use for OBJ so switching
    // between still and animated outputs doesn't jump the framing. FBX from
    // Blender sometimes arrives in cm (unit scale); a one-shot fit handles
    // both cases uniformly.
    const box = new THREE.Box3().setFromObject(root);
    const size = new THREE.Vector3();
    box.getSize(size);
    const targetHeight = 1.7;
    const scale = targetHeight / Math.max(size.y, 1e-3);
    root.scale.setScalar(scale);
    const centered = new THREE.Box3().setFromObject(root);
    const center = new THREE.Vector3();
    centered.getCenter(center);
    root.position.x -= center.x;
    root.position.z -= center.z;
    root.position.y -= centered.min.y;

    scene.add(root);

    // Start the first embedded animation clip, looping so the user can keep
    // watching the motion without re-triggering anything.
    if (root.animations && root.animations.length > 0) {
      mixer = new THREE.AnimationMixer(root);
      animClip = root.animations[0];
      const action = mixer.clipAction(animClip);
      action.setLoop(THREE.LoopRepeat, Infinity);
      action.play();
    } else {
      console.warn("FBX has no animation clips:", url);
    }

    if (framedJob !== jobId) {
      controls.target.set(0, targetHeight * 0.5, 0);
      camera.position.set(0, targetHeight * 0.6, targetHeight * 2.2);
      controls.update();
      framedJob = jobId;
    }
    _dirty = true;

    return {
      hasAnimation: !!animClip,
      clipDuration: animClip ? animClip.duration : 0,
    };
  }

  // Remove whatever mesh / animation is currently shown and leave the
  // viewport empty. The video tab uses this while a Blender rebuild is in
  // flight so the stale body doesn't keep playing on top of a
  // "rebuilding" status label.
  function clearMesh() {
    _disposeMesh();
    meshGroup = null;
    _dirty = true;
  }

  // -------------------------------------------------------------------
  // Pose-edit (rotation) overlay + TransformControls.
  //
  // The image tab's rotation editor maintains an overlay skeleton on top
  // of the posed mesh: one pickable sphere per humanoid bone, a line
  // from each bone to its humanoid parent, and a rotation gizmo
  // (TransformControls) attached to whichever sphere is selected. All
  // state lives inside the viewer so it can reuse the Three.js scene
  // graph / raycaster / render loop without leaking to the rest of app.js.
  // -------------------------------------------------------------------
  const poseEdit = {
    active: false,
    root: null,          // THREE.Group parented to scene while active
    bones: [],           // [{name, jointId, parentName, handle, base:{pos,quatLocal,quatParentWorld}, parentBone, children}]
    byName: new Map(),
    selected: null,
    baseHandleScale: 1.0,
    onBoneChange: null,  // callback: (boneName, localEuler XYZ radians | null) => void
    onBonePick:   null,  // callback: (boneName) => void
    onDragStart:  null,
    onDragEnd:    null,
  };

  // Finger handles are shrunk to 1/8 of the body-bone size — five fingers ×
  // four joints per hand cluster around the palm, so a full-size sphere per
  // joint makes them un-pickable. Test by name to avoid threading a flag
  // through every bone record.
  const _FINGER_HANDLE_SCALE_RATIO = 1 / 8;
  const _FINGER_BONE_RE = /^(Left|Right)Hand(Thumb|Index|Middle|Ring|Pinky)\d$/;
  function _isFingerBoneName(name) {
    return _FINGER_BONE_RE.test(name || "");
  }

  // Lines and handles are Euler-auto-scaled to the posed mesh's height so
  // they stay legible on petite / tall bodies.
  const _poseLineMat  = new THREE.LineBasicMaterial({ color: 0x33aaff, transparent: true, opacity: 0.85, depthTest: false });
  const _poseHandleMat = new THREE.MeshBasicMaterial({ color: 0x33aaff, transparent: true, opacity: 0.9, depthTest: false });
  const _poseHandleHoverMat = new THREE.MeshBasicMaterial({ color: 0xffc84d, transparent: true, opacity: 0.95, depthTest: false });
  const _poseHandleSelectedMat = new THREE.MeshBasicMaterial({ color: 0xff5252, transparent: true, opacity: 0.95, depthTest: false });
  const _poseHandleGeom = new THREE.SphereGeometry(1, 12, 10);

  // TransformControls in TRANSLATE mode drives the IK: the gizmo is
  // attached to an invisible "IK target" point that the user drags; on
  // each move we read the target's world position, run CCD on the
  // selected bone's parent chain, and the bone follows. The bone may
  // not actually reach the target when constraints kick in — the gap
  // between the gizmo (target) and the bone handle is the IK error.
  const tControls = new TransformControls(camera, canvas);
  tControls.setMode("translate");
  tControls.setSpace("world");
  const _tHelper = tControls.getHelper ? tControls.getHelper() : tControls;
  _tHelper.visible = false;
  tControls.enabled = false;
  scene.add(_tHelper);

  // Invisible point the gizmo grabs. Lives in scene (world space) so its
  // position field is directly usable as the IK goal after a mesh-local
  // conversion. Re-synced to the selected bone whenever selection or
  // drag-end happens.
  const _ikTarget = new THREE.Object3D();
  _ikTarget.name = "_ik_target";
  scene.add(_ikTarget);

  // Hips drag is special: it doesn't run CCD. Instead it translates the
  // entire meshGroup (and its bone overlay child) by the gizmo delta.
  // The user expects "moving Hips" to move the whole body in space.
  const _HIPS_BONE_NAME = "Hips";
  const _hipsDrag = {
    active: false,
    lastTargetWorld: new THREE.Vector3(),
  };

  tControls.addEventListener("dragging-changed", (ev) => {
    controls.enabled = !ev.value;
    if (ev.value) {
      if (poseEdit.selected) {
        if (poseEdit.selected.name === _HIPS_BONE_NAME) {
          _hipsDrag.active = true;
          _hipsDrag.lastTargetWorld.copy(_ikTarget.position);
          if (poseEdit.onDragStart) poseEdit.onDragStart();
        } else {
          _ikDrag.bone = poseEdit.selected;
          _ikDrag.chain = _buildIkChain(poseEdit.selected, _IK_MAX_CHAIN_LEN);
          // Length >= 2 → run CCD on the chain. Length === 1 (e.g. Spine
          // when the Hips block kicks in) → run self-rotate IK using the
          // dragged bone's first child as the tip.
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
      _ikDrag.bone = null;
      _ikDrag.chain = [];
      _hipsDrag.active = false;
      if (wasActive && poseEdit.onDragEnd) poseEdit.onDragEnd();
    }
  });

  tControls.addEventListener("change", () => {
    if (!poseEdit.active) {
      // Body-anchor binding: gizmo is attached to meshGroup. Decompose
      // its current position/rotation back into _bodyXform (fit + hips
      // offsets are immutable, anything beyond them is the user's
      // manual transform). Fires the change callback so the outer code
      // can mirror the values into UI / state if it wants.
      if (meshGroup && tControls.object === meshGroup) {
        _bodyXform.translate.set(
          meshGroup.position.x - _meshFitOffset.x - _hipsOffset.x,
          meshGroup.position.y - _meshFitOffset.y - _hipsOffset.y,
          meshGroup.position.z - _meshFitOffset.z - _hipsOffset.z,
        );
        _bodyXform.rotate.copy(meshGroup.rotation);
        if (_onBodyTransformChanged) _onBodyTransformChanged();
      }
      markDirty();
      return;
    }
    // Hips: translate the whole body. Persist the offset into the cached
    // mesh fit so subsequent re-renders (from other settings changes) keep
    // the translation instead of snapping back to the original origin.
    if (_hipsDrag.active) {
      if (meshGroup) {
        const dx = _ikTarget.position.x - _hipsDrag.lastTargetWorld.x;
        const dy = _ikTarget.position.y - _hipsDrag.lastTargetWorld.y;
        const dz = _ikTarget.position.z - _hipsDrag.lastTargetWorld.z;
        meshGroup.position.x += dx;
        meshGroup.position.y += dy;
        meshGroup.position.z += dz;
        // Persist into the dedicated Hips offset (separate from the fit
        // anchor) so re-renders preserve it AND "Reset all bones" can
        // wipe it independently.
        _hipsOffset.x += dx;
        _hipsOffset.y += dy;
        _hipsOffset.z += dz;
      }
      _hipsDrag.lastTargetWorld.copy(_ikTarget.position);
      markDirty();
      return;
    }
    if (!_ikDrag.active || !_ikDrag.bone) { markDirty(); return; }
    // _ikTarget is parented to scene, so its position == world position.
    // Convert to mesh-local before feeding CCD — the bone handles live in
    // mesh-local coords (root → meshGroup with translation + uniform scale).
    _ikTargetLocal.copy(_ikTarget.position);
    if (meshGroup) {
      _ikInvMeshMat.copy(meshGroup.matrixWorld).invert();
      _ikTargetLocal.applyMatrix4(_ikInvMeshMat);
    }
    if (_ikDrag.chain.length >= 2) {
      _runCcdIk(_ikDrag.chain, _ikTargetLocal);
    } else {
      _runSelfRotateIk(_ikDrag.chain[0], _ikTargetLocal);
    }
    // Full-body cascade from every root bone (Hips). CCD's per-iter
    // cascade only refreshes the chain's own subtree; with multiple prior
    // edits (e.g. Spine drag then Hand drag) the descendants outside the
    // current chain can drift out of sync with their stored localDelta
    // values. Re-cascading from the root guarantees every handle reflects
    // the composition (parent.q * base.localQuat * localDelta) consistently.
    for (const rb of poseEdit.bones) {
      if (!rb.parentBone) _propagateBoneTransform(rb);
    }
    _rebuildBoneLines();
    _emitIkChainChange(_ikDrag.chain);
    markDirty();
  });

  // ---- raycaster-driven bone picking on click ----
  // Click a sphere to select. Translate gizmo (TransformControls) appears
  // on the selected bone; dragging it runs CCD IK on the parent chain.
  const _ray = new THREE.Raycaster();
  const _rayMouse = new THREE.Vector2();
  canvas.addEventListener("pointerdown", (ev) => {
    if (!poseEdit.active || ev.button !== 0) return;
    // Don't intercept clicks on the gizmo itself.
    if (tControls.dragging) return;
    const r = canvas.getBoundingClientRect();
    _rayMouse.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
    _rayMouse.y = -((ev.clientY - r.top) / r.height) * 2 + 1;
    _ray.setFromCamera(_rayMouse, camera);
    const handles = poseEdit.bones.map((b) => b.handle);
    const hits = _ray.intersectObjects(handles, false);
    if (hits.length === 0) return;
    const hit = hits[0].object;
    const bone = poseEdit.bones.find((b) => b.handle === hit);
    if (!bone) return;
    ev.preventDefault();
    selectPoseBone(bone.name);
    if (poseEdit.onBonePick) poseEdit.onBonePick(bone.name);
  });

  function _quatMul(q1, q2) {
    // three.js Quaternion.multiply: this = this * q  (left-to-right).
    return q1.clone().multiply(q2);
  }

  function _propagateBoneTransform(bone) {
    // Recompute world transforms of descendants after ``bone`` moved. Each
    // descendant's world q is (parent world q) * (base localQuat) * (own
    // localDelta) — the localDelta term preserves that bone's own stored
    // rotation when an ancestor was the one that moved.
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
    // The bone's world quaternion induced purely by ancestor-override
    // propagation — i.e. where the bone's frame would be if its own
    // override were identity. Backend semantics: apply overrides in
    // topological order, each override is a local delta multiplied into
    // the bone's CURRENT world rotation (post-ancestor-overrides).
    if (!bone.parentBone) {
      return bone.base.worldQuat.clone();
    }
    return bone.parentBone.handle.quaternion.clone().multiply(bone.base.localQuat);
  }

  function _emitLocalEulerForSelected() {
    if (!poseEdit.selected || !poseEdit.onBoneChange) return;
    const bone = poseEdit.selected;
    // TransformControls drags modified bone.handle.quaternion directly.
    // Extract the local delta and stash it on the bone so subsequent
    // ancestor-propagation doesn't lose this edit.
    const parentPropQ = _parentPropagatedWorldQuat(bone);
    const localDeltaQ = parentPropQ.clone().invert().multiply(bone.handle.quaternion);
    bone.localDelta.copy(localDeltaQ);
    const e = new THREE.Euler().setFromQuaternion(localDeltaQ, "XYZ");
    const eps = 1e-6;
    if (Math.abs(e.x) < eps && Math.abs(e.y) < eps && Math.abs(e.z) < eps) {
      poseEdit.onBoneChange(bone.name, null);
    } else {
      poseEdit.onBoneChange(bone.name, [e.x, e.y, e.z]);
    }
  }

  // -------------------------------------------------------------------
  // IK drag — drag the selected (red) handle to translate the end
  // effector; CCD rotates a short chain of parent bones to follow. Each
  // affected bone's rotation_overrides entry is written via the
  // onIkChainChange callback; triggerRender() (called by the outer
  // controller in that callback) gives a live mesh preview, just like
  // the rotate gizmo path.
  // -------------------------------------------------------------------
  const _IK_MAX_CHAIN_LEN = 3;     // end effector + 2 parents (classic 2-bone IK)
  const _IK_ITERATIONS = 12;
  const _IK_ANGLE_TOLERANCE = 0.001;
  // ``DAMPING`` blends the target rotation per step (1.0 = full snap, 0.5
  // = half each iter). Lower values converge slower but more smoothly and
  // avoid IK flips on fast drags. We deliberately don't impose per-step
  // caps or magnitude clamps — empirically the user's natural drag motion
  // + damping is enough, and clamps caused subtle frame-of-reference bugs
  // when ancestor IK had already rotated the bone's local frame.
  const _IK_DAMPING = 0.5;
  const _ikDrag = {
    active: false,
    bone: null,
    chain: [],
  };

  // Bones that act as IK ceilings: the chain may include them but never
  // their parents. Stops shoulder / spine / hips from rotating when the
  // user IK-drags a hand or foot — the upper arm and thigh are treated
  // as anchored to the torso for IK on descendants.
  //
  // Exception: when the user drags the ceiling bone ITSELF (it lands at
  // chain[0] = end effector), the ceiling is released and the chain is
  // free to extend up. So dragging the upper arm rotates shoulder + spine,
  // and dragging the thigh rotates the hips — necessary because those
  // bones can't move at all without their parents pivoting.
  const _IK_CEILING_BONES = new Set([
    "LeftArm", "RightArm",
    "LeftUpLeg", "RightUpLeg",
  ]);

  // Walk up from `bone` to determine which side of the body it belongs to.
  // Returns true if `bone` (or any ancestor) is "Spine" before reaching
  // "Hips". Used so spine / arm IK can be blocked from propagating into
  // Hips (which would drag the legs along), while leg IK is still allowed
  // to rotate Hips when the user explicitly drags a thigh.
  function _isSpineSideBone(bone) {
    let cur = bone;
    while (cur) {
      if (cur.name === "Spine") return true;
      if (cur.name === "Hips") return false;
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
      // Ceiling: stop AFTER adding (current bone added, no further extension).
      // Skipped on chain.length === 1 so dragging the ceiling bone itself
      // releases the ceiling.
      if (chain.length > 1 && _IK_CEILING_BONES.has(cur.name)) break;
      // Hips block (spine-side only): stop BEFORE adding Hips so torso /
      // arm IK never spins the lower body. Leg IK is unaffected — leg-side
      // chains may still propagate to Hips and rotate it.
      if (spineSide && cur.parentBone && cur.parentBone.name === "Hips") break;
      cur = cur.parentBone;
    }
    return chain;
  }

  // Hinge bones (knees, elbows) should never hyperextend. We check the
  // WORLD-frame x-component of the candidate rotation, which encodes the
  // sign of the bend around the lateral X axis (assumes body roughly
  // faces +Z at rest):
  //   - knee natural bend: calf folds BEHIND body → +X rotation. Hyper­
  //     extension is -X. So map["LeftLeg"] = +1 means "+X is natural".
  //   - elbow natural bend: forearm folds FORWARD (hand toward face) →
  //     -X rotation. Hyperextension is +X. So map["LeftForeArm"] = -1.
  // When CCD's candidate rotation has the WRONG sign on world.x, we drop
  // localDelta back to identity so the joint stays straight rather than
  // flipping the wrong way.
  const _IK_HINGE_NATURAL_X = {
    "LeftLeg":      +1,
    "RightLeg":     +1,
    "LeftForeArm":  -1,
    "RightForeArm": -1,
  };

  const _ikHingeTmpInv = new THREE.Quaternion();
  const _ikHingeTmpWorld = new THREE.Quaternion();
  function _enforceHingeNoHyperextend(localDelta, parentPropQ, naturalSign) {
    // worldDelta = parentPropQ × localDelta × parentPropQ^-1 — the rotation
    // that this bone applies in WORLD frame at its pivot.
    _ikHingeTmpInv.copy(parentPropQ).invert();
    _ikHingeTmpWorld.copy(parentPropQ).multiply(localDelta).multiply(_ikHingeTmpInv);
    if (_ikHingeTmpWorld.x * naturalSign < 0) {
      localDelta.set(0, 0, 0, 1);
    }
  }

  // Run CCD in mesh-local space (== world-aligned, since meshGroup applies
  // only translation + uniform scale). All handle.position / quaternion
  // values live in this space directly.
  const _ikTmpV1 = new THREE.Vector3();
  const _ikTmpV2 = new THREE.Vector3();
  const _ikTmpAxis = new THREE.Vector3();
  const _ikTmpDq = new THREE.Quaternion();
  const _ikTmpNewQ = new THREE.Quaternion();
  function _runCcdIk(chain, targetMeshLocal) {
    if (chain.length < 2) return;
    const endBone = chain[0];
    for (let iter = 0; iter < _IK_ITERATIONS; iter++) {
      let maxAngle = 0;
      for (let i = 1; i < chain.length; i++) {
        const joint = chain[i];
        _ikTmpV1.copy(endBone.handle.position).sub(joint.handle.position);
        _ikTmpV2.copy(targetMeshLocal).sub(joint.handle.position);
        if (_ikTmpV1.lengthSq() < 1e-10 || _ikTmpV2.lengthSq() < 1e-10) continue;
        _ikTmpV1.normalize();
        _ikTmpV2.normalize();
        const dot = Math.max(-1, Math.min(1, _ikTmpV1.dot(_ikTmpV2)));
        let angle = Math.acos(dot);
        if (angle < _IK_ANGLE_TOLERANCE) continue;
        angle = angle * _IK_DAMPING;
        _ikTmpAxis.crossVectors(_ikTmpV1, _ikTmpV2);
        if (_ikTmpAxis.lengthSq() < 1e-12) continue;
        _ikTmpAxis.normalize();
        _ikTmpDq.setFromAxisAngle(_ikTmpAxis, angle);
        _ikTmpNewQ.copy(_ikTmpDq).multiply(joint.handle.quaternion);
        // Re-derive joint.localDelta from the candidate world quaternion.
        // Backend semantics: each override is a local-frame Euler applied
        // in the bone's posed frame (after ancestor overrides), so we go
        // through the parent-propagated quaternion as the rest reference.
        const parentPropQ = _parentPropagatedWorldQuat(joint);
        const parentPropQInv = parentPropQ.clone().invert();
        joint.localDelta.copy(parentPropQInv).multiply(_ikTmpNewQ);
        // Hinge constraint: prevent knee/elbow hyperextension by snapping
        // any wrong-direction rotation back to identity.
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

  // Single-bone "self-rotate" IK — used when the chain collapses to just
  // the dragged bone (e.g. dragging Spine, where the Hips block keeps the
  // chain from extending up). We rotate the bone itself so its primary
  // child swings toward the target. The bone's own joint position is
  // anchored at its parent (Hips for Spine), so this lets the user bend
  // the torso without dragging the lower body along.
  function _runSelfRotateIk(bone, targetMeshLocal) {
    if (!bone.children || bone.children.length === 0) return;
    const tip = bone.children[0];
    for (let iter = 0; iter < _IK_ITERATIONS; iter++) {
      _ikTmpV1.copy(tip.handle.position).sub(bone.handle.position);
      _ikTmpV2.copy(targetMeshLocal).sub(bone.handle.position);
      if (_ikTmpV1.lengthSq() < 1e-10 || _ikTmpV2.lengthSq() < 1e-10) break;
      _ikTmpV1.normalize();
      _ikTmpV2.normalize();
      const dot = Math.max(-1, Math.min(1, _ikTmpV1.dot(_ikTmpV2)));
      let angle = Math.acos(dot);
      if (angle < _IK_ANGLE_TOLERANCE) break;
      angle = angle * _IK_DAMPING;
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
    // For multi-bone chains, the end effector (chain[0]) isn't rotated by
    // CCD — only chain[1..N-1] are. For length-1 chains the dragged bone
    // IS the rotated joint (self-rotate path), so include it.
    const startIdx = chain.length === 1 ? 0 : 1;
    for (let i = startIdx; i < chain.length; i++) {
      const joint = chain[i];
      const e = new THREE.Euler().setFromQuaternion(joint.localDelta, "XYZ");
      const isIdent = Math.abs(e.x) < eps && Math.abs(e.y) < eps && Math.abs(e.z) < eps;
      changes.push({
        boneName: joint.name,
        eulerRad: isIdent ? null : [e.x, e.y, e.z],
      });
    }
    poseEdit.onIkChainChange(changes);
  }

  // Reusable temporaries for the TC-change handler's CCD math.
  const _ikTargetLocal = new THREE.Vector3();
  const _ikInvMeshMat = new THREE.Matrix4();

  // Public API ------------------------------------------------------------

  function enterPoseEdit(skeleton, storedOverrides) {
    if (poseEdit.active) return;
    if (!skeleton || !Array.isArray(skeleton.bones)) return;

    // Release any body-anchor binding before claiming the gizmo for IK.
    // Without this the body's translate/rotate gizmo can swallow the
    // first drag in pose-edit mode and shift the whole body.
    if (tControls.object === meshGroup) {
      tControls.detach();
      _tHelper.visible = false;
    }

    // Pick a handle radius roughly proportional to the rendered mesh height
    // so the spheres are legible but don't swamp the body. Computed
    // against the world-space bbox so the visible size is right regardless
    // of meshGroup's local scale; we then divide by meshGroup.scale before
    // assigning so handle.scale (a LOCAL value once parented to meshGroup)
    // produces the intended world size.
    let scale = 0.03;
    if (meshGroup) {
      const box = new THREE.Box3().setFromObject(meshGroup);
      const size = new THREE.Vector3();
      box.getSize(size);
      scale = Math.max(0.015, Math.min(0.06, size.y * 0.022));
    }
    const meshLocalScale = meshGroup ? Math.max(meshGroup.scale.x, 1e-6) : 1.0;
    const handleLocalScale = scale / meshLocalScale;
    poseEdit.baseHandleScale = handleLocalScale;
    poseEdit.lastMeshScale = meshLocalScale;

    const root = new THREE.Group();
    root.name = "_pose_edit_root";
    root.renderOrder = 999;
    // Parent the overlay to meshGroup so it inherits the height-fit + XZ
    // centring + ground-snap transforms applied to the mesh in loadObj —
    // bones come from the server in raw MHR units (~1m total height), the
    // mesh is rescaled to a fixed 1.7 world units, so without this the
    // skeleton sits at the wrong scale + offset relative to the body.
    (meshGroup || scene).add(root);
    const linesGroup = new THREE.Group();
    linesGroup.name = "_bone_lines";
    root.add(linesGroup);

    // Build bones, first pass: create handles at world positions.
    // Bones come from the server at raw MHR coords (matching the unshifted
    // OBJ vertices). The inner mesh is now shifted by ``-_meshLocalCenter``
    // inside the wrapper, so bone handles must subtract the same offset
    // to stay aligned with the rendered body vertices.
    poseEdit.bones = [];
    poseEdit.byName = new Map();
    for (const b of skeleton.bones) {
      const handle = new THREE.Mesh(_poseHandleGeom, _poseHandleMat);
      const perBoneScale = _isFingerBoneName(b.name)
        ? handleLocalScale * _FINGER_HANDLE_SCALE_RATIO
        : handleLocalScale;
      handle.scale.setScalar(perBoneScale);
      handle.renderOrder = 1000;
      handle.position.set(
        b.world_position[0] - _meshLocalCenter.x,
        b.world_position[1] - _meshLocalCenter.y,
        b.world_position[2] - _meshLocalCenter.z,
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
          worldPos: handle.position.clone(),
          worldQuat: handle.quaternion.clone(),
          localPos: new THREE.Vector3(),  // filled below
          localQuat: new THREE.Quaternion(),
        },
        // Bone's own local-frame delta (identity = no override). Kept in
        // sync with settings.pose_adjust.rotation_overrides[joint_id] by
        // the outer controller, and consumed during propagation so
        // ancestor rotations don't wipe out this bone's own edits.
        localDelta: new THREE.Quaternion(),
        parentBone: null,
        children: [],
      };
      poseEdit.bones.push(bone);
      poseEdit.byName.set(b.name, bone);
    }

    // Second pass: resolve parents + compute base local transforms.
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

    // Re-apply stored overrides so re-entering the editor doesn't drop the
    // user's prior edits. Applied in topological order (iteration order of
    // skeleton.bones is ancestor→descendant from the backend).
    if (storedOverrides) {
      for (const bone of poseEdit.bones) {
        const ov = storedOverrides[String(bone.jointId)];
        if (!ov || ov.length !== 3) continue;
        const e = new THREE.Euler(ov[0], ov[1], ov[2], "XYZ");
        bone.localDelta.setFromEuler(e);
        // Parent-propagated world quat reflects ancestors' already-applied
        // overrides; multiplying localDelta in on the right = local-frame
        // rotation relative to that propagated base (matches backend).
        const parentPropQ = _parentPropagatedWorldQuat(bone);
        bone.handle.quaternion.copy(parentPropQ).multiply(bone.localDelta);
        _propagateBoneTransform(bone);
      }
    }

    poseEdit.root = root;
    _rebuildBoneLines();
    // Translate gizmo enabled (rotate is intentionally not used — fine
    // rotation is via the X/Y/Z sliders). Gizmo is attached on selection
    // to the invisible IK target, which the user drags to translate the
    // selected bone via CCD on the parent chain.
    tControls.enabled = true;
    _tHelper.visible = false;   // shown on selection
    poseEdit.active = true;
    markDirty();
  }

  function exitPoseEdit() {
    if (!poseEdit.active) return;
    tControls.detach();
    tControls.enabled = false;
    _tHelper.visible = false;
    if (poseEdit.root) {
      scene.remove(poseEdit.root);
      poseEdit.root.traverse?.((o) => {
        if (o.isMesh || o.isLine) o.geometry?.dispose?.();
      });
      poseEdit.root = null;
    }
    poseEdit.bones = [];
    poseEdit.byName = new Map();
    poseEdit.selected = null;
    poseEdit.active = false;
    // Hand the gizmo back to the body anchor so the user can keep
    // translating / rotating the whole body after exiting pose-edit.
    if (meshGroup) {
      tControls.enabled = true;
      tControls.attach(meshGroup);
      _tHelper.visible = true;
    }
    markDirty();
  }

  function selectPoseBone(name) {
    if (!poseEdit.active) return;
    for (const b of poseEdit.bones) {
      b.handle.material = _poseHandleMat;
    }
    const bone = name ? poseEdit.byName.get(name) : null;
    if (!bone) {
      poseEdit.selected = null;
      tControls.detach();
      _tHelper.visible = false;
      markDirty();
      return;
    }
    bone.handle.material = _poseHandleSelectedMat;
    poseEdit.selected = bone;
    // Position the IK target at the bone's world location and attach the
    // translate gizmo to it. Bones with no parent (Hips) can still be
    // selected for inspection but won't drive IK (chain length < 2).
    bone.handle.getWorldPosition(_ikTarget.position);
    tControls.attach(_ikTarget);
    // Shrink the gizmo when a finger is selected — finger bones are tiny
    // (a phalanx is ~5 mm in mesh-local units), so a default-size gizmo
    // would overshoot finger reach by an order of magnitude on every drag.
    tControls.size = _isFingerBoneName(bone.name) ? 0.35 : 1.0;
    _tHelper.visible = true;
    markDirty();
  }

  function resetPoseBone(name) {
    const bone = poseEdit.byName.get(name);
    if (!bone) return;
    bone.localDelta.identity();
    // Snap bone back to parent-propagated state (localDelta = identity now).
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
    markDirty();
  }

  function resetAllPoseBones() {
    for (const bone of poseEdit.bones) {
      bone.localDelta.identity();
      bone.handle.position.copy(bone.base.worldPos);
      bone.handle.quaternion.copy(bone.base.worldQuat);
      if (poseEdit.onBoneChange) poseEdit.onBoneChange(bone.name, null);
    }
    // Wipe the Hips drag translation too — Reset = full restore to the
    // initial fit pose. Snap the meshGroup back to the fit anchor while
    // preserving the user's body-anchor (translate / rotate) state, so
    // "Reset all bones" doesn't undo a deliberate body offset.
    _hipsOffset.set(0, 0, 0);
    if (meshGroup) {
      meshGroup.position.copy(_meshFitOffset).add(_bodyXform.translate);
    }
    // Re-sync the IK gizmo target to the (now-restored) selected bone so
    // it doesn't dangle at the pre-reset world position.
    if (poseEdit.selected && meshGroup) {
      meshGroup.updateMatrixWorld(true);
      poseEdit.selected.handle.getWorldPosition(_ikTarget.position);
    }
    _rebuildBoneLines();
    markDirty();
  }

  function setPoseBoneLocalEuler(name, rx, ry, rz) {
    const bone = poseEdit.byName.get(name);
    if (!bone) return;
    const e = new THREE.Euler(rx, ry, rz, "XYZ");
    bone.localDelta.setFromEuler(e);
    // "Absolute" semantics: bone.handle.quaternion = parentPropagated * localDelta.
    const parentPropQ = _parentPropagatedWorldQuat(bone);
    bone.handle.quaternion.copy(parentPropQ).multiply(bone.localDelta);
    _propagateBoneTransform(bone);
    _rebuildBoneLines();
    markDirty();
  }

  function getPoseBoneLocalEuler(name) {
    const bone = poseEdit.byName.get(name);
    if (!bone) return null;
    const e = new THREE.Euler().setFromQuaternion(bone.localDelta, "XYZ");
    return [e.x, e.y, e.z];
  }

  function setPoseEditCallbacks({ onBoneChange, onBonePick, onDragStart, onDragEnd, onIkChainChange }) {
    poseEdit.onBoneChange    = onBoneChange    || null;
    poseEdit.onBonePick      = onBonePick      || null;
    poseEdit.onDragStart     = onDragStart     || null;
    poseEdit.onDragEnd       = onDragEnd       || null;
    poseEdit.onIkChainChange = onIkChainChange || null;
  }

  function savePng(filename) {
    // Swap to a white background and hide the ground grid for the capture,
    // then restore both so the on-screen view isn't disturbed. Using
    // toDataURL (synchronous) instead of toBlob avoids races with the tick
    // loop overwriting the back buffer before the async encode completes.
    const prevClearColor = renderer.getClearColor(new THREE.Color());
    const prevClearAlpha = renderer.getClearAlpha();
    const prevGridVisible = grid.visible;
    renderer.setClearColor(0xffffff, 1);
    grid.visible = false;
    try {
      renderer.render(scene, camera);
      const dataUrl = canvas.toDataURL("image/png");
      const a = document.createElement("a");
      a.href = dataUrl;
      a.download = filename || `render_${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } finally {
      renderer.setClearColor(prevClearColor, prevClearAlpha);
      grid.visible = prevGridVisible;
      // Next tick needs to redraw with the restored background + grid.
      _dirty = true;
    }
  }

  // Render-to-PNG-DataURL with a custom background and (optional) crop. Used
  // by the editor confirm pipeline to produce pose_image / body_preset_image
  // outputs. ``cropRect`` is in canvas BACKING-store pixels (i.e. drawing
  // buffer pixels — already multiplied by devicePixelRatio).
  // ``framing`` is "current" (use the orbit camera as-is) or "fit_full_body"
  // (frame the body height-fit, ground-aligned, suitable for body_preset_image).
  function captureImage({ bgColor = 0xffffff, cropRect = null, framing = "current" } = {}) {
    const prevClearColor = renderer.getClearColor(new THREE.Color());
    const prevClearAlpha = renderer.getClearAlpha();
    const prevGridVisible = grid.visible;
    const prevCamPos = camera.position.clone();
    const prevTarget = controls.target.clone();
    renderer.setClearColor(bgColor, 1);
    grid.visible = false;
    let resultUrl = null;
    let outW = 0, outH = 0;
    try {
      if (framing === "fit_full_body" && meshGroup) {
        const box = new THREE.Box3().setFromObject(meshGroup);
        const center = new THREE.Vector3();
        const size = new THREE.Vector3();
        box.getCenter(center);
        box.getSize(size);
        const fov = camera.fov * Math.PI / 180;
        // Distance at which size.y just fills the frame vertically.
        const distH = (Math.max(size.y, 1e-3) / 2) / Math.tan(fov / 2);
        // Horizontal fov = 2 * atan(aspect * tan(fov/2)).
        const distW = (Math.max(size.x, 1e-3) / 2)
                      / (Math.max(camera.aspect, 1e-3) * Math.tan(fov / 2));
        // 15% padding so the body doesn't kiss the frame edges.
        const dist = Math.max(distH, distW) * 1.15;
        controls.target.copy(center);
        camera.position.set(center.x, center.y, center.z + Math.max(dist, 1.5));
        camera.lookAt(center);
        controls.update();
      }
      renderer.render(scene, camera);
      if (cropRect && cropRect.width > 0 && cropRect.height > 0) {
        const { x, y, width, height } = cropRect;
        const cw = canvas.width, ch = canvas.height;
        const sx = Math.max(0, Math.min(cw - 1, Math.round(x)));
        const sy = Math.max(0, Math.min(ch - 1, Math.round(y)));
        const sw = Math.max(1, Math.min(cw - sx, Math.round(width)));
        const sh = Math.max(1, Math.min(ch - sy, Math.round(height)));
        const off = document.createElement("canvas");
        off.width = sw; off.height = sh;
        off.getContext("2d").drawImage(canvas, sx, sy, sw, sh, 0, 0, sw, sh);
        resultUrl = off.toDataURL("image/png");
        outW = sw; outH = sh;
      } else {
        resultUrl = canvas.toDataURL("image/png");
        outW = canvas.width; outH = canvas.height;
      }
    } finally {
      renderer.setClearColor(prevClearColor, prevClearAlpha);
      grid.visible = prevGridVisible;
      if (framing !== "current") {
        camera.position.copy(prevCamPos);
        controls.target.copy(prevTarget);
        controls.update();
      }
      _dirty = true;
    }
    return { dataUrl: resultUrl, width: outW, height: outH };
  }

  // Wipe the Hips drag translation and snap meshGroup back to the fit
  // anchor. Used by the outer code when a fresh pose is estimated and
  // any prior body translation should be discarded.
  function clearHipsTranslation() {
    _hipsOffset.set(0, 0, 0);
    if (meshGroup) {
      meshGroup.position.copy(_meshFitOffset).add(_bodyXform.translate);
    }
    markDirty();
  }

  // Read / write the current Hips translation. Used by the pose-edit UI
  // so it can snapshot Hips state at session start (for Reset All to
  // restore) without leaking _hipsOffset directly.
  function getHipsOffset() {
    return _hipsOffset.clone();
  }
  function setHipsOffset(vec3) {
    _hipsOffset.copy(vec3);
    if (meshGroup) {
      meshGroup.position
        .copy(_meshFitOffset)
        .add(_hipsOffset)
        .add(_bodyXform.translate);
    }
    markDirty();
  }

  // ---- Body-anchor gizmo: manual translate / rotate of the whole body --

  function setGizmoMode(mode /* "translate" | "rotate" */) {
    tControls.setMode(mode === "rotate" ? "rotate" : "translate");
    markDirty();
  }

  // Re-bind the gizmo to meshGroup. Used by the right-toolbar buttons
  // (and exit-IK transition) to make the body-anchor draggable again.
  function bindBodyGizmo() {
    if (poseEdit.active || !meshGroup) return;
    tControls.enabled = true;
    tControls.attach(meshGroup);
    _tHelper.visible = true;
    markDirty();
  }
  function unbindBodyGizmo() {
    if (tControls.object === meshGroup) {
      tControls.detach();
      _tHelper.visible = false;
      markDirty();
    }
  }

  function getBodyTransform() {
    return {
      translate: [_bodyXform.translate.x, _bodyXform.translate.y, _bodyXform.translate.z],
      rotate_deg: [
        THREE.MathUtils.radToDeg(_bodyXform.rotate.x),
        THREE.MathUtils.radToDeg(_bodyXform.rotate.y),
        THREE.MathUtils.radToDeg(_bodyXform.rotate.z),
      ],
    };
  }
  function setBodyTransform(translate, rotate_deg) {
    if (Array.isArray(translate)) {
      _bodyXform.translate.set(
        translate[0] || 0, translate[1] || 0, translate[2] || 0,
      );
    }
    if (Array.isArray(rotate_deg)) {
      _bodyXform.rotate.set(
        THREE.MathUtils.degToRad(rotate_deg[0] || 0),
        THREE.MathUtils.degToRad(rotate_deg[1] || 0),
        THREE.MathUtils.degToRad(rotate_deg[2] || 0),
        "XYZ",
      );
    }
    if (meshGroup) {
      meshGroup.position.copy(_meshFitOffset).add(_hipsOffset).add(_bodyXform.translate);
      meshGroup.rotation.copy(_bodyXform.rotate);
    }
    markDirty();
  }
  function resetBodyTranslate() {
    _bodyXform.translate.set(0, 0, 0);
    if (meshGroup) {
      meshGroup.position.copy(_meshFitOffset).add(_hipsOffset);
    }
    markDirty();
  }
  function resetBodyRotate() {
    _bodyXform.rotate.set(0, 0, 0);
    if (meshGroup) meshGroup.rotation.set(0, 0, 0);
    markDirty();
  }
  function setOnBodyTransformChanged(cb) { _onBodyTransformChanged = cb || null; }

  return {
    loadObj, loadFbxAnimated, savePng, captureImage, switchTab, hasCachedMesh, clearMesh,
    enterPoseEdit, exitPoseEdit, selectPoseBone, resetPoseBone,
    resetAllPoseBones, setPoseBoneLocalEuler, getPoseBoneLocalEuler,
    setPoseEditCallbacks, clearHipsTranslation,
    getHipsOffset, setHipsOffset,
    setGizmoMode, bindBodyGizmo, unbindBodyGizmo,
    getBodyTransform, setBodyTransform,
    resetBodyTranslate, resetBodyRotate,
    setOnBodyTransformChanged,
    isPoseEditActive: () => poseEdit.active,
    getCanvasSize: () => ({ width: canvas.width, height: canvas.height }),
    getCanvasClientSize: () => ({ width: canvas.clientWidth, height: canvas.clientHeight }),
  };
}

// ---------------------------------------------------------------------------
// Health polling
// ---------------------------------------------------------------------------

async function pollHealth() {
  try {
    const r = await fetch("/sam3d/api/health");
    const j = await r.json();
    healthEl.textContent = JSON.stringify(j, null, 2);
    statusEl.textContent = `ready — ${j.python} / torch ${j.torch ?? "n/a"} / ${j.cuda_device ?? "cpu"}`;
    statusEl.classList.add("ok"); statusEl.classList.remove("err", "busy");
  } catch (e) {
    healthEl.textContent = String(e);
    statusEl.textContent = "backend unreachable";
    statusEl.classList.add("err"); statusEl.classList.remove("ok", "busy");
  }
}

// ---------------------------------------------------------------------------
// File / image input
// ---------------------------------------------------------------------------

function onFileChosen(file) {
  if (!file) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewEl.src = url;
  previewEl.hidden = false;
  fileLabel.textContent = file.name;
  runBtn.disabled = false;
  runInfo.textContent = "";
}

fileInput.addEventListener("change", (e) => onFileChosen(e.target.files?.[0]));
// NOTE: `<label class="file-drop">` already forwards clicks to the inner
// <input type="file"> via HTML's label→control binding, so adding
// `fileInput.click()` here on top fires the dialog twice (the user ends
// up needing to select their image in both dialogs). Rely on the native
// label behaviour — only drag&drop needs custom wiring below.
["dragenter", "dragover"].forEach((ev) =>
  fileDrop.addEventListener(ev, (e) => {
    e.preventDefault(); fileDrop.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  fileDrop.addEventListener(ev, (e) => {
    e.preventDefault(); fileDrop.classList.remove("dragover");
  })
);
fileDrop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files?.[0];
  if (f) onFileChosen(f);
});

// ---------------------------------------------------------------------------
// Slider UI (built dynamically from /api/slider_schema)
// ---------------------------------------------------------------------------

function makeSlider(group, def) {
  const wrap = document.createElement("div");
  wrap.className = "slider-row";
  const label = document.createElement("label");
  label.textContent = def.key;
  label.title = def.key;
  const range = document.createElement("input");
  range.type = "range";
  range.min = def.min; range.max = def.max; range.step = def.step;
  range.value = settings[group][def.key] ?? def.default;
  const num = document.createElement("input");
  num.type = "number";
  num.min = def.min; num.max = def.max; num.step = def.step;
  num.value = range.value;

  const onInput = (v) => {
    const clamped = Math.max(def.min, Math.min(def.max, Number(v)));
    range.value = String(clamped);
    num.value = String(clamped);
    settings[group][def.key] = clamped;
    scheduleRender();
  };
  range.addEventListener("input", () => onInput(range.value));
  num.addEventListener("change", () => onInput(num.value));
  wrap.appendChild(label);
  wrap.appendChild(range);
  wrap.appendChild(num);
  return { wrap, range, num, def, group };
}

const sliderRefs = { body_params: {}, bone_lengths: {}, blendshapes: {} };

function buildSliders(el, group, defs) {
  el.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "slider-grid";
  for (const def of defs) {
    if (settings[group][def.key] === undefined) settings[group][def.key] = def.default;
    const s = makeSlider(group, def);
    grid.appendChild(s.wrap);
    sliderRefs[group][def.key] = s;
  }
  el.appendChild(grid);
}

function applySettingsToUi(newSettings) {
  for (const group of ["body_params", "bone_lengths", "blendshapes"]) {
    const src = newSettings[group] || {};
    for (const [k, ref] of Object.entries(sliderRefs[group])) {
      const v = Number(src[k] ?? ref.def.default);
      settings[group][k] = v;
      ref.range.value = String(v);
      ref.num.value = String(v);
    }
  }
}

// Called after rebinding `settings` to a different per-tab slot. Syncs the
// shared slider DOM to whatever the active tab's settings now holds, and
// backfills defaults for keys the tab hadn't touched yet.
function refreshSlidersFromSettings() {
  for (const group of ["body_params", "bone_lengths", "blendshapes"]) {
    for (const [k, ref] of Object.entries(sliderRefs[group])) {
      if (settings[group][k] === undefined) settings[group][k] = ref.def.default;
      const v = Number(settings[group][k]);
      ref.range.value = String(v);
      ref.num.value = String(v);
    }
  }
  if (!settings.pose_adjust) {
    settings.pose_adjust = { lean_correction: LEAN_CORRECTION_DEFAULT };
  } else if (settings.pose_adjust.lean_correction === undefined) {
    settings.pose_adjust.lean_correction = LEAN_CORRECTION_DEFAULT;
  }
}

// The lean-correction slider lives inside each tab's own panel (image /
// video), so we wire them up once at boot and keep them in sync with
// ``tabSettings[tab].pose_adjust.lean_correction``. Unlike the body /
// bone / blendshape sliders (shared DOM in the Body Preset panel), these
// are tab-scoped, so no tab-switch sync is needed.
//
// Image tab re-renders on ``input`` (cheap OBJ mesh swap), video tab only
// fires a rebuild on ``change`` (mouseup / Enter) because each animated
// FBX rebuild spawns a Blender subprocess that takes 10-30 s — dragging
// the knob would otherwise queue a stale rebuild per pointer move.
function initLeanSliders() {
  const cfgs = [
    { tab: "image", rangeId: "img-lean-range",   numId: "img-lean-num",
      liveRebuild: true  },
    { tab: "video", rangeId: "video-lean-range", numId: "video-lean-num",
      liveRebuild: false },
  ];
  for (const c of cfgs) {
    const range = document.getElementById(c.rangeId);
    const num = document.getElementById(c.numId);
    if (!range || !num) continue;
    const slot = tabSettings[c.tab];
    if (!slot.pose_adjust) {
      slot.pose_adjust = { lean_correction: LEAN_CORRECTION_DEFAULT };
    }
    const cur = Number(
      slot.pose_adjust.lean_correction ?? LEAN_CORRECTION_DEFAULT,
    );
    range.value = String(cur);
    num.value = String(cur);

    const writeValue = (v) => {
      let n = Number(v);
      if (!Number.isFinite(n)) n = 0;
      const clamped = Math.max(0, Math.min(1, n));
      range.value = String(clamped);
      num.value = String(clamped);
      slot.pose_adjust.lean_correction = clamped;
      return clamped;
    };
    const maybeFire = () => {
      const activeTab = document.querySelector(".tab-nav button.active")?.dataset.tab;
      if (activeTab === c.tab) scheduleRender();
    };

    range.addEventListener("input", () => {
      writeValue(range.value);
      if (c.liveRebuild) maybeFire();
    });
    range.addEventListener("change", () => {
      writeValue(range.value);
      if (!c.liveRebuild) maybeFire();
    });
    num.addEventListener("change", () => {
      writeValue(num.value);
      maybeFire();
    });
  }
}

// ---------------------------------------------------------------------------
// Pose-edit (rotation) mode — image tab only.
//
// Flow:
//   • Toggle button enters the mode → we lock the tab bar / file drop /
//     preset panel / lean slider, reveal the rotation panel, and ask the
//     viewer to build its bone overlay from the latest humanoid_skeleton
//     snapshot (captured on every /api/render response).
//   • TransformControls drags update `settings.pose_adjust.rotation_overrides`
//     live; only the skeleton overlay moves during a drag (no backend
//     roundtrip). On mouseup, scheduleRender() kicks a fresh render so the
//     mesh catches up.
//   • Export FBX/BVH remain available; while a Blender subprocess runs, we
//     additionally drop a `body.ui-export-busy` flag that greys out the
//     rotation UI too — per spec, "生成中は rotation UI も一時ロック".
//   • Exiting the mode keeps the override values in settings, so the next
//     render / FBX export still carries them ("終了時の挙動: 保持").
// ---------------------------------------------------------------------------

let poseEditMode = false;
const POSE_DEG2RAD = Math.PI / 180;
const POSE_RAD2DEG = 180 / Math.PI;

function _poseLockSidebar(lock) {
  document.body.classList.toggle("pose-edit-mode", lock);
  // Tab nav: disable the inactive ones so keyboard / programmatic clicks
  // can't sneak past the CSS pointer-events block either.
  document.querySelectorAll(".tab-nav button[data-tab]").forEach((b) => {
    if (!b.classList.contains("active")) b.disabled = lock;
  });
  // Explicit form-control disabling (CSS handles the visual gray-out for
  // container elements with [data-pose-lock]; form controls still need
  // disabled=true so Tab-key navigation / keyboard input can't reach them).
  const toDisable = [
    "file-input", "run-btn",
    "preset-select", "load-preset-btn",
    "body-preset-json-input",
    "img-lean-range", "img-lean-num",
  ];
  for (const id of toDisable) {
    const el = document.getElementById(id);
    if (el) el.disabled = lock;
  }
}

function _poseLockForExport(busy) {
  document.body.classList.toggle("ui-export-busy", busy);
}

function _syncRotationSlidersFromBone(boneName) {
  const xr = document.getElementById("pose-rot-x-range");
  const yr = document.getElementById("pose-rot-y-range");
  const zr = document.getElementById("pose-rot-z-range");
  const xn = document.getElementById("pose-rot-x-num");
  const yn = document.getElementById("pose-rot-y-num");
  const zn = document.getElementById("pose-rot-z-num");
  if (!xr || !yr || !zr) return;
  const euler = boneName ? viewer.getPoseBoneLocalEuler(boneName) : null;
  const deg = euler ? euler.map((v) => Number((v * POSE_RAD2DEG).toFixed(2))) : [0, 0, 0];
  xr.value = String(deg[0]); xn.value = String(deg[0]);
  yr.value = String(deg[1]); yn.value = String(deg[1]);
  zr.value = String(deg[2]); zn.value = String(deg[2]);
  const dis = !boneName;
  xr.disabled = yr.disabled = zr.disabled = dis;
  xn.disabled = yn.disabled = zn.disabled = dis;
  const resetBoneBtn = document.getElementById("pose-reset-bone-btn");
  if (resetBoneBtn) resetBoneBtn.disabled = dis;
}

function _rebuildBoneDropdown(skeleton) {
  const sel = document.getElementById("pose-bone-select");
  if (!sel) return;
  sel.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = window.i18n?.t("pose.selected_bone") || "(bone)";
  sel.appendChild(placeholder);
  for (const b of (skeleton?.bones || [])) {
    const opt = document.createElement("option");
    opt.value = b.name;
    const label = window.i18n?.t(`pose.bone.${b.name}`);
    opt.textContent = (label && label !== `pose.bone.${b.name}`) ? label : b.name;
    sel.appendChild(opt);
  }
  sel.value = "";
}

function _writeOverride(boneName, eulerRad) {
  const bone = (tabHumanoidSkeleton.image?.bones || []).find((b) => b.name === boneName);
  if (!bone) return;
  const key = String(bone.joint_id);
  if (!settings.pose_adjust) settings.pose_adjust = { lean_correction: LEAN_CORRECTION_DEFAULT };
  if (!settings.pose_adjust.rotation_overrides) settings.pose_adjust.rotation_overrides = {};
  if (!eulerRad) {
    delete settings.pose_adjust.rotation_overrides[key];
  } else {
    settings.pose_adjust.rotation_overrides[key] = [
      Number(eulerRad[0]), Number(eulerRad[1]), Number(eulerRad[2]),
    ];
  }
}

function initPoseEditUi() {
  const toggleBtn = document.getElementById("pose-edit-toggle-btn");
  const panel     = document.getElementById("pose-edit-panel");
  const boneSel   = document.getElementById("pose-bone-select");
  const resetBone = document.getElementById("pose-reset-bone-btn");
  const resetAll  = document.getElementById("pose-reset-all-btn");
  if (!toggleBtn || !panel) return;

  // Snapshot of pose-adjust state at the START of the current pose-edit
  // session. "Reset all bones" / "Reset bone" restore to THIS, not to a
  // pristine T-pose — so a user who made changes in a previous session,
  // closed pose-edit, then re-opened and clicked Reset doesn't lose the
  // prior session's adjustments.
  let _sessionBaseline = null;

  // Undo / redo stacks for pose-adjust operations. One entry per drag
  // (D&D from gizmo press to release) or single-shot edit (slider, reset).
  // Each entry is a complete snapshot of {rotation_overrides, hipsOffset}
  // — small enough that storing many is fine.
  const _UNDO_MAX = 50;
  const _undoStack = [];
  const _redoStack = [];
  let _pendingUndoSnap = null;   // captured at op start, pushed at op end
  const undoBtn = document.getElementById("pose-undo-btn");
  const redoBtn = document.getElementById("pose-redo-btn");

  function _snapshotPoseState() {
    const overrides = settings.pose_adjust?.rotation_overrides || {};
    return {
      rotationOverrides: Object.fromEntries(
        Object.entries(overrides).map(([k, v]) => [k, [v[0], v[1], v[2]]]),
      ),
      hipsOffset: viewer.getHipsOffset(),
    };
  }

  function _applyPoseState(snap) {
    if (!settings.pose_adjust) settings.pose_adjust = { lean_correction: LEAN_CORRECTION_DEFAULT };
    settings.pose_adjust.rotation_overrides = Object.fromEntries(
      Object.entries(snap.rotationOverrides).map(([k, v]) => [k, [v[0], v[1], v[2]]]),
    );
    viewer.setHipsOffset(snap.hipsOffset);
    if (viewer.isPoseEditActive()) {
      _rebuildOverlay(settings.pose_adjust.rotation_overrides);
    }
    _syncRotationSlidersFromBone(boneSel.value || "");
    scheduleRender();
  }

  function _refreshUndoButtons() {
    if (undoBtn) undoBtn.disabled = _undoStack.length === 0;
    if (redoBtn) redoBtn.disabled = _redoStack.length === 0;
  }

  function _pushUndoEntry(snap) {
    _undoStack.push(snap);
    if (_undoStack.length > _UNDO_MAX) _undoStack.shift();
    _redoStack.length = 0;
    _refreshUndoButtons();
  }

  // Capture the current state — call at the START of an op.
  function _beginPoseOp() {
    _pendingUndoSnap = _snapshotPoseState();
  }
  // Commit the captured snapshot to the undo stack — call at op END.
  function _commitPoseOp() {
    if (_pendingUndoSnap) {
      _pushUndoEntry(_pendingUndoSnap);
      _pendingUndoSnap = null;
    }
  }

  function _undoPoseEdit() {
    if (_undoStack.length === 0) return;
    _redoStack.push(_snapshotPoseState());
    const snap = _undoStack.pop();
    _applyPoseState(snap);
    _refreshUndoButtons();
  }
  function _redoPoseEdit() {
    if (_redoStack.length === 0) return;
    _undoStack.push(_snapshotPoseState());
    const snap = _redoStack.pop();
    _applyPoseState(snap);
    _refreshUndoButtons();
  }

  // Keyboard shortcuts — only active inside pose-edit mode so they don't
  // hijack browser undo elsewhere.
  window.addEventListener("keydown", (ev) => {
    if (!poseEditMode) return;
    if (!ev.ctrlKey && !ev.metaKey) return;
    const k = ev.key.toLowerCase();
    if (k === "z" && !ev.shiftKey) {
      ev.preventDefault();
      _undoPoseEdit();
    } else if (k === "y" || (k === "z" && ev.shiftKey)) {
      ev.preventDefault();
      _redoPoseEdit();
    }
  });

  // Tear down + rebuild the overlay with the given override map. Used
  // after an IK drag ends (clear cascade drift) and after Reset All
  // (snap back to baseline). Selection is preserved.
  function _rebuildOverlay(overrides) {
    if (!viewer.isPoseEditActive()) return;
    const skel = tabHumanoidSkeleton.image;
    if (!skel) return;
    const previousSelected = boneSel.value || null;
    viewer.exitPoseEdit();
    viewer.enterPoseEdit(skel, overrides || {});
    if (previousSelected) viewer.selectPoseBone(previousSelected);
  }

  function enter() {
    // Need a base skeleton. If /api/render hasn't completed yet, nudge.
    const skel = tabHumanoidSkeleton.image;
    if (!skel || !skel.bones || skel.bones.length === 0) {
      runInfo.textContent = window.i18n?.t("pose.edit_locked") || "skeleton not ready yet";
      return;
    }
    // Seed overrides dict so the viewer restores prior edits on re-enter.
    if (!settings.pose_adjust) settings.pose_adjust = { lean_correction: LEAN_CORRECTION_DEFAULT };
    const stored = settings.pose_adjust.rotation_overrides || {};
    // Snapshot the pose-adjust state as the Reset baseline for THIS
    // session. Deep-copy each Euler triple so further edits don't mutate
    // the snapshot.
    _sessionBaseline = {
      rotationOverrides: Object.fromEntries(
        Object.entries(stored).map(([k, v]) => [k, [v[0], v[1], v[2]]]),
      ),
      hipsOffset: viewer.getHipsOffset(),
    };

    viewer.setPoseEditCallbacks({
      onBoneChange: (boneName, eulerRad) => {
        _writeOverride(boneName, eulerRad);
        _syncRotationSlidersFromBone(boneName);
        // Live mesh preview during drag. ``triggerRender`` short-circuits
        // while a fetch is in flight (it just sets renderDirty + re-fires
        // on completion), so continuous change events collapse into roughly
        // one render per backend roundtrip (~5 fps) instead of queuing.
        // This beats ``scheduleRender`` here, whose 60 ms debounce resets
        // on every event and would therefore never fire mid-drag.
        triggerRender();
      },
      onBonePick: (boneName) => {
        boneSel.value = boneName;
        _syncRotationSlidersFromBone(boneName);
      },
      onDragStart: () => {
        // Snapshot the pre-drag state so the whole D&D becomes a single
        // undo entry on release.
        _beginPoseOp();
      },
      onDragEnd: () => {
        // One final render with the drag's last state — in case the last
        // change event was coalesced into an in-flight fetch and its
        // follow-up was skipped because renderDirty was cleared between.
        // The pose-edit overlay rebuild is DEFERRED to the moment that
        // render's response arrives (see ``callRender``'s finally) so
        // the bones snap onto the FRESHEST posed skeleton — a sync
        // rebuild here would use the previous frame's skeleton and
        // leave a 1-frame misalignment between bones and mesh.
        _pendingOverlayRebuild = () =>
          _rebuildOverlay(settings.pose_adjust?.rotation_overrides);
        scheduleRender();
        _commitPoseOp();
      },
      // IK drag — viewer ran CCD on a parent chain and now needs us to
      // persist the rotation overrides for every affected bone, refresh
      // the slider readout for whichever bone is selected (it may be the
      // end effector OR one of the rotated parents), and trigger a live
      // mesh re-render.
      onIkChainChange: (changes) => {
        for (const { boneName, eulerRad } of changes) {
          _writeOverride(boneName, eulerRad);
        }
        const sel = boneSel.value;
        if (sel) _syncRotationSlidersFromBone(sel);
        triggerRender();
      },
    });
    viewer.enterPoseEdit(skel, stored);

    poseEditMode = true;
    _poseLockSidebar(true);
    panel.hidden = false;
    toggleBtn.textContent = window.i18n?.t("pose.edit_exit") || "Finish pose adjust";
    _rebuildBoneDropdown(skel);
    _syncRotationSlidersFromBone("");
  }

  function exit() {
    viewer.exitPoseEdit();
    poseEditMode = false;
    _poseLockSidebar(false);
    panel.hidden = true;
    toggleBtn.textContent = window.i18n?.t("pose.edit_enter") || "Adjust pose (rotate)";
    // One final render with the final overrides so the viewer OBJ reflects
    // the edits even if the last drag's scheduleRender was still in flight.
    scheduleRender();
  }

  toggleBtn.addEventListener("click", () => {
    if (poseEditMode) exit(); else enter();
    // Notify other UI (right-toolbar visibility hook) that pose-edit
    // mode just toggled. Cheap CustomEvent — no payload needed since
    // listeners read the latest state from ``viewer.isPoseEditActive()``.
    document.dispatchEvent(new CustomEvent("sam3d-pose-edit-changed"));
  });

  boneSel.addEventListener("change", () => {
    viewer.selectPoseBone(boneSel.value || null);
    _syncRotationSlidersFromBone(boneSel.value);
  });

  // Number / range sliders — two-way bind to the selected bone. ABSOLUTE
  // semantics: slider value == bone's current local delta; moving the
  // slider resets the bone to parent-propagated state and applies the new
  // delta (handled inside setPoseBoneLocalEuler).
  const writeEuler = () => {
    const name = boneSel.value;
    if (!name) return;
    const xv = Number(document.getElementById("pose-rot-x-num").value) || 0;
    const yv = Number(document.getElementById("pose-rot-y-num").value) || 0;
    const zv = Number(document.getElementById("pose-rot-z-num").value) || 0;
    const rx = xv * POSE_DEG2RAD;
    const ry = yv * POSE_DEG2RAD;
    const rz = zv * POSE_DEG2RAD;
    viewer.setPoseBoneLocalEuler(name, rx, ry, rz);
    _writeOverride(name, (rx === 0 && ry === 0 && rz === 0) ? null : [rx, ry, rz]);
  };
  const linkPair = (rangeId, numId, live) => {
    const range = document.getElementById(rangeId);
    const num = document.getElementById(numId);
    if (!range || !num) return;
    let sliderOpActive = false;
    range.addEventListener("input", () => {
      if (!sliderOpActive) { _beginPoseOp(); sliderOpActive = true; }
      num.value = range.value;
      writeEuler();
      // Live mesh preview while dragging the slider — see onBoneChange above
      // for why triggerRender is preferred over the debounced scheduleRender.
      triggerRender();
    });
    range.addEventListener("change", () => {
      num.value = range.value;
      writeEuler();
      scheduleRender();
      if (sliderOpActive) { _commitPoseOp(); sliderOpActive = false; }
    });
    num.addEventListener("change", () => {
      _beginPoseOp();
      range.value = num.value;
      writeEuler();
      scheduleRender();
      _commitPoseOp();
    });
  };
  linkPair("pose-rot-x-range", "pose-rot-x-num", true);
  linkPair("pose-rot-y-range", "pose-rot-y-num", true);
  linkPair("pose-rot-z-range", "pose-rot-z-num", true);

  resetBone.addEventListener("click", () => {
    const name = boneSel.value;
    if (!name) return;
    _beginPoseOp();
    // Restore THIS bone to its session-baseline value (or identity if it
    // wasn't set at session start). Earlier-session edits on other bones
    // are untouched.
    const meta = (tabHumanoidSkeleton.image?.bones || []).find((b) => b.name === name);
    const baselineEuler = meta && _sessionBaseline
      ? _sessionBaseline.rotationOverrides[String(meta.joint_id)]
      : null;
    if (baselineEuler) {
      viewer.setPoseBoneLocalEuler(name, baselineEuler[0], baselineEuler[1], baselineEuler[2]);
      _writeOverride(name, baselineEuler);
    } else {
      viewer.resetPoseBone(name);
      _writeOverride(name, null);
    }
    _syncRotationSlidersFromBone(name);
    scheduleRender();
    _commitPoseOp();
  });
  resetAll.addEventListener("click", () => {
    _beginPoseOp();
    if (!_sessionBaseline) {
      // No baseline (shouldn't happen if pose-edit is active) — fall back
      // to a full clear.
      viewer.resetAllPoseBones();
      if (settings.pose_adjust) settings.pose_adjust.rotation_overrides = {};
      _syncRotationSlidersFromBone(boneSel.value || "");
      scheduleRender();
      _commitPoseOp();
      return;
    }
    // Restore settings.rotation_overrides to the baseline snapshot.
    if (!settings.pose_adjust) settings.pose_adjust = { lean_correction: LEAN_CORRECTION_DEFAULT };
    settings.pose_adjust.rotation_overrides = Object.fromEntries(
      Object.entries(_sessionBaseline.rotationOverrides).map(
        ([k, v]) => [k, [v[0], v[1], v[2]]],
      ),
    );
    // Restore Hips translation to baseline.
    viewer.setHipsOffset(_sessionBaseline.hipsOffset);
    // Rebuild the overlay so localDeltas reflect the restored baseline.
    _rebuildOverlay(settings.pose_adjust.rotation_overrides);
    _syncRotationSlidersFromBone(boneSel.value || "");
    scheduleRender();
    _commitPoseOp();
  });

  // Wire up Undo / Redo buttons.
  if (undoBtn) undoBtn.addEventListener("click", () => _undoPoseEdit());
  if (redoBtn) redoBtn.addEventListener("click", () => _redoPoseEdit());

  // Expose the export-lock toggle for the FBX / BVH handlers further down.
  window.__poseExportLock = _poseLockForExport;
  // Allow other modules (e.g. the /api/process success path) to wipe the
  // undo/redo history when the underlying pose changes — old snapshots
  // wouldn't make sense against a fresh skeleton.
  window.__resetPoseUndoHistory = () => {
    _undoStack.length = 0;
    _redoStack.length = 0;
    _pendingUndoSnap = null;
    _refreshUndoButtons();
  };
}

// Sync the shared Body Preset panel DOM (preset dropdown, source info text,
// JSON upload label) to the active tab's cached char state.
function refreshBodyPresetPanelFromState() {
  const name = document.querySelector(".tab-nav button.active")?.dataset.tab || "image";
  const state = tabBodyPresetState[name];
  if (!state) return;
  presetSelect.value = state.preset ?? "";
  bodyPresetSourceInfo.textContent = state.sourceText ?? "";
  bodyPresetJsonLabel.textContent =
    state.jsonLabel ??
    (window.i18n?.t("body_preset.json_drop") || "Upload body preset JSON (optional)");
  // Clear the <input type="file"> so re-selecting the same file still triggers change.
  bodyPresetJsonInput.value = "";
}

// True when at least one slider differs from its declared default. Used to
// decide whether to trigger an immediate re-render after /api/process
// (the pipeline's initial OBJ is always the neutral body, so if the user
// had already staged a preset / uploaded JSON we need to apply it).
function _settingsNonDefault() {
  for (const group of ["body_params", "bone_lengths", "blendshapes"]) {
    for (const [k, ref] of Object.entries(sliderRefs[group])) {
      if (Math.abs((settings[group][k] ?? ref.def.default) - ref.def.default) > 1e-9) {
        return true;
      }
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Debounced render request
// ---------------------------------------------------------------------------

let renderInFlight = false;
let renderDirty = false;
// Pose-edit overlay rebuild deferred to the next render-settle. Set by
// ``onDragEnd`` (see ``initPoseEditUi``) to a closure that captures the
// CURRENT pose_adjust state; ``callRender``'s finally invokes it once
// the queue drains so the rebuild uses the FRESHEST skeleton from the
// last response — eliminates the 1-frame misalignment a sync rebuild
// at drag-end-time would leave.
let _pendingOverlayRebuild = null;
let debounceTimer = null;

function scheduleRender() {
  // On the video tab, once motion inference has produced a cache we route
  // to /api/build_animated_fbx (animated FBX). Without motion the video
  // tab falls through to /api/render with MAKE_JOB_ID so the viewport
  // still shows the neutral MHR body with the tab's body preset settings.
  const activeTab = document.querySelector(".tab-nav button.active")?.dataset.tab;
  if (activeTab === "video" && currentMotionId) {
    rebuildAnimatedFbx();
    return;
  }
  if (!currentJobId) return;
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(triggerRender, 60);
}

// Most recent humanoid skeleton snapshot returned by /api/render. Stored
// per tab so entering the rotation editor after switching back has a
// consistent "base" frame. Updated on every successful render.
const tabHumanoidSkeleton = { image: null, video: null, make: null };

async function triggerRender() {
  if (!currentJobId) return;
  if (renderInFlight) { renderDirty = true; return; }
  renderInFlight = true;
  renderDirty = false;
  try {
    const t0 = performance.now();
    const r = await fetch("/sam3d/api/render", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ job_id: currentJobId, settings }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const j = await r.json();
    const activeTab = document.querySelector(".tab-nav button.active")?.dataset.tab;
    if (activeTab && activeTab in tabHumanoidSkeleton) {
      tabHumanoidSkeleton[activeTab] = j.humanoid_skeleton || null;
    }
    // Editor: keep the live confirm-payload up to date with whatever
    // sliders / pose adjust state the user just shipped to the renderer.
    refreshEditorPosePayload();
    refreshEditorBodyPresetPayload();
    await viewer.loadObj(j.obj_url, currentJobId);
    // First rendered frame unlocks the PNG save button. Export FBX is
    // image-tab only and gated separately by /api/process success — we
    // don't flip it here so the neutral body preview before inference
    // can't be exported as a rigged FBX.
    renderAvailable = true;
    const savePngBtn = document.getElementById("save-render-btn");
    if (savePngBtn) savePngBtn.disabled = false;
    const rangeBtn = document.getElementById("range-select-btn");
    if (rangeBtn) rangeBtn.disabled = false;
    // Clear the initial viewport hint once a mesh is actually on screen.
    if (overlay.textContent) overlay.textContent = "";
    const ms = Math.round(performance.now() - t0);
    renderInfo.textContent = `render ${j.elapsed_sec}s (rt ${ms}ms)`;
  } catch (e) {
    console.error(e);
    renderInfo.textContent = `render error: ${e.message || e}`;
  } finally {
    renderInFlight = false;
    if (renderDirty) {
      triggerRender();
    } else if (_pendingOverlayRebuild) {
      // The drag ended and the render queue has drained. The skeleton
      // we just received is the freshest one possible; rebuild the
      // pose-edit overlay against it so IK bones snap exactly onto
      // the rendered mesh's joint positions (matches Pose Editor +
      // behaviour, eliminates 1-frame misalignment).
      const cb = _pendingOverlayRebuild;
      _pendingOverlayRebuild = null;
      try { cb(); } catch (e) { console.warn("overlay rebuild failed:", e); }
    }
  }
}

// ---------------------------------------------------------------------------
// Preset I/O
// ---------------------------------------------------------------------------

async function refreshPresets() {
  try {
    const r = await fetch("/sam3d/api/presets");
    const j = await r.json();
    presetSelect.innerHTML = '<option value="">(preset)</option>';
    for (const name of (j.presets || [])) {
      const opt = document.createElement("option");
      opt.value = name; opt.textContent = name;
      presetSelect.appendChild(opt);
    }
  } catch (e) { console.error(e); }
}

loadPresetBtn.addEventListener("click", async () => {
  const name = presetSelect.value;
  if (!name) return;
  try {
    const r = await fetch(`/sam3d/api/preset/${encodeURIComponent(name)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const preset = await r.json();
    applySettingsToUi(preset);
    const prefix = window.i18n?.t("body_preset.preset_loaded") || "preset applied:";
    const sourceText = `${prefix} ${name}`;
    bodyPresetSourceInfo.textContent = sourceText;
    // Reset any lingering JSON upload label — picking a preset is the new source.
    bodyPresetJsonInput.value = "";
    const defaultJsonLabel = window.i18n?.t("body_preset.json_drop") || "Upload body preset JSON (optional)";
    bodyPresetJsonLabel.textContent = defaultJsonLabel;
    // Persist to the active tab's char state so tab switches restore it.
    const activeTab = document.querySelector(".tab-nav button.active")?.dataset.tab;
    if (activeTab && tabBodyPresetState[activeTab]) {
      tabBodyPresetState[activeTab].preset = name;
      tabBodyPresetState[activeTab].sourceText = sourceText;
      tabBodyPresetState[activeTab].jsonLabel = null;
    }
    scheduleRender();
  } catch (e) { renderInfo.textContent = `load preset failed: ${e.message || e}`; }
});

// Upload a body preset JSON → parse → push into settings + sliders → re-render.
// The uploaded values override the currently-selected preset (whichever
// source runs last wins; that matches what the user expects).
async function handleCharJsonFile(file) {
  if (!file) return;
  try {
    const text = await file.text();
    const parsed = JSON.parse(text);
    if (typeof parsed !== "object" || parsed === null) {
      throw new Error("JSON must be an object with body_params / bone_lengths / blendshapes");
    }
    applySettingsToUi(parsed);
    const prefix = window.i18n?.t("body_preset.json_loaded") || "JSON applied:";
    const sourceText = `${prefix} ${file.name}`;
    bodyPresetSourceInfo.textContent = sourceText;
    bodyPresetJsonLabel.textContent = file.name;
    // Deselect any preset — it's been overridden by the upload.
    presetSelect.value = "";
    const activeTab = document.querySelector(".tab-nav button.active")?.dataset.tab;
    if (activeTab && tabBodyPresetState[activeTab]) {
      tabBodyPresetState[activeTab].preset = "";
      tabBodyPresetState[activeTab].sourceText = sourceText;
      tabBodyPresetState[activeTab].jsonLabel = file.name;
    }
    scheduleRender();
  } catch (err) {
    bodyPresetSourceInfo.textContent = `JSON parse error: ${err.message || err}`;
  }
}

bodyPresetJsonInput.addEventListener("change", (e) => handleCharJsonFile(e.target.files?.[0]));
["dragenter", "dragover"].forEach((ev) =>
  bodyPresetJsonDrop.addEventListener(ev, (e) => { e.preventDefault(); bodyPresetJsonDrop.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((ev) =>
  bodyPresetJsonDrop.addEventListener(ev, (e) => { e.preventDefault(); bodyPresetJsonDrop.classList.remove("dragover"); })
);
bodyPresetJsonDrop.addEventListener("drop", (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f) handleCharJsonFile(f);
});

resetBtn.addEventListener("click", () => {
  for (const group of ["body_params", "bone_lengths", "blendshapes"]) {
    for (const [k, ref] of Object.entries(sliderRefs[group])) {
      settings[group][k] = ref.def.default;
      ref.range.value = String(ref.def.default);
      ref.num.value = String(ref.def.default);
    }
  }
  // Reset is make-tab only (button is inside data-tab-show="make"). Clear
  // the make tab's preset/JSON source too so the Body Preset panel truthfully
  // reflects "no source".
  presetSelect.value = "";
  bodyPresetSourceInfo.textContent = "";
  bodyPresetJsonInput.value = "";
  bodyPresetJsonLabel.textContent =
    window.i18n?.t("body_preset.json_drop") || "Upload body preset JSON (optional)";
  if (tabBodyPresetState.make) {
    tabBodyPresetState.make.preset = "";
    tabBodyPresetState.make.sourceText = "";
    tabBodyPresetState.make.jsonLabel = null;
  }
  scheduleRender();
});

exportFbxBtn.addEventListener("click", async () => {
  if (!currentJobId) { fbxInfo.textContent = "先にポーズを推定してください"; return; }
  exportFbxBtn.disabled = true;
  if (window.__poseExportLock) window.__poseExportLock(true);
  fbxInfo.textContent = "Blender subprocess 起動中... (初回は数秒かかる)";
  try {
    const t0 = performance.now();
    const r = await fetch("/sam3d/api/export_fbx", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ job_id: currentJobId, settings }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const j = await r.json();
    const ms = Math.round(performance.now() - t0);
    // Server always writes to tmp/rigged.fbx; offer a meaningful save name
    // so the user-side Save dialog does not default to "rigged.fbx".
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const savedAs = `body3d_rigged_${stamp}.fbx`;
    const a = document.createElement("a");
    a.href = j.fbx_url;
    a.download = savedAs;
    document.body.appendChild(a);
    a.click();
    a.remove();
    fbxInfo.textContent = `FBX ${j.elapsed_sec}s (wall ${ms}ms) — saved as ${savedAs}`;
  } catch (e) {
    console.error(e);
    fbxInfo.textContent = `FBX export error: ${e.message || e}`;
  } finally {
    exportFbxBtn.disabled = false;
    if (window.__poseExportLock) window.__poseExportLock(false);
  }
});

exportBvhBtn?.addEventListener("click", async () => {
  if (!currentJobId) {
    if (bvhInfo) bvhInfo.textContent = "Run pose estimation first";
    return;
  }
  exportBvhBtn.disabled = true;
  if (window.__poseExportLock) window.__poseExportLock(true);
  if (bvhInfo) bvhInfo.textContent = "Blender subprocess running... (converting to BVH)";
  try {
    const t0 = performance.now();
    const r = await fetch("/sam3d/api/export_bvh", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ job_id: currentJobId, settings, strength: 1.0 }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const j = await r.json();
    const ms = Math.round(performance.now() - t0);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const savedAs = `body3d_rigged_${stamp}.bvh`;
    triggerDownload(j.bvh_url, savedAs);
    if (bvhInfo) bvhInfo.textContent = `BVH ${j.elapsed_sec}s (wall ${ms}ms) - saved as ${savedAs}`;
  } catch (e) {
    console.error(e);
    if (bvhInfo) bvhInfo.textContent = `BVH export error: ${e.message || e}`;
  } finally {
    exportBvhBtn.disabled = false;
    if (window.__poseExportLock) window.__poseExportLock(false);
  }
});

// ---------------------------------------------------------------------------
// Inference lock — aborts the in-flight fetch and freezes tab navigation so
// the user can't switch contexts mid-run. Client-side abort drops the
// response; the server keeps grinding on the current request until
// completion. (True server-side cancellation would require threading an
// abort signal through the model forward pass.)
// ---------------------------------------------------------------------------

let currentAbortController = null;
let inferenceLocked = false;

// All interactive form-control IDs that should be frozen while inference is
// in flight. The active run button is excluded by name in lockUiForInference()
// because it doubles as the stop button while running.
const INFERENCE_LOCK_IDS = [
  // Header
  "lang-select",
  // Image tab
  "file-input",
  "img-lean-range", "img-lean-num",
  "pose-edit-toggle-btn",
  "pose-bone-select",
  "pose-rot-x-range", "pose-rot-x-num",
  "pose-rot-y-range", "pose-rot-y-num",
  "pose-rot-z-range", "pose-rot-z-num",
  "pose-undo-btn", "pose-redo-btn",
  "pose-reset-bone-btn", "pose-reset-all-btn",
  "save-render-btn", "range-select-btn", "bg-color-btn",
  "export-fbx-btn", "export-bvh-btn",
  // Video tab
  "video-input",
  "video-bbox-thr", "video-root-mode",
  "video-fps", "video-stride", "video-max-frames",
  "video-lean-range", "video-lean-num",
  "video-download-btn", "video-download-bvh-btn",
  // Make tab
  "reset-btn", "make-download-btn",
  // Admin tab
  "pack-select", "pack-switch-btn",
  "pack-new-name", "pack-clone-btn",
  "pack-delete-btn",
  "fbx-input", "fbx-rebuild-btn",
  // Body Preset section (shared)
  "preset-select", "load-preset-btn", "body-preset-json-input",
  // The other tab's run button (one-of-two should always be locked).
  "run-btn", "video-run-btn",
];

let _inferenceSavedDisabled = null;

function lockUiForInference(activeRunBtn) {
  _inferenceSavedDisabled = {};
  for (const id of INFERENCE_LOCK_IDS) {
    if (activeRunBtn && id === activeRunBtn.id) continue;
    const el = document.getElementById(id);
    if (!el) continue;
    _inferenceSavedDisabled[id] = el.disabled;
    el.disabled = true;
  }
  // Tabs (active one stays clickable visually but the click is a no-op
  // because activateTab() refuses to switch while inferenceLocked).
  document.querySelectorAll(".tab-nav button[data-tab]").forEach((b) => {
    _inferenceSavedDisabled["__tab__" + b.dataset.tab] = b.disabled;
    if (!b.classList.contains("active")) b.disabled = true;
  });
  // Dynamic Body Preset sliders (Make tab) — created at runtime, so disable
  // the entire container's form controls in one sweep. We don't bother
  // saving prior state: these sliders aren't selectively disabled outside
  // of inference, so the unlock side just re-enables them all.
  for (const containerId of ["body-params-sliders", "bone-lengths-sliders", "blendshapes-sliders"]) {
    const c = document.getElementById(containerId);
    if (!c) continue;
    c.querySelectorAll("input, select, button").forEach((el) => {
      el.disabled = true;
    });
  }
}

function unlockUiForInference() {
  if (!_inferenceSavedDisabled) return;
  for (const id of INFERENCE_LOCK_IDS) {
    if (!(id in _inferenceSavedDisabled)) continue;
    const el = document.getElementById(id);
    if (el) el.disabled = _inferenceSavedDisabled[id];
  }
  document.querySelectorAll(".tab-nav button[data-tab]").forEach((b) => {
    const key = "__tab__" + b.dataset.tab;
    if (key in _inferenceSavedDisabled) b.disabled = _inferenceSavedDisabled[key];
  });
  for (const containerId of ["body-params-sliders", "bone-lengths-sliders", "blendshapes-sliders"]) {
    const c = document.getElementById(containerId);
    if (!c) continue;
    c.querySelectorAll("input, select, button").forEach((el) => {
      el.disabled = false;
    });
  }
  _inferenceSavedDisabled = null;
}

// Swap a run button into / out of "stop" mode: red styling + new label.
// `runI18nKey` and `stopI18nKey` are the data-i18n keys for the two states
// so language switches keep the right text.
function _setRunButtonRunning(btn, running, runI18nKey, stopI18nKey) {
  if (!btn) return;
  if (running) {
    btn.classList.add("is-running");
    btn.dataset.i18n = stopI18nKey;
    btn.textContent = window.i18n?.t(stopI18nKey) || "推定を停止";
  } else {
    btn.classList.remove("is-running");
    btn.dataset.i18n = runI18nKey;
    btn.textContent = window.i18n?.t(runI18nKey)
      || (runI18nKey === "video.run" ? "モーションを推定" : "ポーズを推定");
  }
}

function beginInference(runBtnEl, runI18nKey, stopI18nKey) {
  currentAbortController = new AbortController();
  inferenceLocked = true;
  lockUiForInference(runBtnEl);
  _setRunButtonRunning(runBtnEl, true, runI18nKey, stopI18nKey);
  // The running button itself must stay clickable so the user can stop.
  if (runBtnEl) runBtnEl.disabled = false;
  return currentAbortController.signal;
}

function endInference(runBtnEl, runI18nKey, stopI18nKey) {
  inferenceLocked = false;
  unlockUiForInference();
  _setRunButtonRunning(runBtnEl, false, runI18nKey, stopI18nKey);
  currentAbortController = null;
}

function cancelInference() {
  if (currentAbortController) currentAbortController.abort();
}

function triggerDownload(url, filename) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// ---------------------------------------------------------------------------
// Pipeline run
// ---------------------------------------------------------------------------

runBtn.addEventListener("click", async () => {
  // The run button doubles as the stop button while inference is in flight.
  if (inferenceLocked) {
    cancelInference();
    return;
  }
  if (!selectedFile) return;
  statusEl.textContent = "processing…";
  statusEl.classList.add("busy"); statusEl.classList.remove("ok", "err");
  runInfo.textContent = "推論実行中…";

  const signal = beginInference(runBtn, "input.run", "input.stop");
  const form = new FormData();
  form.append("image", selectedFile);
  // Optional per-hand crops. The server runs the hand-only decoder on
  // each one and overrides the body's hand pose; absent fields are
  // ignored. Filenames are arbitrary — the API keys (left_hand_image /
  // right_hand_image) are what matter.
  try {
    const hands = window.__sam3dGetHandImages?.() || {};
    if (hands.left?.dataUrl) {
      const blob = window.__sam3dDataUrlToBlob?.(hands.left.dataUrl);
      if (blob) form.append("left_hand_image", blob, "left_hand.png");
    }
    if (hands.right?.dataUrl) {
      const blob = window.__sam3dDataUrlToBlob?.(hands.right.dataUrl);
      if (blob) form.append("right_hand_image", blob, "right_hand.png");
    }
  } catch (e) { console.warn("[SAM3D] attaching hand images failed:", e); }
  try {
    const r = await fetch("/sam3d/api/process", { method: "POST", body: form, signal });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const j = await r.json();
    currentJobId = j.job_id;
    tabJobIds.image = j.job_id;
    // Editor: cache the pose response so confirm-and-close has the
    // pose_json + image size to ship to the ComfyUI node.
    window.__sam3dPoseLastResponse = {
      pose: j.pose_json,
      image_size: { width: j.width, height: j.height },
    };
    refreshEditorPosePayload();
    runInfo.textContent = window.i18n?.t("input.done") || "推定終了";
    bodyPresetUnlocked = true;
    // Fresh pose estimation arrived → discard any prior pose adjustments
    // (rotation overrides + Hips translation). The previous bone tree was
    // tied to the OLD pose; carrying its rotations into the new one would
    // double-apply transforms and visually break.
    if (settings.pose_adjust) {
      settings.pose_adjust.rotation_overrides = {};
    }
    if (poseEditMode) {
      // Exit pose-edit cleanly — bones get rebuilt from the new skeleton
      // when the user re-enters.
      const toggleBtn = document.getElementById("pose-edit-toggle-btn");
      if (toggleBtn) toggleBtn.click();
    }
    viewer.clearHipsTranslation();
    if (window.__resetPoseUndoHistory) window.__resetPoseUndoHistory();
    // Inference landed — unlock the image-tab FBX download and re-render
    // the now-posed body with whatever body preset the user was viewing.
    exportFbxBtn.hidden = false;
    exportBvhBtn.hidden = false;
    const poseEditToggleBtn = document.getElementById("pose-edit-toggle-btn");
    if (poseEditToggleBtn) poseEditToggleBtn.hidden = false;
    scheduleRender();

    statusEl.textContent = `done — ${j.elapsed_sec}s`;
    statusEl.classList.add("ok"); statusEl.classList.remove("busy", "err");
  } catch (err) {
    if (err.name === "AbortError") {
      runInfo.textContent = window.i18n?.t("input.cancelled") || "キャンセルしました";
      statusEl.textContent = "cancelled";
      statusEl.classList.remove("busy", "ok", "err");
    } else {
      console.error(err);
      runInfo.textContent = `エラー: ${err.message || err}`;
      statusEl.textContent = "failed";
      statusEl.classList.add("err"); statusEl.classList.remove("busy", "ok");
    }
  } finally {
    endInference(runBtn, "input.run", "input.stop");
  }
});

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Preset Pack Admin (Phase 5)
// ---------------------------------------------------------------------------

const packSelect = $("pack-select");
const packSwitchBtn = $("pack-switch-btn");
const packNewName = $("pack-new-name");
const packCloneBtn = $("pack-clone-btn");
const packDeleteBtn = $("pack-delete-btn");
const packActiveLabel = $("pack-active-label");
const fbxStatusEl = $("fbx-status");
const fbxInput = $("fbx-input");
const fbxDrop = $("fbx-drop");
const fbxLabel = $("fbx-label");
const fbxRebuildBtn = $("fbx-rebuild-btn");
const fbxRebuildInfo = $("fbx-rebuild-info");

async function refreshPackList() {
  try {
    const r = await fetch("/sam3d/api/preset_packs");
    const j = await r.json();
    packSelect.innerHTML = "";
    for (const name of (j.packs || [])) {
      const opt = document.createElement("option");
      opt.value = name; opt.textContent = name;
      if (name === j.active) opt.textContent += "  (active)";
      packSelect.appendChild(opt);
    }
    packSelect.value = j.active;
    packActiveLabel.textContent = `active: ${j.active}`;
  } catch (e) { console.error(e); }
}

async function refreshFbxStatus() {
  try {
    const r = await fetch("/sam3d/api/fbx_status");
    const j = await r.json();
    const mtime = j.fbx_mtime ? new Date(j.fbx_mtime * 1000).toLocaleString() : "-";
    const size = j.fbx_size ? `${(j.fbx_size / (1024 * 1024)).toFixed(2)} MB` : "-";
    const staleHint = j.stale ? "  ⚠ npz stale, rebuild required" : "";
    fbxStatusEl.textContent = `FBX: ${size} (${mtime})${staleHint}`;
  } catch (_e) {}
}

packSwitchBtn.addEventListener("click", async () => {
  const name = packSelect.value;
  if (!name) return;
  try {
    const r = await fetch("/sam3d/api/preset_packs/active", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!r.ok) throw new Error(await r.text());
    await refreshPackList();
    // Slider schema (and blendshape list) depends on active pack — refresh.
    await refreshSliderSchemaPreservingValues();
  } catch (e) { console.error(e); packActiveLabel.textContent = `switch failed: ${e.message || e}`; }
});

packCloneBtn.addEventListener("click", async () => {
  const target = packNewName.value.trim();
  if (!target) { return; }
  try {
    const r = await fetch("/sam3d/api/preset_packs/clone", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ source: packSelect.value, target }),
    });
    if (!r.ok) throw new Error(await r.text());
    packNewName.value = "";
    await refreshPackList();
    packSelect.value = target;
  } catch (e) { packActiveLabel.textContent = `clone failed: ${e.message || e}`; }
});

packDeleteBtn.addEventListener("click", async () => {
  const name = packSelect.value;
  if (!name || !confirm(`Delete pack "${name}"? (cannot be undone)`)) return;
  try {
    const r = await fetch(`/sam3d/api/preset_packs/${encodeURIComponent(name)}`, { method: "DELETE" });
    if (!r.ok) throw new Error(await r.text());
    await refreshPackList();
    await refreshSliderSchemaPreservingValues();
  } catch (e) { packActiveLabel.textContent = `delete failed: ${e.message || e}`; }
});

fbxInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  fbxLabel.textContent = `uploading ${file.name}...`;
  const form = new FormData();
  form.append("fbx", file);
  try {
    const r = await fetch("/sam3d/api/fbx_upload", { method: "POST", body: form });
    if (!r.ok) throw new Error(await r.text());
    fbxLabel.textContent = `${file.name} ← uploaded, click Rebuild`;
    await refreshFbxStatus();
  } catch (err) {
    fbxLabel.textContent = `upload failed: ${err.message || err}`;
  }
});

["dragenter", "dragover"].forEach((ev) =>
  fbxDrop.addEventListener(ev, (e) => { e.preventDefault(); fbxDrop.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((ev) =>
  fbxDrop.addEventListener(ev, (e) => { e.preventDefault(); fbxDrop.classList.remove("dragover"); })
);
fbxDrop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files?.[0];
  if (f) { fbxInput.files = e.dataTransfer.files; fbxInput.dispatchEvent(new Event("change")); }
});

fbxRebuildBtn.addEventListener("click", async () => {
  fbxRebuildBtn.disabled = true;
  fbxRebuildInfo.textContent = "Running Blender (30-60s)…";
  try {
    const r = await fetch("/sam3d/api/rebuild_blendshapes", {
      method: "POST", headers: { "content-type": "application/json" }, body: "{}",
    });
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    fbxRebuildInfo.textContent =
      `OK: ${j.num_blendshapes} blendshapes, ${j.num_vertex_jsons} vertex JSONs (${j.elapsed_sec}s)`;
    await refreshFbxStatus();
    // Slider list may have grown / shrunk.
    await refreshSliderSchemaPreservingValues();
  } catch (e) {
    fbxRebuildInfo.textContent = `rebuild failed: ${e.message || e}`;
  } finally {
    fbxRebuildBtn.disabled = false;
  }
});

async function refreshSliderSchemaPreservingValues() {
  // Snapshot the in-memory values so shape-key list changes don't wipe user edits.
  const snapshot = JSON.parse(JSON.stringify(settings));
  try {
    const r = await fetch("/sam3d/api/slider_schema");
    schema = await r.json();
    buildSliders(bodyParamsEl, "body_params", schema.body_params);
    buildSliders(boneLengthsEl, "bone_lengths", schema.bone_lengths);
    buildSliders(blendshapesEl, "blendshapes", schema.blendshapes);
    applySettingsToUi(snapshot);
  } catch (e) { console.error("slider schema refresh failed:", e); }
  if (currentJobId) scheduleRender();
}

// ---------------------------------------------------------------------------
// Video → Animated FBX
// ---------------------------------------------------------------------------

const videoInput = $("video-input");
const videoDrop = $("video-drop");
const videoLabel = $("video-label");
const videoInfo = $("video-info");
const videoRunBtn = $("video-run-btn");
const videoRunInfo = $("video-run-info");
const videoDownloadBtn = $("video-download-btn");
const videoDownloadBvhBtn = $("video-download-bvh-btn");
const videoDownloadInfo = $("video-download-info");
const videoDownloadBvhInfo = $("video-download-bvh-info");
let selectedVideo = null;
// Motion cache id returned by /api/infer_motion. While set, body preset
// tweaks re-run /api/build_animated_fbx instead of the pose/image pipeline.
let currentMotionId = null;
// Most recent animated-FBX URL (for the Download button).
let currentAnimatedFbxUrl = null;
let videoRebuildInFlight = false;
let videoRebuildDirty = false;

function onVideoChosen(file) {
  if (!file) return;
  selectedVideo = file;
  videoLabel.textContent = file.name;
  videoRunBtn.disabled = false;
  videoInfo.textContent = `${(file.size / (1024 * 1024)).toFixed(1)} MB`;
  // Probe the video so the user sees fps / frame count before kicking off export.
  const form = new FormData();
  form.append("video", file);
  fetch("/sam3d/api/probe_video", { method: "POST", body: form })
    .then((r) => r.ok ? r.json() : null)
    .then((info) => {
      if (info) {
        videoInfo.textContent =
          `${(file.size / (1024 * 1024)).toFixed(1)} MB · ${info.width}x${info.height} · ${info.fps.toFixed(1)} fps · ${info.frame_count || "?"} frames · ${info.duration_sec.toFixed(1)} s`;
      }
    })
    .catch(() => {});
}

videoInput.addEventListener("change", (e) => onVideoChosen(e.target.files?.[0]));
["dragenter", "dragover"].forEach((ev) =>
  videoDrop.addEventListener(ev, (e) => {
    e.preventDefault(); videoDrop.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  videoDrop.addEventListener(ev, (e) => {
    e.preventDefault(); videoDrop.classList.remove("dragover");
  })
);
videoDrop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files?.[0];
  if (f) onVideoChosen(f);
});

// Phase 1: "モーションを推定" — runs only the slow segmentation + pose pass
// and caches the raw per-frame params. Phase 2 (FBX build) is triggered
// immediately afterward with the currently-selected body preset settings
// so the user sees the rig animate right away; subsequent preset/JSON
// changes call only Phase 2.
videoRunBtn.addEventListener("click", async () => {
  // Same toggle pattern as the image-tab run button: while inference is in
  // flight a click on the (now-red) button cancels.
  if (inferenceLocked) {
    cancelInference();
    return;
  }
  if (!selectedVideo) return;
  // Button is shown again only when rebuildAnimatedFbx finishes successfully.
  videoDownloadBtn.hidden = true;
  videoDownloadBvhBtn.hidden = true;
  videoRunInfo.textContent = "Running motion inference (minutes for long clips)…";
  const signal = beginInference(videoRunBtn, "video.run", "video.stop");
  const form = new FormData();
  form.append("video", selectedVideo);
  form.append("fps", $("video-fps").value || "0");
  const maxF = Number($("video-max-frames").value || "0");
  if (maxF > 0) form.append("max_frames", String(maxF));
  form.append("stride", $("video-stride").value || "1");
  form.append("bbox_threshold", $("video-bbox-thr").value || "0.8");
  try {
    const t0 = performance.now();
    const r = await fetch("/sam3d/api/infer_motion", { method: "POST", body: form, signal });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const j = await r.json();
    currentMotionId = j.motion_id;
    videoRunInfo.textContent = window.i18n?.t("video.done") || "推定終了";
    // Motion is cached — immediately bake and loop the animation on top of
    // the body the user was already viewing (default MHR body before
    // any preset change, or their chosen preset / uploaded JSON).
    await rebuildAnimatedFbx(signal);
  } catch (e) {
    if (e.name === "AbortError") {
      videoRunInfo.textContent = window.i18n?.t("video.cancelled") || "キャンセルしました";
    } else {
      console.error(e);
      videoRunInfo.textContent = `video FBX error: ${e.message || e}`;
    }
  } finally {
    endInference(videoRunBtn, "video.run", "video.stop");
  }
});

// Phase 2: (re)build the animated FBX for the cached motion using the
// current Body Preset settings. Called after motion inference completes and
// again whenever the user swaps preset/JSON while on the video tab. An
// optional AbortSignal plumbs the main-run cancel button through to Phase 2.
async function rebuildAnimatedFbx(signal = null) {
  if (!currentMotionId) return;
  if (videoRebuildInFlight) { videoRebuildDirty = true; return; }
  videoRebuildInFlight = true;
  videoRebuildDirty = false;
  const label0 = videoRunInfo.textContent;
  // Visible feedback so slider drags on video don't look like a no-op while
  // the Blender subprocess grinds (~10-30s per rebuild).
  videoRunInfo.textContent =
    window.i18n?.t("video.rebuilding") || "rebuilding FBX…";
  // Clear the viewport so the stale body doesn't keep animating on top
  // of the "rebuilding" status. Re-populated when the new FBX lands.
  viewer.clearMesh();
  _setVideoBusy(true);
  try {
    const t0 = performance.now();
    const payload = {
      motion_id: currentMotionId,
      // Always bake the VIDEO tab's cached body preset — not whichever tab
      // happens to be active when the recursion in `finally` fires.
      settings: tabSettings.video,
      root_motion_mode: $("video-root-mode").value,
    };
    console.debug("rebuildAnimatedFbx →",
      "lean=", tabSettings.video?.pose_adjust?.lean_correction);
    const r = await fetch("/sam3d/api/build_animated_fbx", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const j = await r.json();
    // A newer settings change (e.g. preset Load) was queued while this
    // rebuild was in flight — skip swapping the stale result into the
    // viewer so the user doesn't see a flash of the default body before
    // the queued preset-aware rebuild catches up.
    if (videoRebuildDirty) return;
    currentAnimatedFbxUrl = j.fbx_url;
    videoDownloadBtn.hidden = false;
    videoDownloadBvhBtn.hidden = false;
    const ms = Math.round(performance.now() - t0);
    const name = j.fbx_url.split("/").pop();
    videoRunInfo.innerHTML =
      `motion ${j.num_frames} frames · FBX build ${Math.round(ms/1000)}s: ${name}`;
    try {
      await viewer.loadFbxAnimated(j.fbx_url, `video-${Date.now()}`);
      renderAvailable = true;
      const savePngBtn = document.getElementById("save-render-btn");
      if (savePngBtn) savePngBtn.disabled = false;
      if (overlay.textContent) overlay.textContent = "";
    } catch (err) {
      console.warn("FBX playback failed:", err);
    }
  } catch (e) {
    if (e.name === "AbortError") {
      // Propagate — the videoRunBtn handler reports the cancellation label.
      throw e;
    }
    console.error(e);
    videoRunInfo.textContent = `FBX build error: ${e.message || e}`;
  } finally {
    videoRebuildInFlight = false;
    if (videoRebuildDirty) {
      // Still busy — next rebuild takes over without unlocking the UI.
      rebuildAnimatedFbx();
    } else {
      _setVideoBusy(false);
    }
  }
}

// Lock / unlock sidebar + tab nav while a Blender FBX rebuild is running,
// and show a centred "rebuilding…" overlay so the user can't miss it.
function _setVideoBusy(busy) {
  document.body.classList.toggle("ui-busy", busy);
  if (busy) {
    overlay.textContent = window.i18n?.t("video.rebuilding") || "rebuilding FBX…";
    overlay.classList.add("busy");
  } else {
    overlay.textContent = "";
    overlay.classList.remove("busy");
  }
}

// "FBX をダウンロード" — save the currently-playing animation to disk.
// The server re-uses tmp/animated.fbx so we tag the download with a
// timestamped filename; browsers honour the `download` attribute.
videoDownloadBtn.addEventListener("click", () => {
  if (!currentAnimatedFbxUrl) return;
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const savedAs = `body3d_animated_${stamp}.fbx`;
  triggerDownload(currentAnimatedFbxUrl, savedAs);
  if (videoDownloadInfo) videoDownloadInfo.textContent = `saved as ${savedAs}`;
});

videoDownloadBvhBtn?.addEventListener("click", async () => {
  if (!currentMotionId) return;
  videoDownloadBvhBtn.disabled = true;
  if (videoDownloadBvhInfo) {
    videoDownloadBvhInfo.textContent = "Blender subprocess running... (converting to BVH)";
  }
  try {
    const t0 = performance.now();
    const r = await fetch("/sam3d/api/build_animated_bvh", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        motion_id: currentMotionId,
        settings: tabSettings.video,
        root_motion_mode: $("video-root-mode").value,
        strength: 1.0,
      }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const j = await r.json();
    const ms = Math.round(performance.now() - t0);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const savedAs = `body3d_animated_${stamp}.bvh`;
    triggerDownload(j.bvh_url, savedAs);
    if (videoDownloadBvhInfo) {
      videoDownloadBvhInfo.textContent =
        `BVH ${j.elapsed_sec}s (wall ${ms}ms) - saved as ${savedAs}`;
    }
  } catch (e) {
    console.error(e);
    if (videoDownloadBvhInfo) {
      videoDownloadBvhInfo.textContent = `BVH export error: ${e.message || e}`;
    }
  } finally {
    videoDownloadBvhBtn.disabled = false;
  }
});

// ---------------------------------------------------------------------------

// Wire the PNG save button on the 3D viewer overlay. Stays disabled until a
// mesh has been rendered at least once (otherwise you get a blank viewport PNG).
document.getElementById("save-render-btn")?.addEventListener("click", async () => {
  if (!renderAvailable) return;
  const base = currentJobId ? `render_${currentJobId}` : `render_${Date.now()}`;
  try {
    await viewer.savePng(`${base}.png`);
  } catch (e) {
    console.error("save PNG failed:", e);
  }
});

// Body Preset tab: download the current slider values as a standalone
// body preset JSON (same schema as body_preset_settings/*.json, drop-in
// compatible with Preset Pack's Load menu).
document.getElementById("make-download-btn")?.addEventListener("click", () => {
  // Download the make tab's cached parameters explicitly — settings still
  // points there while the make tab is active, but pinning the slot is
  // safer if this ever runs from another callsite.
  const payload = JSON.stringify(tabSettings.make, null, 2);
  const blob = new Blob([payload], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const a = document.createElement("a");
  a.href = url;
  a.download = `body_preset_${stamp}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5_000);
  const info = document.getElementById("make-download-info");
  if (info) info.textContent = window.i18n?.t("make.downloaded") || "JSON saved";
});

async function boot() {
  // Pull feature flags first so the tab nav can reflect the server's config.ini.
  try {
    const r = await fetch("/sam3d/api/features");
    if (r.ok) {
      const f = await r.json();
      featureFlags.preset_pack_admin = !!f.preset_pack_admin;
      featureFlags.debug = !!f.debug;
    }
  } catch (_e) {}
  // Hide the Preset Admin tab button entirely when the flag is off.
  const adminBtn = document.querySelector('.tab-nav button[data-tab="admin"]');
  if (adminBtn) {
    adminBtn.hidden = !featureFlags.preset_pack_admin;
    if (!featureFlags.preset_pack_admin) {
      adminBtn.title = "Disabled in config.ini ([features] preset_pack_admin)";
    }
  }
  // Health panel is a debug-only surface. Drop health polling entirely when
  // it isn't visible so we don't stream CUDA info to an invisible <pre>.
  const healthSection = document.getElementById("health-section");
  if (healthSection) healthSection.hidden = !featureFlags.debug;
  initTabs();
  try {
    const r = await fetch("/sam3d/api/slider_schema");
    schema = await r.json();
    buildSliders(bodyParamsEl, "body_params", schema.body_params);
    buildSliders(boneLengthsEl, "bone_lengths", schema.bone_lengths);
    buildSliders(blendshapesEl, "blendshapes", schema.blendshapes);
  } catch (e) { console.error("slider schema load failed:", e); }
  initLeanSliders();
  initPoseEditUi();
  await refreshPresets();
  await refreshPackList();
  await refreshFbxStatus();
  // Still poll in debug mode so the Health panel reflects the live state;
  // in non-debug mode we skip polling to keep the network quiet.
  if (featureFlags.debug) {
    pollHealth();
    setInterval(pollHealth, 15_000);
  }
}

boot();

// ---------------------------------------------------------------------------
// Editor confirm-payload glue.
//
// editor_init.js reads ``window.__sam3dPosePayload`` /
// ``window.__sam3dBodyPresetPayload`` when the user clicks "Confirm & Close".
// Snapshot live state from app.js's globals here so we don't have to
// thread an extra channel through the original module.
// ---------------------------------------------------------------------------
function refreshEditorPosePayload() {
  const last = window.__sam3dPoseLastResponse;
  if (!last || !last.pose) {
    window.__sam3dPosePayload = null;
    return;
  }
  const p = last.pose;
  const imageSlot = tabSettings.image || {};
  const adjust = imageSlot.pose_adjust || {};
  // FLAT layout — key-compatible with SAM3DBodyProcessToJson's output
  // so SAM3DBodyRenderFromPoseAndBodyPresetJson reads the editor's pose
  // exactly the same way as the legacy in-graph pose node.
  // Two field renames vs. /sam3d/api/process's response:
  //   pred_cam_t        -> camera
  //   pred_keypoints_3d -> keypoints_3d
  window.__sam3dPosePayload = {
    body_pose_params: p.body_pose_params ?? null,
    hand_pose_params: p.hand_pose_params ?? null,
    global_rot:       p.global_rot ?? null,
    camera:           p.pred_cam_t ?? null,
    focal_length:     p.focal_length ?? null,
    bbox:             p.bbox ?? null,
    keypoints_3d:     p.pred_keypoints_3d ?? null,
    shape_params:     p.shape_params ?? null,
    scale_params:     p.scale_params ?? null,
    expr_params:      p.expr_params ?? null,
    image_size:       last.image_size || null,
    // Editor-only extras (Render node currently ignores these; downstream
    // pose-aware nodes can pick them up).
    pose_adjust: {
      lean_correction: typeof adjust.lean_correction === "number"
        ? adjust.lean_correction : 0.0,
      rotation_overrides: adjust.rotation_overrides
        ? JSON.parse(JSON.stringify(adjust.rotation_overrides))
        : {},
    },
  };
}

function refreshEditorBodyPresetPayload() {
  const m = tabSettings.make || {};
  window.__sam3dBodyPresetPayload = {
    body_params:  { ...(m.body_params  || {}) },
    bone_lengths: { ...(m.bone_lengths || {}) },
    blendshapes:  { ...(m.blendshapes  || {}) },
  };
}

// First snapshot once boot() has had a chance to populate tabSettings.
setTimeout(() => {
  refreshEditorPosePayload();
  refreshEditorBodyPresetPayload();
}, 100);

// ---------------------------------------------------------------------------
// Pose-editor image-output wiring (画像範囲 / 背景色指定 / pose_image &
// input_image capture for the SAM3DBodyPoseEditor node).
//
// Lives at the tail of editor_core.js so all module-level state (viewer,
// selectedFile, refreshEditorPosePayload, ...) is already in scope. Every
// hook checks for the existence of its DOM element so body_preset_editor.html
// — which doesn't expose these controls — silently no-ops.
// ---------------------------------------------------------------------------
(function initPoseImageRangeUi() {
  // Persistent capture state. Range rect is in canvas BACKING-store pixels
  // (devicePixelRatio applied) so captureImage can crop directly without
  // re-mapping. ``null`` = no range = capture full viewport on confirm.
  const state = {
    bgColor: "#ffffff",
    rangeMode: false,
    rect: null,         // {x, y, width, height} in backing-store px
    inputImage: null,   // {dataUrl, width, height}
    handLImage: null,   // {dataUrl, width, height} — set by left-hand drop UI
    handRImage: null,   // same for the right hand
  };
  window.__sam3dPoseImageState = state;

  // Hook the file-input handler so input_image always reflects the latest
  // upload. We can't easily monkey-patch onFileChosen (declared with const
  // in the closure above), but the file input fires a "change" event we can
  // listen to in addition to the existing handler.
  const fileInputEl = document.getElementById("file-input");
  function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = () => reject(r.error || new Error("file read failed"));
      r.readAsDataURL(file);
    });
  }
  async function captureInputImage(file) {
    if (!file) { state.inputImage = null; return; }
    try {
      const dataUrl = await readFileAsDataUrl(file);
      // Decode dimensions for downstream ergonomics.
      const dim = await new Promise((resolve) => {
        const im = new Image();
        im.onload = () => resolve({ w: im.naturalWidth, h: im.naturalHeight });
        im.onerror = () => resolve({ w: 0, h: 0 });
        im.src = dataUrl;
      });
      state.inputImage = { dataUrl, width: dim.w, height: dim.h };
    } catch (e) {
      console.warn("[SAM3D] input image capture failed:", e);
      state.inputImage = null;
    }
  }
  if (fileInputEl) {
    fileInputEl.addEventListener("change", (e) => {
      const f = e.target.files?.[0];
      if (f) captureInputImage(f);
    });
  }
  const fileDropEl = document.getElementById("file-drop");
  if (fileDropEl) {
    fileDropEl.addEventListener("drop", (e) => {
      const f = e.dataTransfer?.files?.[0];
      if (f) captureInputImage(f);
    });
  }

  // ---- Hand image inputs (Left / Right) ---------------------------------
  // Each hand has a drop area + 左右反転 button. The dataUrl stored here is
  // POSTed alongside the main image to /sam3d/api/process so the server can
  // run the hand decoder and override the body's hand pose. The same dataUrl
  // is also returned in the confirm payload so the Pose Editor node can
  // surface it as ``hand_l_image`` / ``hand_r_image`` IMAGE outputs.
  function _decodeFileToImage(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => {
        const url = r.result;
        const im = new Image();
        im.onload = () => resolve({ dataUrl: url, width: im.naturalWidth, height: im.naturalHeight });
        im.onerror = () => reject(new Error("hand image decode failed"));
        im.src = url;
      };
      r.onerror = () => reject(r.error || new Error("file read failed"));
      r.readAsDataURL(file);
    });
  }

  function _renderHandPreview(side, cap) {
    const previewEl = document.getElementById(`${side}hand-preview`);
    const labelEl = document.getElementById(`${side}hand-file-label`);
    const flipBtn = document.getElementById(`${side}hand-flip-btn`);
    if (cap) {
      if (previewEl) {
        previewEl.src = cap.dataUrl;
        previewEl.hidden = false;
      }
      if (labelEl) labelEl.textContent = `${cap.width}×${cap.height}`;
      if (flipBtn) flipBtn.disabled = false;
    } else {
      if (previewEl) {
        previewEl.removeAttribute("src");
        previewEl.hidden = true;
      }
      if (labelEl) {
        const key = side === "l" ? "input.lhand_drop" : "input.rhand_drop";
        labelEl.textContent =
          window.i18n?.t(key) || (side === "l" ? "Left hand image" : "Right hand image");
      }
      if (flipBtn) flipBtn.disabled = true;
    }
  }

  async function _handleHandFile(side, file) {
    if (!file) return;
    try {
      const cap = await _decodeFileToImage(file);
      if (side === "l") state.handLImage = cap; else state.handRImage = cap;
      _renderHandPreview(side, cap);
    } catch (exc) {
      console.warn(`[SAM3D] ${side}hand load failed:`, exc);
    }
  }

  // Mirror the loaded hand image in place. We re-decode through a canvas so
  // the dataUrl actually contains pixel-flipped bytes — server-side hand
  // inference doesn't see CSS transforms.
  function _mirrorDataUrl(dataUrl) {
    return new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => {
        const off = document.createElement("canvas");
        off.width = im.naturalWidth;
        off.height = im.naturalHeight;
        const ctx = off.getContext("2d");
        ctx.translate(off.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(im, 0, 0);
        resolve({
          dataUrl: off.toDataURL("image/png"),
          width: off.width,
          height: off.height,
        });
      };
      im.onerror = () => reject(new Error("mirror decode failed"));
      im.src = dataUrl;
    });
  }

  async function _flipHand(side) {
    const cur = side === "l" ? state.handLImage : state.handRImage;
    if (!cur) return;
    try {
      const flipped = await _mirrorDataUrl(cur.dataUrl);
      if (side === "l") state.handLImage = flipped; else state.handRImage = flipped;
      _renderHandPreview(side, flipped);
    } catch (exc) {
      console.warn(`[SAM3D] ${side}hand flip failed:`, exc);
    }
  }

  for (const side of ["l", "r"]) {
    const inputEl = document.getElementById(`${side}hand-file-input`);
    const dropEl  = document.getElementById(`${side}hand-drop`);
    const flipBtn = document.getElementById(`${side}hand-flip-btn`);
    if (inputEl) {
      inputEl.addEventListener("change", (e) => {
        const f = e.target.files?.[0];
        if (f) _handleHandFile(side, f);
      });
    }
    if (dropEl) {
      ["dragenter", "dragover"].forEach((ev) =>
        dropEl.addEventListener(ev, (e) => {
          e.preventDefault();
          dropEl.classList.add("dragover");
        }),
      );
      ["dragleave", "drop"].forEach((ev) =>
        dropEl.addEventListener(ev, (e) => {
          e.preventDefault();
          dropEl.classList.remove("dragover");
        }),
      );
      dropEl.addEventListener("drop", (e) => {
        const f = e.dataTransfer?.files?.[0];
        if (f) _handleHandFile(side, f);
      });
    }
    if (flipBtn) {
      flipBtn.addEventListener("click", () => _flipHand(side));
    }
  }

  // Used by the run handler (multipart POST) and the confirm payload glue
  // (postMessage to ComfyUI).
  window.__sam3dGetHandImages = function () {
    return { left: state.handLImage, right: state.handRImage };
  };
  // Convert a "data:image/...;base64,xxx" URL to a Blob so it can be
  // appended to a FormData. Returns null when the dataUrl is missing /
  // malformed so the caller can skip the field.
  window.__sam3dDataUrlToBlob = function (dataUrl) {
    if (!dataUrl || typeof dataUrl !== "string") return null;
    const m = dataUrl.match(/^data:([^;]+);base64,(.+)$/);
    if (!m) return null;
    try {
      const bytes = atob(m[2]);
      const buf = new Uint8Array(bytes.length);
      for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i);
      return new Blob([buf], { type: m[1] });
    } catch (_e) {
      return null;
    }
  };

  // ---- Background-color picker -------------------------------------------
  const bgBtn = document.getElementById("bg-color-btn");
  const bgInput = document.getElementById("bg-color-input");
  const bgSwatch = document.getElementById("bg-color-swatch");
  function setBgColor(hex) {
    state.bgColor = hex;
    if (bgSwatch) bgSwatch.style.background = hex;
    if (bgInput && bgInput.value !== hex) bgInput.value = hex;
  }
  setBgColor(state.bgColor);
  if (bgBtn && bgInput) {
    bgBtn.addEventListener("click", (ev) => {
      // Avoid recursive open when clicking the input itself bubbles up.
      if (ev.target === bgInput) return;
      bgInput.click();
    });
    bgInput.addEventListener("input", () => setBgColor(bgInput.value));
    bgInput.addEventListener("change", () => setBgColor(bgInput.value));
  }

  // ---- Range-select overlay ---------------------------------------------
  const rangeBtn = document.getElementById("range-select-btn");
  const rangeResetBtn = document.getElementById("range-reset-btn");
  const rangeOverlay = document.getElementById("range-overlay");
  const rangeRect = document.getElementById("range-rect");
  const viewportEl = document.getElementById("viewport");

  function clientRectToBacking(rectClient) {
    if (!viewer.getCanvasSize || !viewer.getCanvasClientSize) return null;
    const back = viewer.getCanvasSize();
    const front = viewer.getCanvasClientSize();
    const sx = back.width  / Math.max(front.width,  1);
    const sy = back.height / Math.max(front.height, 1);
    return {
      x: rectClient.x * sx,
      y: rectClient.y * sy,
      width: rectClient.width  * sx,
      height: rectClient.height * sy,
    };
  }

  function updateRangeButtonLabel() {
    if (rangeBtn) {
      const key = state.rangeMode ? "viewport.range_exit" : "viewport.range_enter";
      rangeBtn.dataset.i18n = key;
      rangeBtn.textContent = window.i18n?.t(key) ||
        (state.rangeMode ? "決定" : "画像範囲");
    }
    if (rangeResetBtn) {
      // Reset is visible whenever a saved range exists OR mode is on (so
      // the user can wipe a half-drawn rect mid-mode). Disabled state is
      // kept consistent with the saved rect availability.
      rangeResetBtn.hidden = !(state.rect || state.rangeMode);
      rangeResetBtn.disabled = !state.rect;
    }
  }

  // Hide the red rect from the viewport AND clear ``state.rect``.
  // Used by the reset button so the next confirm captures the full
  // viewport again.
  function hideRectVisual() {
    if (rangeRect) rangeRect.hidden = true;
  }

  // Re-draw the saved client rect (if any) in the overlay. Called both
  // on mode entry (so the user sees what they previously confirmed) AND
  // on mode exit (so the saved rect remains visible as a red marker —
  // mirrors the Pose Editor + behaviour).
  function redrawSavedRect() {
    if (!rangeRect) return;
    if (!state.rectClient) {
      rangeRect.hidden = true;
      return;
    }
    const c = state.rectClient;
    rangeRect.hidden = false;
    rangeRect.style.left   = `${c.x}px`;
    rangeRect.style.top    = `${c.y}px`;
    rangeRect.style.width  = `${c.width}px`;
    rangeRect.style.height = `${c.height}px`;
  }

  function setRangeMode(on) {
    state.rangeMode = !!on;
    document.body.classList.toggle("range-select-mode", state.rangeMode);
    // Always reflect the saved rect (if any) on screen — both during
    // range-select mode (so the user sees their previous rect to refine)
    // and after exiting (so the captured area stays visually marked
    // until the user explicitly resets it).
    redrawSavedRect();
    updateRangeButtonLabel();
  }

  function showRect(client) {
    if (!rangeRect || !viewportEl) return;
    const vp = viewportEl.getBoundingClientRect();
    rangeRect.hidden = false;
    rangeRect.style.left   = `${client.x}px`;
    rangeRect.style.top    = `${client.y}px`;
    rangeRect.style.width  = `${client.width}px`;
    rangeRect.style.height = `${client.height}px`;
    // Also stash the BACKING-store rect for capture time.
    state.rect = clientRectToBacking(client);
    state.rectClient = { ...client };
    updateRangeButtonLabel();
    void vp;  // (vp unused — kept for reference; client coords are overlay-relative)
  }

  // Drag-to-define logic. Pointer events fire only when body has the
  // ``range-select-mode`` class (CSS toggles overlay pointer-events).
  let dragging = false;
  let startX = 0, startY = 0;
  if (rangeOverlay) {
    rangeOverlay.addEventListener("pointerdown", (ev) => {
      if (!state.rangeMode) return;
      const r = rangeOverlay.getBoundingClientRect();
      startX = ev.clientX - r.left;
      startY = ev.clientY - r.top;
      dragging = true;
      try { rangeOverlay.setPointerCapture(ev.pointerId); } catch (_e) {}
      showRect({ x: startX, y: startY, width: 1, height: 1 });
      ev.preventDefault();
    });
    rangeOverlay.addEventListener("pointermove", (ev) => {
      if (!dragging) return;
      const r = rangeOverlay.getBoundingClientRect();
      const cx = Math.max(0, Math.min(r.width,  ev.clientX - r.left));
      const cy = Math.max(0, Math.min(r.height, ev.clientY - r.top));
      const x = Math.min(startX, cx);
      const y = Math.min(startY, cy);
      const w = Math.abs(cx - startX);
      const h = Math.abs(cy - startY);
      showRect({ x, y, width: w, height: h });
    });
    const endDrag = (ev) => {
      if (!dragging) return;
      dragging = false;
      try { rangeOverlay.releasePointerCapture(ev.pointerId); } catch (_e) {}
      // Drop ranges below a few pixels — likely an accidental tap.
      if (state.rect && (state.rect.width < 4 || state.rect.height < 4)) {
        state.rect = null;
        state.rectClient = null;
        if (rangeRect) rangeRect.hidden = true;
      }
      updateRangeButtonLabel();
    };
    rangeOverlay.addEventListener("pointerup", endDrag);
    rangeOverlay.addEventListener("pointercancel", endDrag);
  }

  if (rangeBtn) {
    rangeBtn.addEventListener("click", () => setRangeMode(!state.rangeMode));
    updateRangeButtonLabel();
    document.addEventListener("body3d:langchange", updateRangeButtonLabel);
  }

  if (rangeResetBtn) {
    rangeResetBtn.addEventListener("click", () => {
      // Drop the saved rect AND the on-screen indicator so the next
      // confirm captures the full viewport again.
      state.rect = null;
      state.rectClient = null;
      hideRectVisual();
      updateRangeButtonLabel();
    });
    // Initial state: nothing saved, not in mode → hide entirely.
    rangeResetBtn.hidden = true;
    rangeResetBtn.disabled = true;
  }

  // Public capture entry-point used by the confirm-and-close payload glue.
  window.__sam3dCapturePoseImage = function () {
    if (!viewer || !viewer.captureImage) return null;
    if (!renderAvailable) return null;
    const bg = state.bgColor || "#ffffff";
    const cropRect = state.rect || null;
    const cap = viewer.captureImage({
      bgColor: new THREE.Color(bg).getHex(),
      cropRect,
      framing: "current",
    });
    return cap;
  };
  window.__sam3dGetInputImage = function () {
    return state.inputImage;
  };
  window.__sam3dCaptureBodyPresetImage = function () {
    if (!viewer || !viewer.captureImage) return null;
    const cap = viewer.captureImage({
      bgColor: 0xffffff,
      cropRect: null,
      framing: "fit_full_body",
    });
    return cap;
  };
})();

// ---------------------------------------------------------------------------
// Body-anchor right-toolbar wiring (translate / rotate gizmo, resets, image
// preview overlay). Hidden while pose-edit (IK) mode is active so the
// gizmo binding can switch over to bones without conflicting controls.
// ---------------------------------------------------------------------------
(function initBodyAnchorToolbar() {
  const tb       = document.querySelector(".viewport-toolbar");
  const trBtn    = document.getElementById("gizmo-translate-btn");
  const rotBtn   = document.getElementById("gizmo-rotate-btn");
  const rstTr    = document.getElementById("reset-translate-btn");
  const rstRot   = document.getElementById("reset-rotate-btn");
  const imgBtn   = document.getElementById("toggle-img-btn");
  const overlay  = document.getElementById("image-display-overlay");
  const overlayImg = document.getElementById("image-display-img");
  const toggleBtn = document.getElementById("pose-edit-toggle-btn");
  if (!tb || !trBtn || !rotBtn) return;

  let _gizmoMode = "translate";
  function setGizmoMode(mode) {
    _gizmoMode = mode === "rotate" ? "rotate" : "translate";
    if (viewer.setGizmoMode) viewer.setGizmoMode(_gizmoMode);
    trBtn.classList.toggle("active",  _gizmoMode === "translate");
    rotBtn.classList.toggle("active", _gizmoMode === "rotate");
  }

  trBtn.addEventListener("click", () => setGizmoMode("translate"));
  rotBtn.addEventListener("click", () => setGizmoMode("rotate"));
  if (rstTr)  rstTr.addEventListener("click",  () => viewer.resetBodyTranslate?.());
  if (rstRot) rstRot.addEventListener("click", () => viewer.resetBodyRotate?.());

  // Input-image overlay toggle. Reads ``state.inputImage.dataUrl`` from
  // the pose-image module's shared state so the thumbnail mirrors the
  // currently-loaded source picture.
  let _imgOverlayOn = false;
  function _refreshOverlayContent() {
    if (!overlay || !overlayImg) return;
    const cap = window.__sam3dGetInputImage?.();
    if (cap && cap.dataUrl) {
      overlayImg.src = cap.dataUrl;
      overlay.classList.toggle("visible", _imgOverlayOn);
    } else {
      overlay.classList.remove("visible");
    }
  }
  if (imgBtn && overlay) {
    imgBtn.addEventListener("click", () => {
      _imgOverlayOn = !_imgOverlayOn;
      imgBtn.classList.toggle("active", _imgOverlayOn);
      _refreshOverlayContent();
    });
  }
  // The shared input-image state is updated asynchronously when the user
  // drops or selects a file — refresh the thumbnail src each time the
  // window regains focus (cheap heuristic) and on a slow timer so the
  // overlay stays in sync without a dedicated callback.
  setInterval(_refreshOverlayContent, 1000);

  // Hide the toolbar (and overlay) while pose-edit mode is active. The
  // pose-edit toggle button is the only entrypoint to the mode in the
  // single-person editor; ``initPoseEditUi`` dispatches the
  // ``sam3d-pose-edit-changed`` CustomEvent right after flipping the
  // state, so we just listen and refresh.
  function _refreshToolbarVisibility() {
    const ikActive = !!viewer.isPoseEditActive?.();
    tb.style.display = ikActive ? "none" : "flex";
    if (overlay && ikActive) overlay.classList.remove("visible");
  }
  document.addEventListener("sam3d-pose-edit-changed", _refreshToolbarVisibility);
  _refreshToolbarVisibility();
})();

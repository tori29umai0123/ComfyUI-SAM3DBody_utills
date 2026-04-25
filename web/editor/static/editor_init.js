// editor_init.js — runs *before* editor_core.js to set up the editor
// shell (popup-vs-standalone detection, confirm-and-close handling,
// ComfyUI-side postMessage, and graceful no-op stubs for endpoints we
// did NOT port from the standalone app).
//
// The two HTML pages (pose_editor.html / character_editor.html) each
// set ``window.SAM3D_EDITOR_MODE = "pose" | "character"`` *before* the
// script tags load, so this module can branch on it.

(() => {
  const VERSION = "editor_init v3 (tab-nav + diag)";
  console.log("[SAM3D editor_init]", VERSION, "loading...");
  const MODE = (window.SAM3D_EDITOR_MODE || "pose").toLowerCase();
  console.log("[SAM3D editor_init] MODE =", MODE);
  const CONFIRM_TYPE_BY_MODE = {
    pose:      "sam3d-pose-confirmed",
    character: "sam3d-chara-confirmed",
  };
  const PAYLOAD_KEY_BY_MODE = {
    pose:      "pose_json",
    character: "chara_json",
  };

  const params = new URLSearchParams(location.search);
  const NODE_ID = params.get("node_id");
  window.SAM3D_NODE_ID = NODE_ID;

  // Force the initial tab. editor_core.js reads localStorage("body3d.tab")
  // on boot — for the editor we always want to land on the mode's tab and
  // never restore an image-tab session into the character editor (or vice
  // versa).
  const INITIAL_TAB = MODE === "character" ? "make" : "image";
  try {
    localStorage.setItem("body3d.tab", INITIAL_TAB);
  } catch (_e) {}

  // editor_core.js was written for the multi-tab standalone app — it
  // queries ``.tab-nav button.active`` in 6+ places to decide which tab
  // owns a render response, whether to re-fire after a slider change,
  // etc. Without a real tab-nav those queries return undefined and the
  // app silently loses every "this update belongs to tab X" decision
  // (skeletons aren't stashed → pose-adjust toggle bails out, lean
  // slider's maybeFire never reaches scheduleRender, etc.).
  //
  // Inject a hidden tab-nav with the four canonical tab buttons here,
  // pre-marking the editor's mode tab as ``.active`` so the queries land.
  // Buttons stay ``hidden`` + ``disabled`` so the user can't sneak into
  // a tab the editor doesn't actually expose.
  function injectHiddenTabNav() {
    if (document.querySelector(".tab-nav")) return;
    const nav = document.createElement("nav");
    nav.className = "tab-nav";
    nav.hidden = true;
    nav.style.display = "none";
    for (const name of ["image", "video", "make", "admin"]) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.dataset.tab = name;
      btn.disabled = true;
      btn.hidden = true;
      btn.style.display = "none";
      if (name === INITIAL_TAB) btn.classList.add("active");
      nav.appendChild(btn);
    }
    (document.body || document.documentElement).appendChild(nav);
  }
  if (document.body) {
    injectHiddenTabNav();
  } else {
    document.addEventListener("DOMContentLoaded", injectHiddenTabNav, { once: true });
  }

  // Diagnostic: confirm the injection landed and the active button is the
  // mode tab. If the user reports "slider doesn't move", checking for these
  // log lines is the fastest way to tell stale-cache from a real bug.
  function _diagnose() {
    const nav = document.querySelector(".tab-nav");
    const active = document.querySelector(".tab-nav button.active");
    console.log("[SAM3D editor_init] tab-nav present:", !!nav,
                "buttons:", nav?.children.length ?? 0,
                "active tab:", active?.dataset.tab);
    const lean = document.getElementById("img-lean-range");
    const num  = document.getElementById("img-lean-num");
    console.log("[SAM3D editor_init] lean slider/num:", !!lean, !!num,
                "disabled:", lean?.disabled, num?.disabled);
    console.log("[SAM3D editor_init] body.pose-edit-mode:",
                document.body.classList.contains("pose-edit-mode"));
    // Attach a passive sniffer so we can SEE the event in DevTools even if
    // editor_core.js's own listener is somehow not bound.
    if (lean) {
      lean.addEventListener("input", () => {
        console.log("[SAM3D editor_init] lean input fired:", lean.value);
      });
    }
    const toggleBtn = document.getElementById("pose-edit-toggle-btn");
    console.log("[SAM3D editor_init] pose-edit-toggle-btn present:", !!toggleBtn,
                "hidden:", toggleBtn?.hidden, "disabled:", toggleBtn?.disabled);
    if (toggleBtn) {
      toggleBtn.addEventListener("click", () => {
        console.log("[SAM3D editor_init] pose-edit-toggle clicked.",
                    "skel for image:",
                    !!(window.tabHumanoidSkeleton?.image?.bones?.length));
      });
    }
  }
  if (document.readyState !== "loading") {
    setTimeout(_diagnose, 800);
  } else {
    document.addEventListener("DOMContentLoaded",
      () => setTimeout(_diagnose, 800), { once: true });
  }

  // ---------------------------------------------------------------------
  // 1. Stub endpoints we didn't port from the standalone app so that
  //    refreshPackList / refreshFbxStatus / pollHealth fail "gracefully"
  //    (returning {} or 404) instead of spamming the network panel with
  //    aborted requests. We keep the implemented endpoints untouched.
  // ---------------------------------------------------------------------
  const PORTED = new Set([
    "/sam3d/api/ping",
    "/sam3d/api/process",
    "/sam3d/api/render",
    "/sam3d/api/presets",
    "/sam3d/api/preset/",        // prefix
    "/sam3d/api/slider_schema",
    "/sam3d/api/active_pack",
  ]);

  const isPorted = (url) => {
    for (const p of PORTED) {
      if (p.endsWith("/")) {
        if (url.startsWith(p)) return true;
      } else if (url === p || url.startsWith(p + "?")) {
        return true;
      }
    }
    return false;
  };

  const origFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    let url = "";
    try {
      url = typeof input === "string" ? input : input?.url || "";
    } catch (_e) {}
    // Only intercept calls heading to /sam3d/api/* that we know don't exist.
    if (url.startsWith("/sam3d/api/") && !isPorted(url)) {
      // Return a 404-shaped response so .ok checks are false and the
      // caller's catch / fallback path runs without a console error.
      return Promise.resolve(new Response(
        JSON.stringify({ ok: false, stub: true, url }),
        { status: 404, headers: { "Content-Type": "application/json" } },
      ));
    }
    return origFetch(input, init);
  };

  // ---------------------------------------------------------------------
  // 2. Confirm-and-close: build the payload from app.js global state and
  //    post it back to the ComfyUI window that opened us.
  // ---------------------------------------------------------------------
  function buildPayload() {
    if (MODE === "pose") {
      // ``window.__sam3dPosePayload`` is populated by editor_core.js's
      // refreshEditorPosePayload() in a flat layout that's key-compatible
      // with SAM3DBodyProcessToJson — Render reads body_pose_params /
      // global_rot / camera / image_size / ... straight from the top
      // level. We pass it through verbatim.
      const p = window.__sam3dPosePayload;
      if (!p || !p.body_pose_params) {
        throw new Error(
          "no pose has been estimated yet — pick an image and press " +
          "'Run pose estimation' first.\n" +
          "ポーズが推定されていません。画像を選び『ポーズを推定』を押してください。"
        );
      }
      return p;
    }
    // character: ship the live tabSettings.make subset.
    const m = window.__sam3dCharaPayload;
    if (!m) {
      throw new Error(
        "character settings are empty — adjust at least one slider " +
        "or load a preset before confirming.\n" +
        "キャラクター設定が空です。スライダーまたはプリセットで一度設定してください。"
      );
    }
    return {
      body_params:  m.body_params || {},
      bone_lengths: m.bone_lengths || {},
      blendshapes:  m.blendshapes || {},
    };
  }

  // Resolve the ComfyUI window to post the confirmed payload back to.
  // Modal/iframe → window.parent. Popup window → window.opener. Direct
  // standalone navigation → null (clipboard fallback).
  function resolveTargetWindow() {
    try {
      if (window.parent && window.parent !== window) return window.parent;
    } catch (_e) {}
    try {
      if (window.opener && !window.opener.closed) return window.opener;
    } catch (_e) {}
    return null;
  }

  function confirmAndClose() {
    let payload;
    try {
      payload = buildPayload();
    } catch (exc) {
      alert(String(exc?.message || exc));
      return;
    }
    const json = JSON.stringify(payload);
    const msg = {
      type: CONFIRM_TYPE_BY_MODE[MODE],
      node_id: NODE_ID,
      [PAYLOAD_KEY_BY_MODE[MODE]]: json,
    };
    const target = resolveTargetWindow();
    if (!target) {
      try { navigator.clipboard.writeText(json); } catch (_e) {}
      alert(
        "Editor was not opened from a ComfyUI node, so the result " +
        "cannot be posted back automatically. The JSON has been copied " +
        "to your clipboard instead."
      );
      return;
    }
    try {
      target.postMessage(msg, window.location.origin);
    } catch (e) {
      console.error("[SAM3D editor_init] postMessage failed:", e);
      return;
    }
    // Popup case: close ourselves so the user lands back on ComfyUI.
    // Iframe case: window.close() is a no-op; the parent removes the
    // modal in response to the confirm message.
    if (window.opener && window.opener !== window.parent) {
      setTimeout(() => window.close(), 30);
    }
  }

  // ---------------------------------------------------------------------
  // 3. Wire the confirm button once the DOM is ready.
  // ---------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("confirm-and-close-btn");
    if (btn) btn.addEventListener("click", confirmAndClose);
    document.addEventListener("keydown", (ev) => {
      // Ctrl/Cmd + Enter shortcut.
      if ((ev.ctrlKey || ev.metaKey) && ev.key === "Enter") {
        ev.preventDefault();
        confirmAndClose();
      }
    });
  });

  // Expose for editor_core.js to populate.
  window.SAM3D_EDITOR = {
    mode: MODE,
    nodeId: NODE_ID,
    confirmAndClose,
  };
})();

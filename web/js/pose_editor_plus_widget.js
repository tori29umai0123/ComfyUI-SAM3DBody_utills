/**
 * SAM 3D Body — Pose Editor + node widget.
 *
 * The node has no input sockets; everything happens inside the popup at
 * /sam3d/editor/pose_plus. The popup posts back via window.parent.postMessage
 * with type "sam3d-pose-plus-confirmed" and the captured base64 PNG strings
 * (composite render, per-person renders, input image, per-person hand crops).
 * We mirror those into hidden node widgets so they serialise with the
 * workflow and the Python node can decode them on execute().
 */

import { app } from "../../../../scripts/app.js";

const TARGET_NODE = "SAM3DBodyPoseEditorPlus";
const CONFIRM_MSG = "sam3d-pose-plus-confirmed";
const CANCEL_MSG  = "sam3d-pose-plus-cancelled";
const HIDDEN_NAMES = [
  "pose_image",
  "pose_images",
  "input_image",
  "hand_l_images",
  "hand_r_images",
];
const STATUS_NAME = "status";
const EDITOR_PATH = "/sam3d/editor/pose_plus";
const MODAL_ID    = "sam3d-pose-plus-editor-modal";

function _ensureHidden(node, name) {
  let w = node.widgets?.find((x) => x.name === name);
  if (!w) {
    w = node.addWidget("text", name, "", () => {}, {
      multiline: false, serialize: true,
    });
  }
  w.serialize = true;
  w.computeSize = () => [0, -4];
  w.draw = () => {};
  w.type = "hidden";
  return w;
}

function attachWidgets(node) {
  if (node.__sam3dMultiAttached) return;
  node.__sam3dMultiAttached = true;

  const hidden = {};
  for (const name of HIDDEN_NAMES) {
    hidden[name] = _ensureHidden(node, name);
  }

  const statusWidget = node.addWidget(
    "text",
    STATUS_NAME,
    hidden.pose_image.value ? "confirmed" : "(未確定 / unset)",
    () => {},
    { serialize: false },
  );
  statusWidget.disabled = true;

  node.addWidget("button", "Open Pose Editor +", null, () => {
    openEditor(node);
  });

  const refresh = () => {
    const v = hidden.pose_image.value || "";
    if (!v) {
      statusWidget.value = "(未確定 / unset)";
    } else {
      let n = 0;
      try { n = JSON.parse(hidden.pose_images.value || "[]").length; } catch (_e) {}
      statusWidget.value = `confirmed (${n} 人)`;
    }
    node.setDirtyCanvas?.(true, true);
  };
  refresh();
  node.__sam3dMultiRefresh = refresh;
  node.__sam3dMultiHidden  = hidden;

  const origConfig = node.onConfigure;
  node.onConfigure = function (info) {
    const r = origConfig?.apply(this, arguments);
    refresh();
    return r;
  };

  if (node.size?.[0] < 240) node.size[0] = 240;
}

// ---------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------

function ensureModalStyles() {
  if (document.getElementById("sam3d-pose-plus-modal-styles")) return;
  const style = document.createElement("style");
  style.id = "sam3d-pose-plus-modal-styles";
  style.textContent = `
.sam3d-pose-plus-modal-backdrop {
  position: fixed; inset: 0; background: rgba(0,0,0,0.6);
  z-index: 99999; display: flex; align-items: center; justify-content: center;
}
.sam3d-pose-plus-modal-frame {
  position: relative; width: 96vw; height: 94vh;
  background: #1e1e1e; border: 1px solid #444; border-radius: 8px;
  box-shadow: 0 10px 40px rgba(0,0,0,0.7); overflow: hidden;
}
.sam3d-pose-plus-modal-frame > iframe {
  width: 100%; height: 100%; border: 0; display: block; background: #1e1e1e;
}
.sam3d-pose-plus-modal-close {
  position: absolute; top: 6px; right: 8px; z-index: 2;
  width: 28px; height: 28px; line-height: 28px;
  border: none; border-radius: 4px;
  background: rgba(0,0,0,0.45); color: #ddd;
  font-size: 18px; font-weight: 700; cursor: pointer;
}
.sam3d-pose-plus-modal-close:hover { background: rgba(255,80,80,0.7); color: white; }
`;
  document.head.appendChild(style);
}

function closeModal() {
  const modal = document.getElementById(MODAL_ID);
  if (!modal) return;
  modal.remove();
}

function openEditor(node) {
  ensureModalStyles();
  closeModal();
  // Forward the live ``blender_exe`` widget value to the iframe so the
  // in-editor FBX / BVH buttons use whatever the user typed on the node
  // (without having to run the node first).
  const blenderWidget = node.widgets?.find((w) => w.name === "blender_exe");
  const blenderExe = blenderWidget?.value ? String(blenderWidget.value) : "";
  const params = new URLSearchParams({
    node_id: String(node.id),
    embed: "1",
    _t: String(Date.now()),
  });
  if (blenderExe) params.set("blender_exe", blenderExe);
  const url = `${EDITOR_PATH}?${params.toString()}`;
  const backdrop = document.createElement("div");
  backdrop.className = "sam3d-pose-plus-modal-backdrop";
  backdrop.id = MODAL_ID;
  backdrop.innerHTML = `
    <div class="sam3d-pose-plus-modal-frame">
      <button class="sam3d-pose-plus-modal-close" title="Close (Esc)">×</button>
      <iframe src="${url}" allow="clipboard-write"></iframe>
    </div>
  `;
  backdrop.querySelector(".sam3d-pose-plus-modal-close").addEventListener("click", closeModal);
  backdrop.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") closeModal();
  });
  document.body.appendChild(backdrop);
  backdrop.tabIndex = -1;
  backdrop.focus();
}

window.addEventListener("message", (evt) => {
  if (!evt.data) return;
  if (evt.origin !== window.location.origin) return;
  if (evt.data.type === CANCEL_MSG) { closeModal(); return; }
  if (evt.data.type !== CONFIRM_MSG) return;

  const target = app.graph?.getNodeById?.(Number(evt.data.node_id));
  if (!target || target.comfyClass !== TARGET_NODE) { closeModal(); return; }
  const hidden = target.__sam3dMultiHidden;
  if (!hidden) { closeModal(); return; }

  for (const name of HIDDEN_NAMES) {
    const v = evt.data[name];
    hidden[name].value = typeof v === "string" ? v : "";
  }
  target.__sam3dMultiRefresh?.();
  closeModal();
});

app.registerExtension({
  name: "SAM3DBody.PoseEditorPlusWidget",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET_NODE) return;
    const orig = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = orig?.apply(this, arguments);
      attachWidgets(this);
      return r;
    };
  },
});

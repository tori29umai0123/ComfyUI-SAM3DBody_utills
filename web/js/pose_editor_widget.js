/**
 * SAM 3D Body — Pose Editor node widget.
 *
 * Adds an "Open Pose Editor" button + a hidden pose_json text widget to
 * SAM3DBodyPoseEditor nodes. Clicking the button opens an in-page modal
 * that embeds /sam3d/editor/pose?node_id=... in an iframe. The iframe
 * confirms via window.parent.postMessage({type: "sam3d-pose-confirmed",
 * ...}); the listener mirrors the JSON into the hidden widget so it
 * serialises with the workflow, then removes the modal.
 */

import { app } from "../../../../scripts/app.js";

const TARGET_NODE = "SAM3DBodyPoseEditor";
const CONFIRM_MSG = "sam3d-pose-confirmed";
const CANCEL_MSG  = "sam3d-pose-cancelled";
const HIDDEN_NAME = "pose_json";
const STATUS_NAME = "status";
const EDITOR_PATH = "/sam3d/editor/pose";
const MODAL_ID    = "sam3d-pose-editor-modal";

function attachWidgets(node) {
    if (node.__sam3dPoseEditorAttached) return;
    node.__sam3dPoseEditorAttached = true;

    let hiddenWidget = node.widgets?.find((w) => w.name === HIDDEN_NAME);
    if (!hiddenWidget) {
        hiddenWidget = node.addWidget("text", HIDDEN_NAME, "", () => {}, {
            multiline: false,
            serialize: true,
        });
    }
    hiddenWidget.serialize = true;
    hiddenWidget.computeSize = () => [0, -4];
    hiddenWidget.draw = () => {};
    hiddenWidget.type = "hidden";

    const statusWidget = node.addWidget(
        "text",
        STATUS_NAME,
        hiddenWidget.value ? `confirmed (${hiddenWidget.value.length} chars)` : "(未確定 / unset)",
        () => {},
        { serialize: false },
    );
    statusWidget.disabled = true;

    node.addWidget("button", "Open Pose Editor", null, () => {
        openEditor(node);
    });

    const refresh = () => {
        const v = hiddenWidget.value || "";
        statusWidget.value = v
            ? `confirmed (${v.length} chars)`
            : "(未確定 / unset)";
        node.setDirtyCanvas?.(true, true);
    };
    refresh();
    node.__sam3dPoseRefresh = refresh;
    node.__sam3dPoseHidden = hiddenWidget;

    const origConfig = node.onConfigure;
    node.onConfigure = function (info) {
        const r = origConfig?.apply(this, arguments);
        refresh();
        return r;
    };

    if (node.size?.[0] < 220) node.size[0] = 240;
}

// ---------------------------------------------------------------------------
// In-page modal that hosts the editor inside an iframe.
// ---------------------------------------------------------------------------

function ensureModalStyles() {
    if (document.getElementById("sam3d-modal-styles")) return;
    const style = document.createElement("style");
    style.id = "sam3d-modal-styles";
    style.textContent = `
.sam3d-modal-backdrop {
    position: fixed; inset: 0; background: rgba(0,0,0,0.6);
    z-index: 99999; display: flex; align-items: center; justify-content: center;
}
.sam3d-modal-frame {
    position: relative; width: 96vw; height: 94vh;
    background: #1e1e1e; border: 1px solid #444; border-radius: 8px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.7); overflow: hidden;
}
.sam3d-modal-frame > iframe {
    width: 100%; height: 100%; border: 0; display: block; background: #1e1e1e;
}
.sam3d-modal-close {
    position: absolute; top: 6px; right: 8px; z-index: 2;
    width: 28px; height: 28px; line-height: 28px;
    border: none; border-radius: 4px;
    background: rgba(0,0,0,0.45); color: #ddd;
    font-size: 18px; font-weight: 700; cursor: pointer;
}
.sam3d-modal-close:hover { background: rgba(255,80,80,0.7); color: white; }
`;
    document.head.appendChild(style);
}

function closeModal() {
    const modal = document.getElementById(MODAL_ID);
    if (modal) modal.remove();
}

function openEditor(node) {
    ensureModalStyles();
    closeModal();  // belt-and-suspenders if a previous modal got stuck
    const url = `${EDITOR_PATH}?node_id=${encodeURIComponent(node.id)}&embed=1`;
    const backdrop = document.createElement("div");
    backdrop.className = "sam3d-modal-backdrop";
    backdrop.id = MODAL_ID;
    backdrop.innerHTML = `
        <div class="sam3d-modal-frame">
            <button class="sam3d-modal-close" title="Close (Esc)">×</button>
            <iframe src="${url}" allow="clipboard-write"></iframe>
        </div>
    `;
    backdrop.querySelector(".sam3d-modal-close").addEventListener("click", closeModal);
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
    if (evt.data.type === CANCEL_MSG) {
        closeModal();
        return;
    }
    if (evt.data.type !== CONFIRM_MSG) return;
    const { node_id, pose_json } = evt.data;
    if (typeof pose_json !== "string") return;
    const target = app.graph?.getNodeById?.(Number(node_id));
    if (target && target.comfyClass === TARGET_NODE && target.__sam3dPoseHidden) {
        target.__sam3dPoseHidden.value = pose_json;
        target.__sam3dPoseRefresh?.();
    }
    closeModal();
});

app.registerExtension({
    name: "SAM3DBody.PoseEditorWidget",
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

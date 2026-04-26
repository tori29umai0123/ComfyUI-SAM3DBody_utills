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
const HIDDEN_POSE_IMG  = "pose_image";
const HIDDEN_INPUT_IMG = "input_image";
const HIDDEN_LHAND_IMG = "hand_l_image";
const HIDDEN_RHAND_IMG = "hand_r_image";
const STATUS_NAME = "status";
const EDITOR_PATH = "/sam3d/editor/pose";
const MODAL_ID    = "sam3d-pose-editor-modal";

function _ensureHidden(node, name) {
    let w = node.widgets?.find((x) => x.name === name);
    if (!w) {
        w = node.addWidget("text", name, "", () => {}, {
            multiline: false,
            serialize: true,
        });
    }
    w.serialize = true;
    w.computeSize = () => [0, -4];
    w.draw = () => {};
    w.type = "hidden";
    return w;
}

function attachWidgets(node) {
    if (node.__sam3dPoseEditorAttached) return;
    node.__sam3dPoseEditorAttached = true;

    const hiddenWidget    = _ensureHidden(node, HIDDEN_NAME);
    const hiddenPoseImg   = _ensureHidden(node, HIDDEN_POSE_IMG);
    const hiddenInputImg  = _ensureHidden(node, HIDDEN_INPUT_IMG);
    const hiddenLHandImg  = _ensureHidden(node, HIDDEN_LHAND_IMG);
    const hiddenRHandImg  = _ensureHidden(node, HIDDEN_RHAND_IMG);

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
        const imgFlag = hiddenPoseImg.value ? " +img" : "";
        const inFlag  = hiddenInputImg.value ? " +in" : "";
        const hlFlag  = hiddenLHandImg.value ? " +hl" : "";
        const hrFlag  = hiddenRHandImg.value ? " +hr" : "";
        statusWidget.value = v
            ? `confirmed (${v.length} chars)${imgFlag}${inFlag}${hlFlag}${hrFlag}`
            : "(未確定 / unset)";
        node.setDirtyCanvas?.(true, true);
    };
    refresh();
    node.__sam3dPoseRefresh    = refresh;
    node.__sam3dPoseHidden     = hiddenWidget;
    node.__sam3dPoseImgHidden  = hiddenPoseImg;
    node.__sam3dPoseInHidden   = hiddenInputImg;
    node.__sam3dPoseLHandHidden = hiddenLHandImg;
    node.__sam3dPoseRHandHidden = hiddenRHandImg;

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
    // ``_t`` is a per-open cache-buster so a stale browser cache from the
    // pre-image-output version of the editor can't keep serving the old
    // PNG-save UI after this widget is updated.
    const url = `${EDITOR_PATH}?node_id=${encodeURIComponent(node.id)}&embed=1&_t=${Date.now()}`;
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
    const { node_id, pose_json, pose_image, input_image,
            hand_l_image, hand_r_image } = evt.data;
    if (typeof pose_json !== "string") return;
    const target = app.graph?.getNodeById?.(Number(node_id));
    if (target && target.comfyClass === TARGET_NODE && target.__sam3dPoseHidden) {
        target.__sam3dPoseHidden.value = pose_json;
        // Image widgets are optional — clear them when the editor didn't
        // ship one so a stale capture from a previous confirm doesn't get
        // re-served on the next workflow run.
        if (target.__sam3dPoseImgHidden) {
            target.__sam3dPoseImgHidden.value =
                typeof pose_image === "string" ? pose_image : "";
        }
        if (target.__sam3dPoseInHidden) {
            target.__sam3dPoseInHidden.value =
                typeof input_image === "string" ? input_image : "";
        }
        if (target.__sam3dPoseLHandHidden) {
            target.__sam3dPoseLHandHidden.value =
                typeof hand_l_image === "string" ? hand_l_image : "";
        }
        if (target.__sam3dPoseRHandHidden) {
            target.__sam3dPoseRHandHidden.value =
                typeof hand_r_image === "string" ? hand_r_image : "";
        }
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

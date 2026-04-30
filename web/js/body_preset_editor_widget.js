/**
 * SAM 3D Body — Body Preset Editor node widget.
 *
 * Mirror of pose_editor_widget.js for SAM3DBodyBodyPresetEditor — opens
 * /sam3d/editor/body_preset in an in-page iframe modal and listens for
 * sam3d-body-preset-confirmed.
 */

import { app } from "../../../../scripts/app.js";

const TARGET_NODE = "SAM3DBodyBodyPresetEditor";
const CONFIRM_MSG = "sam3d-body-preset-confirmed";
const CANCEL_MSG  = "sam3d-body-preset-cancelled";
const HIDDEN_NAME = "body_preset_json";
const HIDDEN_BODY_PRESET_IMG = "body_preset_image";
const STATUS_NAME = "status";
const EDITOR_PATH = "/sam3d/editor/body_preset";
const MODAL_ID    = "sam3d-body-preset-editor-modal";

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
    if (node.__sam3dBodyPresetEditorAttached) return;
    node.__sam3dBodyPresetEditorAttached = true;

    const hiddenWidget         = _ensureHidden(node, HIDDEN_NAME);
    const hiddenBodyPresetImg  = _ensureHidden(node, HIDDEN_BODY_PRESET_IMG);

    const statusWidget = node.addWidget(
        "text",
        STATUS_NAME,
        hiddenWidget.value ? `confirmed (${hiddenWidget.value.length} chars)` : "(未確定 / unset)",
        () => {},
        { serialize: false },
    );
    statusWidget.disabled = true;

    node.addWidget("button", "Open Body Preset Editor", null, () => {
        openEditor(node);
    });

    const refresh = () => {
        const v = hiddenWidget.value || "";
        const imgFlag = hiddenBodyPresetImg.value ? " +img" : "";
        statusWidget.value = v
            ? `confirmed (${v.length} chars)${imgFlag}`
            : "(未確定 / unset)";
        node.setDirtyCanvas?.(true, true);
    };
    refresh();
    node.__sam3dBodyPresetRefresh    = refresh;
    node.__sam3dBodyPresetHidden     = hiddenWidget;
    node.__sam3dBodyPresetImgHidden  = hiddenBodyPresetImg;

    const origConfig = node.onConfigure;
    node.onConfigure = function (info) {
        const r = origConfig?.apply(this, arguments);
        refresh();
        return r;
    };

    if (node.size?.[0] < 240) node.size[0] = 260;
}

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
    closeModal();
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
    const { node_id, body_preset_json, body_preset_image } = evt.data;
    if (typeof body_preset_json !== "string") return;
    const target = app.graph?.getNodeById?.(Number(node_id));
    if (target && target.comfyClass === TARGET_NODE && target.__sam3dBodyPresetHidden) {
        target.__sam3dBodyPresetHidden.value = body_preset_json;
        if (target.__sam3dBodyPresetImgHidden) {
            target.__sam3dBodyPresetImgHidden.value =
                typeof body_preset_image === "string" ? body_preset_image : "";
        }
        target.__sam3dBodyPresetRefresh?.();
    }
    closeModal();
});

app.registerExtension({
    name: "SAM3DBody.BodyPresetEditorWidget",
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

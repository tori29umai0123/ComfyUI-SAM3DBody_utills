/**
 * SAM 3D Body — Character Editor node widget.
 *
 * Mirror of pose_editor_widget.js for SAM3DBodyCharacterEditor — opens
 * /sam3d/editor/character in an in-page iframe modal and listens for
 * sam3d-chara-confirmed.
 */

import { app } from "../../../../scripts/app.js";

const TARGET_NODE = "SAM3DBodyCharacterEditor";
const CONFIRM_MSG = "sam3d-chara-confirmed";
const CANCEL_MSG  = "sam3d-chara-cancelled";
const HIDDEN_NAME = "chara_json";
const HIDDEN_CHARA_IMG = "chara_image";
const STATUS_NAME = "status";
const EDITOR_PATH = "/sam3d/editor/character";
const MODAL_ID    = "sam3d-chara-editor-modal";

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
    if (node.__sam3dCharaEditorAttached) return;
    node.__sam3dCharaEditorAttached = true;

    const hiddenWidget    = _ensureHidden(node, HIDDEN_NAME);
    const hiddenCharaImg  = _ensureHidden(node, HIDDEN_CHARA_IMG);

    const statusWidget = node.addWidget(
        "text",
        STATUS_NAME,
        hiddenWidget.value ? `confirmed (${hiddenWidget.value.length} chars)` : "(未確定 / unset)",
        () => {},
        { serialize: false },
    );
    statusWidget.disabled = true;

    node.addWidget("button", "Open Character Editor", null, () => {
        openEditor(node);
    });

    const refresh = () => {
        const v = hiddenWidget.value || "";
        const imgFlag = hiddenCharaImg.value ? " +img" : "";
        statusWidget.value = v
            ? `confirmed (${v.length} chars)${imgFlag}`
            : "(未確定 / unset)";
        node.setDirtyCanvas?.(true, true);
    };
    refresh();
    node.__sam3dCharaRefresh    = refresh;
    node.__sam3dCharaHidden     = hiddenWidget;
    node.__sam3dCharaImgHidden  = hiddenCharaImg;

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
    const { node_id, chara_json, chara_image } = evt.data;
    if (typeof chara_json !== "string") return;
    const target = app.graph?.getNodeById?.(Number(node_id));
    if (target && target.comfyClass === TARGET_NODE && target.__sam3dCharaHidden) {
        target.__sam3dCharaHidden.value = chara_json;
        if (target.__sam3dCharaImgHidden) {
            target.__sam3dCharaImgHidden.value =
                typeof chara_image === "string" ? chara_image : "";
        }
        target.__sam3dCharaRefresh?.();
    }
    closeModal();
});

app.registerExtension({
    name: "SAM3DBody.CharacterEditorWidget",
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

/**
 * SAM3DBody: preset → slider sync for the
 * `SAM 3D Body: Setting Body Preset JSON` node.
 *
 * The `preset` dropdown drives the slider widgets. When the user picks
 * a preset — including the default `autosave` entry — the frontend
 * fetches `body_preset_settings/<name>.json` and copies its
 * body/bone/blendshape values into the corresponding widgets. The user
 * can then tweak the sliders further; the Python side does not re-apply
 * the preset at build time, so manual adjustments are respected.
 *
 * (Previous revision targeted the legacy
 * ``SAM3DBodyRenderFromJson`` node, which has been split into
 * SettingBodyPresetJson + RenderFromPoseAndBodyPresetJson. The preset →
 * slider sync logic moved to SettingBodyPresetJson because that's where
 * the widgets now live; render is purely body_preset_json-driven.)
 */

import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const TARGET_NODE = "SAM3DBodySettingBodyPresetJson";

async function fetchPreset(name) {
    if (!name) return null;
    try {
        const resp = await api.fetchApi(
            `/sam3d/preset/${encodeURIComponent(name)}`,
        );
        if (!resp.ok) return null;
        return await resp.json();
    } catch (e) {
        console.warn(`[SAM3DBody] preset fetch failed (${name}):`, e);
        return null;
    }
}

function buildWidgetMap(preset) {
    // Flatten body_params / bone_lengths / blendshapes to widget names.
    const out = {};
    const body = preset.body_params || {};
    const bone = preset.bone_lengths || {};
    const bs   = preset.blendshapes  || {};
    for (const [k, v] of Object.entries(body)) out[`body_${k}`] = v;
    for (const [k, v] of Object.entries(bone)) out[`bone_${k}`] = v;
    for (const [k, v] of Object.entries(bs))   out[`bs_${k}`]   = v;
    return out;
}

function applyToWidgets(node, values) {
    if (!node?.widgets) return;
    for (const widget of node.widgets) {
        if (Object.prototype.hasOwnProperty.call(values, widget.name)) {
            widget.value = values[widget.name];
            if (widget.callback) {
                try { widget.callback(widget.value); } catch (_) { /* ignore */ }
            }
        }
    }
    node.setDirtyCanvas?.(true, true);
}

function hookPresetWidget(node) {
    const presetWidget = node.widgets?.find((w) => w.name === "preset");
    if (!presetWidget) return;
    if (presetWidget.__sam3dPresetHooked) return;
    presetWidget.__sam3dPresetHooked = true;

    const origCallback = presetWidget.callback;
    presetWidget.callback = function (value) {
        if (origCallback) {
            try { origCallback.apply(this, arguments); } catch (_) { /* ignore */ }
        }
        fetchPreset(value).then((preset) => {
            if (!preset) return;
            applyToWidgets(node, buildWidgetMap(preset));
        });
    };
}

app.registerExtension({
    name: "SAM3DBody.Autosave",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);
            hookPresetWidget(this);
            return r;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = origOnConfigure?.apply(this, arguments);
            hookPresetWidget(this);
            return r;
        };
    },
});

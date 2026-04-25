# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""Character Editor terminal node — analogous to PoseEditor but emits
``chara_json`` (body_params + bone_lengths + blendshapes)."""

from __future__ import annotations


class SAM3DBodyCharacterEditor:
    """Terminal node that surfaces a confirmed character JSON string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "node_id": "UNIQUE_ID",
                "chara_json": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("chara_json",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "SAM3DBody/Editor"

    @classmethod
    def IS_CHANGED(cls, node_id="", chara_json="", **_):
        return chara_json or "<unset>"

    def execute(self, node_id="", chara_json="", **_):
        if not chara_json:
            raise RuntimeError(
                "[SAM3DBody] Character Editor: キャラクターが未確定です。"
                "ノード上の『Open Character Editor』ボタンを押し、"
                "エディタで『確定して閉じる』を押してください。\n"
                "Character Editor: character has not been confirmed yet. "
                "Click 'Open Character Editor' on the node and press "
                "'Confirm & Close' inside the editor."
            )
        return (chara_json,)


NODE_CLASS_MAPPINGS = {
    "SAM3DBodyCharacterEditor": SAM3DBodyCharacterEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyCharacterEditor": "SAM 3D Body: Character Editor",
}

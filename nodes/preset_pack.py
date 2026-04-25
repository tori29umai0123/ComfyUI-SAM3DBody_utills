"""Project-wide configuration accessor.

Settings live in ``config.ini`` at the repo root. Two sections so far:

    [active]
    pack = default                                                 ; preset pack name

    [blender]
    exe_path = C:\\Program Files\\Blender Foundation\\Blender 4.1\\blender.exe

Files older than this revision used ``active_preset.ini`` (with only the
``[active]`` section). On first read the legacy name is migrated to
``config.ini`` automatically — the user doesn't have to do anything.

Public surface kept stable so existing imports keep working:
    repo_root, active_pack_name, active_pack_dir, npz_path,
    vertices_json_path, chara_settings_dir.

Added in this revision:
    get_blender_exe_path, set_blender_exe_path.
"""

from __future__ import annotations

import configparser
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / "config.ini"
_LEGACY_INI_PATH = _REPO_ROOT / "active_preset.ini"
_DEFAULT_PACK = "default"
_DEFAULT_BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"


def repo_root() -> Path:
    """Repository root (the `ComfyUI-SAM3DBody_utills/` folder)."""
    return _REPO_ROOT


# ---------------------------------------------------------------------------
# Internal: read/write config.ini, with one-shot migration from the legacy
# ``active_preset.ini`` filename.
# ---------------------------------------------------------------------------

def _read_config() -> configparser.ConfigParser:
    cp = configparser.ConfigParser()
    if _CONFIG_PATH.is_file():
        try:
            cp.read(_CONFIG_PATH, encoding="utf-8")
        except Exception as exc:
            print(f"[SAM3DBody] config.ini parse failed: {exc}; using defaults")
        return cp
    # Legacy migration: an older install only has ``active_preset.ini``.
    # Read its contents, then write them out as ``config.ini`` so subsequent
    # calls hit the modern path. The legacy file is renamed to .bak rather
    # than deleted so a worried user can roll back manually.
    if _LEGACY_INI_PATH.is_file():
        try:
            cp.read(_LEGACY_INI_PATH, encoding="utf-8")
        except Exception as exc:
            print(f"[SAM3DBody] {_LEGACY_INI_PATH.name} parse failed: {exc}")
        try:
            _write_config(cp)
            backup = _LEGACY_INI_PATH.with_suffix(".ini.bak")
            try:
                if backup.exists():
                    backup.unlink()
                _LEGACY_INI_PATH.rename(backup)
                print(f"[SAM3DBody] migrated {_LEGACY_INI_PATH.name} → "
                      f"{_CONFIG_PATH.name} (old file kept as {backup.name})")
            except Exception as exc:
                print(f"[SAM3DBody] migration write OK but rename failed: {exc}")
        except Exception as exc:
            print(f"[SAM3DBody] migration to config.ini failed: {exc}")
    return cp


def _write_config(cp: configparser.ConfigParser) -> None:
    """Persist ``cp`` to ``config.ini``, ensuring the file gets the
    standard explanatory header on (re)creation."""
    header = (
        "; SAM3DBody runtime configuration.\n"
        ";\n"
        "; [active]   pack       = directory name under presets/ that supplies\n"
        ";                         face_blendshapes.npz, mhr_reference_vertices.json,\n"
        ";                         and chara_settings_presets/*.json.\n"
        "; [blender]  exe_path   = absolute path to blender(.exe). Used by the\n"
        ";                         FBX/BVH export nodes. Updated automatically\n"
        ";                         whenever an export node runs with a new path,\n"
        ";                         so newly-added export nodes inherit the latest\n"
        ";                         working location as their UI default.\n"
        "\n"
    )
    # Make sure the canonical sections exist so a fresh write isn't empty.
    if not cp.has_section("active"):
        cp.add_section("active")
    cp["active"].setdefault("pack", _DEFAULT_PACK)
    if not cp.has_section("blender"):
        cp.add_section("blender")
    cp["blender"].setdefault("exe_path", _DEFAULT_BLENDER_EXE)

    with _CONFIG_PATH.open("w", encoding="utf-8") as f:
        f.write(header)
        cp.write(f)


def _set_kv(section: str, key: str, value: str) -> None:
    """Update a single ``[section] key = value`` pair in config.ini."""
    cp = _read_config()
    if not cp.has_section(section):
        cp.add_section(section)
    cp[section][key] = value
    _write_config(cp)


# ---------------------------------------------------------------------------
# [active] pack
# ---------------------------------------------------------------------------

def active_pack_name() -> str:
    """Read the active pack name. Falls back to ``default`` if the ini is
    missing, malformed, or the named pack folder doesn't exist on disk."""
    name = _DEFAULT_PACK
    cp = _read_config()
    candidate = cp.get("active", "pack", fallback=_DEFAULT_PACK).strip()
    if candidate:
        name = candidate
    # Final safety — fall back to default if the named pack isn't on disk.
    if not (_REPO_ROOT / "presets" / name).is_dir():
        fallback_dir = _REPO_ROOT / "presets" / _DEFAULT_PACK
        if fallback_dir.is_dir() and name != _DEFAULT_PACK:
            print(f"[SAM3DBody] preset pack '{name}' not found under "
                  f"{_REPO_ROOT / 'presets'}; falling back to '{_DEFAULT_PACK}'")
            name = _DEFAULT_PACK
    return name


def active_pack_dir() -> Path:
    """Absolute path to the active preset pack's folder."""
    return _REPO_ROOT / "presets" / active_pack_name()


def npz_path() -> Path:
    return active_pack_dir() / "face_blendshapes.npz"


def vertices_json_path(object_name: str) -> Path:
    return active_pack_dir() / f"{object_name}_vertices.json"


def chara_settings_dir() -> Path:
    return active_pack_dir() / "chara_settings_presets"


# ---------------------------------------------------------------------------
# [blender] exe_path
# ---------------------------------------------------------------------------

def get_blender_exe_path() -> str:
    """Return the configured Blender executable path, or the project default
    if the ini hasn't been written yet. Always returns a non-empty string —
    callers can use the value directly as a subprocess argv[0]."""
    cp = _read_config()
    raw = cp.get("blender", "exe_path", fallback=_DEFAULT_BLENDER_EXE).strip()
    return raw or _DEFAULT_BLENDER_EXE


def set_blender_exe_path(path: str) -> None:
    """Persist ``path`` as the new ``[blender] exe_path``. No-op if ``path``
    is empty / whitespace, or already equal to the stored value (so nodes
    can call this unconditionally on every execute without churning the
    file). Tilde and environment variables are NOT expanded — the value
    travels through the export nodes verbatim, preserving whatever format
    the user typed."""
    if not path or not path.strip():
        return
    new = path.strip()
    cp = _read_config()
    cur = cp.get("blender", "exe_path", fallback="").strip()
    if cur == new:
        return
    _set_kv("blender", "exe_path", new)


# Convenience constant for callers that want the hard-coded fallback
# without having to recompute it (e.g., for tooltips).
DEFAULT_BLENDER_EXE = _DEFAULT_BLENDER_EXE

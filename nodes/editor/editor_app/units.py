# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""Display-unit handling for the Pose Editor +.

Internal math always uses meters. The frontend can present height inputs in
either centimeters or inches depending on ``[units] display`` in
``config.ini``; this module is the single conversion boundary so no other
code has to remember which side of the boundary it is on.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...preset_pack import (
    DEFAULT_ADULT_HEIGHT_M,
    DEFAULT_DISPLAY_UNIT,
    VALID_DISPLAY_UNITS,
    get_default_adult_height_m,
    get_display_unit,
)

_INCH_TO_M = 0.0254
_CM_TO_M = 0.01


@dataclass(frozen=True)
class UnitConfig:
    display_unit: str            # "cm" | "inch"
    default_adult_height_m: float


def read_unit_config() -> UnitConfig:
    return UnitConfig(
        display_unit=get_display_unit(),
        default_adult_height_m=get_default_adult_height_m(),
    )


def to_meters(value: float, unit: str) -> float:
    """Convert ``value`` from ``unit`` ("cm" | "inch" | "m") into meters.

    Anything not matching the three known units is treated as meters so
    callers passing a pre-converted m value (the API contract) keep working
    without surprises.
    """
    u = (unit or "").strip().lower()
    if u == "cm":
        return float(value) * _CM_TO_M
    if u == "inch":
        return float(value) * _INCH_TO_M
    return float(value)


def from_meters(meters: float, unit: str) -> float:
    """Convert a meter value into the display unit. ``unit="m"`` returns as-is."""
    u = (unit or "").strip().lower()
    if u == "cm":
        return float(meters) / _CM_TO_M
    if u == "inch":
        return float(meters) / _INCH_TO_M
    return float(meters)


__all__ = [
    "DEFAULT_ADULT_HEIGHT_M",
    "DEFAULT_DISPLAY_UNIT",
    "VALID_DISPLAY_UNITS",
    "UnitConfig",
    "from_meters",
    "read_unit_config",
    "to_meters",
]

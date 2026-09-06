"""Streamlit bridge for the locally bundled, interactive turbine viewer."""

from __future__ import annotations

import math
from pathlib import Path

import streamlit.components.v2 as components


_ASSET_DIRECTORY = Path(__file__).resolve().parent / "viewer"


def _read_asset(name: str) -> str:
    try:
        content = (_ASSET_DIRECTORY / name).read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"The turbine viewer asset '{name}' could not be read. "
            "Restore the tracked src/viewer files from a complete project checkout."
        ) from error
    # Multiline content is treated as inline source, including minified JavaScript.
    return content + "\n"


_TURBINE_COMPONENT = components.component(
    "gas_turbine_viewer",
    html=_read_asset("index.html"),
    css=_read_asset("viewer.css"),
    js=_read_asset("turbine.bundle.js"),
)


def normalize_rpm(value: object) -> float | None:
    """Preserve valid source RPM; unavailable or invalid readings stop animation."""
    try:
        rpm = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return rpm if math.isfinite(rpm) and rpm >= 0 else None


def render_turbine(rpm: object, *, key: str = "gas-turbine-viewer") -> None:
    """Render the detailed turbine with a stable identity across dataset reruns."""
    _TURBINE_COMPONENT(
        data={"rpm": normalize_rpm(rpm)},
        key=key,
        height=640,
        width="stretch",
    )

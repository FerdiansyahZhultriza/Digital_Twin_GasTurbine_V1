from __future__ import annotations

import json

import pytest

from src.turbine_model import normalize_rpm


@pytest.mark.parametrize(
    ("source", "expected"),
    [(0, 0.0), (3000, 3000.0), (3009.43828125, 3009.43828125), ("2987.125", 2987.125)],
)
def test_viewer_preserves_valid_speed_precision(source: object, expected: float) -> None:
    """Measured RPM must reach JavaScript without the former 50 RPM rounding."""
    result = normalize_rpm(source)

    assert result == expected
    assert isinstance(result, float)
    assert json.loads(json.dumps({"rpm": result}, allow_nan=False))["rpm"] == expected


@pytest.mark.parametrize(
    "source",
    [None, -1, "-0.01", float("nan"), float("inf"), float("-inf"), "NaN", "unknown", {}, 10**400],
)
def test_invalid_speed_stays_unavailable_and_json_safe(source: object) -> None:
    """Unavailable/invalid telemetry must not invent rotation or break serialization."""
    result = normalize_rpm(source)

    assert result is None
    assert json.dumps({"rpm": result}, allow_nan=False) == '{"rpm": null}'

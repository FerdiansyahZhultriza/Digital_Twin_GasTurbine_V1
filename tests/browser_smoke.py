"""Manual browser regression for the interactive turbine (requires Playwright/Chrome).

Start the app, then run ``python tests/browser_smoke.py --url http://localhost:8501``.
If Playwright was installed into the local ``.test-deps`` folder it is found automatically.
Screenshots are saved under ``.artifacts``; all nonlocal network requests are blocked.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import sys
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
if (PROJECT_ROOT / ".test-deps").is_dir():
    sys.path.insert(0, str(PROJECT_ROOT / ".test-deps"))

from playwright.sync_api import Page, expect, sync_playwright

from src.config import COL, DEFAULT_DATA_FILE
from src.data_loader import load_dataset


def check_rpm(viewer, expected: float) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        value = viewer.get_attribute("data-rpm")
        if value is not None and math.isclose(float(value), expected, rel_tol=0, abs_tol=1e-9):
            return
        viewer.page.wait_for_timeout(100)
    raise AssertionError(f"Viewer RPM did not update to {expected}: {value!r}")


def check_viewer(page: Page, artifacts: Path) -> dict:
    viewer = page.get_by_test_id("turbine-viewer")
    expect(viewer).to_have_attribute("data-ready", "true", timeout=60_000)
    viewer.scroll_into_view_if_needed()
    canvas = viewer.get_by_label("Interactive three-dimensional gas turbine", exact=True)
    expect(canvas).to_be_visible()
    dimensions = canvas.evaluate("el => ({width: el.width, height: el.height})")
    assert dimensions["width"] > 100 and dimensions["height"] > 100, dimensions
    expect(viewer).to_have_attribute("data-mode", "cutaway")
    expect(viewer).to_have_attribute("data-rotating", "true")
    expect(viewer).to_have_attribute("data-rotor-angle", re.compile(r"^-?\d"))

    first_angle = float(viewer.get_attribute("data-rotor-angle"))
    expect(viewer).not_to_have_attribute("data-rotor-angle", str(first_angle))
    page.wait_for_timeout(250)
    assert float(viewer.get_attribute("data-rotor-angle")) != first_angle, "Rotor is not moving"
    viewer.get_by_role("button", name="Pause rotor", exact=True).click()
    expect(viewer).to_have_attribute("data-rotating", "false")
    page.wait_for_timeout(100)
    paused_angle = float(viewer.get_attribute("data-rotor-angle"))
    page.wait_for_timeout(300)
    assert math.isclose(float(viewer.get_attribute("data-rotor-angle")), paused_angle, abs_tol=1e-8), (
        "Rotor moved while paused"
    )
    viewer.screenshot(path=str(artifacts / "turbine-cutaway-desktop.png"))

    viewer.get_by_role("button", name="Exterior", exact=True).click()
    expect(viewer).to_have_attribute("data-mode", "exterior")
    viewer.screenshot(path=str(artifacts / "turbine-exterior-desktop.png"))
    viewer.get_by_role("button", name="Cutaway", exact=True).click()
    expect(viewer).to_have_attribute("data-mode", "cutaway")

    camera = viewer.get_by_role("combobox", name="Camera view", exact=True)
    for name in ("compressor", "combustors", "turbine", "end", "overview"):
        camera.select_option(name)
        expect(camera).to_have_value(name)
        expect(canvas).to_be_visible()
    viewer.get_by_role("button", name="Show labels", exact=True).click()
    expect(viewer.get_by_role("button", name="Hide labels", exact=True)).to_be_visible()
    viewer.get_by_role("button", name="Hide labels", exact=True).click()
    expect(viewer.get_by_role("button", name="Show labels", exact=True)).to_be_visible()

    # A rerun must update unrounded telemetry without rebuilding/resetting the viewer.
    camera.select_option("turbine")
    viewer.get_by_role("button", name="Exterior", exact=True).click()
    history = page.get_by_role("slider", name="Historical record", exact=True)
    current_index = int(history.get_attribute("aria-valuenow"))
    dataset = load_dataset(DEFAULT_DATA_FILE)
    initial_rpm = float(dataset.iloc[current_index][COL["speed"]])
    actual_rpm = float(viewer.get_attribute("data-rpm"))
    assert math.isclose(actual_rpm, initial_rpm, rel_tol=0, abs_tol=1e-9), (
        f"Initial RPM {actual_rpm} differs from dataset {initial_rpm}"
    )
    target_index = 0 if current_index else len(dataset) - 1
    expected_rpm = float(dataset.iloc[target_index][COL["speed"]])
    history.focus()
    history.press("Home" if target_index == 0 else "End")
    expect(history).to_have_attribute("aria-valuenow", str(target_index))
    check_rpm(viewer, expected_rpm)
    expect(viewer).to_have_attribute("data-rotating", "false")
    expect(viewer).to_have_attribute("data-mode", "exterior")
    expect(camera).to_have_value("turbine")

    viewer.get_by_role("button", name="Cutaway", exact=True).click()
    viewer.get_by_role("button", name="Reset view", exact=True).click()
    expect(camera).to_have_value("overview")
    viewer.get_by_role("button", name="Rotate rotor", exact=True).click()
    expect(viewer).to_have_attribute("data-rotating", "true")
    viewer.scroll_into_view_if_needed()
    page.screenshot(path=str(artifacts / "turbine-app-desktop.png"), full_page=True)

    # The same component must resize without clipping its toolbar on a narrow viewport.
    page.set_viewport_size({"width": 390, "height": 844})
    viewer.scroll_into_view_if_needed()
    expect(canvas).to_be_visible()
    page.wait_for_timeout(400)
    bounds = viewer.bounding_box()
    assert bounds is not None
    assert bounds["width"] <= 390, f"Viewer exceeds mobile viewport: {bounds}"
    assert bounds["x"] >= -1 and bounds["x"] + bounds["width"] <= 391, bounds
    overflow = viewer.evaluate("el => el.scrollWidth > el.clientWidth + 1")
    assert not overflow, "Viewer overflows its mobile container"
    viewer.get_by_role("button", name="Pause rotor", exact=True).click()
    expect(viewer).to_have_attribute("data-rotating", "false")
    camera.select_option("compressor")
    expect(camera).to_have_value("compressor")
    viewer.get_by_role("button", name="Reset view", exact=True).click()
    expect(camera).to_have_value("overview")
    viewer.screenshot(path=str(artifacts / "turbine-cutaway-mobile.png"))
    return {
        "canvas": dimensions,
        "initial_rpm": initial_rpm,
        "updated_rpm": expected_rpm,
        "architecture": json.loads(viewer.get_attribute("data-architecture") or "{}"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8501")
    args = parser.parse_args()
    artifacts = PROJECT_ROOT / ".artifacts"
    artifacts.mkdir(exist_ok=True)
    page_errors: list[str] = []
    graphics_errors: list[str] = []
    failed_local_requests: list[str] = []
    blocked_hosts: set[str] = set()
    allowed_hosts = {urlsplit(args.url).hostname, "localhost", "127.0.0.1", "::1"}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(
            viewport={"width": 1600, "height": 1100},
            device_scale_factor=1,
            reduced_motion="no-preference",
        )

        def local_only(route) -> None:
            parsed = urlsplit(route.request.url)
            if parsed.scheme in {"http", "https"} and parsed.hostname not in allowed_hosts:
                blocked_hosts.add(parsed.hostname or "unknown")
                route.abort()
            else:
                route.continue_()

        context.route("**/*", local_only)
        page = context.new_page()
        page.on("pageerror", lambda error: page_errors.append(str(error)))

        def console_message(message) -> None:
            if message.type == "error" and any(word in message.text for word in ("WebGL", "THREE.", "GL_INVALID")):
                graphics_errors.append(message.text)

        page.on("console", console_message)

        def request_failed(request) -> None:
            parsed = urlsplit(request.url)
            if parsed.hostname in allowed_hosts and "ERR_ABORTED" not in str(request.failure):
                failed_local_requests.append(f"{parsed.path}: {request.failure}")

        page.on("requestfailed", request_failed)
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
            result = check_viewer(page, artifacts)
            assert not page_errors, "Browser page errors: " + " | ".join(page_errors)
            assert not graphics_errors, "Graphics errors: " + " | ".join(graphics_errors)
            assert not failed_local_requests, "Local request failures: " + " | ".join(failed_local_requests)
            result.update(
                passed=True,
                browser=browser.version,
                blocked_external_hosts=sorted(blocked_hosts),
                screenshots=str(artifacts),
            )
            print(json.dumps(result, indent=2))
        except Exception:
            page.screenshot(path=str(artifacts / "browser-failure.png"), full_page=True)
            print(json.dumps({
                "page_errors": page_errors,
                "graphics_errors": graphics_errors,
                "failed_local_requests": failed_local_requests,
            }))
            raise
        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    main()

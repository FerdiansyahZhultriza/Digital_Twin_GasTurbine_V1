from __future__ import annotations

import time

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import assistant, data_loader
from src.config import DEFAULT_DATA_FILE, PROJECT_ROOT


@pytest.fixture(autouse=True)
def isolate_app(monkeypatch):
    """Keep cached uploads isolated and ensure UI checks never call the assistant."""
    monkeypatch.setenv("OPENAI_API_KEY", "")

    def unexpected_assistant_call(*args, **kwargs):
        pytest.fail("App startup and playback must not call the assistant")

    monkeypatch.setattr(assistant, "ask_turbine_assistant", unexpected_assistant_call)
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def test_app_starts_and_advances_playback_without_widget_state_error(monkeypatch) -> None:
    app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=20).run()
    assert not app.exception, [error.message for error in app.exception]
    history = next(slider for slider in app.slider if slider.label == "Historical record")
    initial_index = history.value

    # End one playback tick at its rerun request so AppTest does not autoplay forever.
    # The next explicit run applies the pending record before recreating the slider.
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(st, "rerun", st.stop)
    app.toggle[0].set_value(True).run()
    assert not app.exception, [error.message for error in app.exception]
    app.toggle[0].set_value(False).run()
    assert not app.exception, [error.message for error in app.exception]
    history = next(slider for slider in app.slider if slider.label == "Historical record")
    assert history.value == initial_index + 1


@pytest.mark.parametrize("record_count", [0, 1], ids=["empty-dataset", "single-record"])
def test_short_dataset_has_no_uncaught_exception(monkeypatch, record_count: int) -> None:
    frame = data_loader.load_dataset(DEFAULT_DATA_FILE).iloc[:record_count].copy()
    monkeypatch.setattr(data_loader, "load_dataset", lambda source: frame)

    app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=20).run()

    assert not app.exception, [error.message for error in app.exception]
    if record_count == 0:
        assert len(app.error) == 1
        assert "no valid turbine readings" in app.error[0].value
    else:
        assert not app.error
        assert not any(slider.label == "Historical record" for slider in app.slider)
        assert app.session_state.record_index == 0

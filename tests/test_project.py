from __future__ import annotations

from src.assistant import ask_turbine_assistant
from src.config import DEFAULT_DATA_FILE
from src.data_loader import engineering_snapshot, load_dataset, summarize
from src.turbine_model import build_gas_turbine_figure


def test_packaged_dataset() -> None:
    frame = load_dataset(DEFAULT_DATA_FILE)
    summary = summarize(frame)

    assert summary.rows == 10_080
    assert summary.anomaly_count == 5_213
    assert summary.start.strftime("%d/%m/%Y %H:%M") == "15/08/2019 00:00"
    assert summary.end.strftime("%d/%m/%Y %H:%M") == "21/08/2019 23:59"


def test_engineering_context_is_compact_and_explicit() -> None:
    frame = load_dataset(DEFAULT_DATA_FILE)
    snapshot = engineering_snapshot(frame, 0)

    assert snapshot["architecture"]["compressor_stages"] == 17
    assert snapshot["architecture"]["combustors"] == 20
    assert snapshot["architecture"]["turbine_stages"] == 4
    assert snapshot["dataset"]["records"] == 10_080
    assert len(snapshot["limitations"]) >= 5


def test_offline_bilingual_assistant(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    frame = load_dataset(DEFAULT_DATA_FILE)
    result = ask_turbine_assistant("Cek vibrasi bearing", frame, 0)

    assert result.mode == "Local engineering rules"
    assert "bearing #1" in result.text.lower()
    assert "alarm/trip" in result.text.lower()


def test_turbine_model_animation() -> None:
    figure = build_gas_turbine_figure(3000)

    assert len(figure.frames) == 24
    assert any(trace.name == "17 compressor rotor stages" for trace in figure.data)
    assert any(trace.name == "4 turbine rotor stages" for trace in figure.data)
    assert sum(str(trace.name).startswith("Combustor ") for trace in figure.data) == 20


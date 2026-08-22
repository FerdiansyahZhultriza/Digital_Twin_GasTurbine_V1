from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from .config import ANOMALY_COLUMN, COL, DATE_COLUMN, REQUIRED_COLUMNS, UNIT_BY_COLUMN


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    anomaly_count: int
    anomaly_rate: float
    average_load: float
    maximum_exhaust_temp: float


def load_dataset(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Read and validate the semicolon-delimited gas-turbine CSV."""
    frame = pd.read_csv(source, sep=";", encoding="utf-8-sig")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Required CSV columns are missing: " + ", ".join(missing))

    frame = frame.copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], dayfirst=True, errors="coerce")
    numeric_columns = [column for column in REQUIRED_COLUMNS if column != DATE_COLUMN]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=[DATE_COLUMN, COL["speed"], COL["load"]]).sort_values(DATE_COLUMN)
    frame[ANOMALY_COLUMN] = frame[ANOMALY_COLUMN].fillna(0).astype(int).clip(0, 1)
    return frame.reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> DatasetSummary:
    if frame.empty:
        raise ValueError("The dataset has no valid turbine readings.")
    anomaly_count = int(frame[ANOMALY_COLUMN].eq(1).sum())
    return DatasetSummary(
        rows=len(frame),
        start=frame[DATE_COLUMN].min(),
        end=frame[DATE_COLUMN].max(),
        anomaly_count=anomaly_count,
        anomaly_rate=anomaly_count / len(frame),
        average_load=float(frame[COL["load"]].mean()),
        maximum_exhaust_temp=float(frame[COL["exhaust_temp"]].max()),
    )


def daily_summary(frame: pd.DataFrame) -> pd.DataFrame:
    daily = (
        frame.assign(Day=frame[DATE_COLUMN].dt.date)
        .groupby("Day", as_index=False)
        .agg(
            Average_Load_MW=(COL["load"], "mean"),
            Average_Exhaust_Temp_C=(COL["exhaust_temp"], "mean"),
            Anomaly_Labels=(ANOMALY_COLUMN, "sum"),
            Records=(ANOMALY_COLUMN, "size"),
        )
    )
    daily["Anomaly_Rate_Percent"] = daily["Anomaly_Labels"] / daily["Records"] * 100
    return daily


def z_score(value: float, series: pd.Series) -> float:
    deviation = float(series.std(ddof=0))
    if not np.isfinite(deviation) or deviation == 0:
        return 0.0
    return (float(value) - float(series.mean())) / deviation


def current_record_context(frame: pd.DataFrame, index: int) -> dict[str, object]:
    index = max(0, min(int(index), len(frame) - 1))
    row = frame.iloc[index]
    values: dict[str, object] = {"timestamp": row[DATE_COLUMN].isoformat(sep=" ")}
    for key, column in COL.items():
        value = row[column]
        values[key] = None if pd.isna(value) else round(float(value), 4)
        values[f"{key}_unit"] = UNIT_BY_COLUMN.get(column, "source unit")
    values["anomaly_label"] = int(row[ANOMALY_COLUMN])
    return values


def engineering_snapshot(frame: pd.DataFrame, index: int) -> dict[str, object]:
    """Compact context sent to the optional LLM; the whole CSV is never sent."""
    current = current_record_context(frame, index)
    summary = summarize(frame)
    statistics: dict[str, dict[str, float]] = {}
    for key, column in COL.items():
        series = frame[column].dropna()
        if series.empty:
            continue
        statistics[key] = {
            "min": round(float(series.min()), 4),
            "mean": round(float(series.mean()), 4),
            "max": round(float(series.max()), 4),
        }
    return {
        "architecture": {
            "compressor_stages": 17,
            "combustors": 20,
            "turbine_stages": 4,
            "arrangement": "single-shaft axial-flow conceptual cutaway",
        },
        "dataset": {
            "records": summary.rows,
            "period": f"{summary.start} to {summary.end}",
            "anomaly_label_1_count": summary.anomaly_count,
            "anomaly_label_1_rate_percent": round(summary.anomaly_rate * 100, 3),
        },
        "current_record": current,
        "seven_day_statistics": statistics,
        "limitations": [
            "Anomaly Prediction is an existing source label, not a model trained by this app.",
            "Bearing vibration and corrected fuel-flow engineering units are absent.",
            "OEM alarm and trip thresholds are absent.",
            "All TEMP fields are interpreted as degree Celsius and PRESS fields as bar by user instruction.",
            "Exhaust-duct and filter differential pressure magnitudes should be checked against instrument scaling.",
            "No individual compressor-stage or combustor temperature/pressure measurements are available.",
        ],
    }

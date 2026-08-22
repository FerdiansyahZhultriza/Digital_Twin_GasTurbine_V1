from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import pandas as pd

from .config import ANOMALY_COLUMN, COL, OPENAI_MODEL
from .data_loader import engineering_snapshot


@dataclass(frozen=True)
class AssistantResult:
    text: str
    mode: str


def _indonesian(question: str) -> bool:
    return bool(
        re.search(
            r"\b(apa|berapa|bagaimana|kenapa|mengapa|jelaskan|suhu|tekanan|"
            r"getaran|vibrasi|bantalan|beban|kondisi|analisis|kompresor|turbin|"
            r"pelumas|bahan bakar|cek|rata-rata|tanggal|anomali)\b",
            question.lower(),
        )
    )


def _number(value: float, digits: int = 1) -> str:
    return f"{float(value):,.{digits}f}"


def local_engineering_answer(question: str, frame: pd.DataFrame, index: int) -> str:
    """Useful offline fallback when no OpenAI key is configured."""
    q = question.lower()
    is_id = _indonesian(question)
    row = frame.iloc[max(0, min(int(index), len(frame) - 1))]
    timestamp = row["Date"].strftime("%d %b %Y %H:%M")
    mean = lambda key: float(frame[COL[key]].mean())
    anomaly_count = int(frame[ANOMALY_COLUMN].eq(1).sum())
    day_match = re.search(r"\b(1[5-9]|2[01])(?:\s*(?:aug(?:ust)?|agu(?:stus)?|/08))?\b", q)
    daily = frame
    scope = "15–21 Aug 2019"
    if day_match:
        day = int(day_match.group(1))
        daily = frame[frame["Date"].dt.day.eq(day)]
        scope = f"{day} Aug 2019"

    if re.search(r"anomal|abnormal|flag|fault|gangguan", q):
        flags = int(daily[ANOMALY_COLUMN].eq(1).sum())
        rate = flags / max(1, len(daily)) * 100
        if is_id:
            return (
                f"{scope}: {flags:,} dari {len(daily):,} rekaman memiliki Anomaly Prediction = 1 "
                f"({_number(rate)}%). Pada {timestamp}, label saat ini {int(row[ANOMALY_COLUMN])}, "
                f"beban {_number(row[COL['load']])} MW dan exhaust {_number(row[COL['exhaust_temp']])} °C. "
                "Label berasal dari CSV, bukan model baru; periksa alarm DCS, transien operasi, dan hasil inspeksi untuk root cause."
            )
        return (
            f"{scope}: {flags:,} of {len(daily):,} records have Anomaly Prediction = 1 "
            f"({_number(rate)}%). At {timestamp}, the current label is {int(row[ANOMALY_COLUMN])}, "
            f"load {_number(row[COL['load']])} MW, and exhaust {_number(row[COL['exhaust_temp']])} °C. "
            "These labels come from the CSV, not a newly trained model; check DCS alarms, operating transients, and inspection findings for root cause."
        )

    if re.search(r"bearing|vibrat|getar|vibrasi|bantalan", q):
        values = [row[COL["bearing_1"]], row[COL["bearing_2"]], row[COL["bearing_3"]]]
        averages = [mean("bearing_1"), mean("bearing_2"), mean("bearing_3")]
        if is_id:
            return (
                f"Pada {timestamp}: bearing #1 sisi turbin {_number(values[0])}, #2 sisi kompresor {_number(values[1])}, "
                f"dan #3 sisi turbin {_number(values[2])}. Rata-rata tujuh hari: {_number(averages[0])}, "
                f"{_number(averages[1])}, dan {_number(averages[2])}. Satuan serta alarm/trip limit tidak tersedia, "
                f"jadi severity belum dapat ditentukan. Periksa tren, alignment, balancing, clearance, dan tekanan pelumas {_number(row[COL['lube_oil_pressure']], 2)} bar."
            )
        return (
            f"At {timestamp}: bearing #1 turbine side {_number(values[0])}, #2 compressor side {_number(values[1])}, "
            f"and #3 turbine side {_number(values[2])}. Seven-day averages: {_number(averages[0])}, "
            f"{_number(averages[1])}, and {_number(averages[2])}. Units and alarm/trip thresholds are unavailable, "
            f"so severity cannot be established. Check trends, alignment, balance, clearances, and {_number(row[COL['lube_oil_pressure']], 2)} bar oil pressure."
        )

    if re.search(r"pressure|tekanan|press|fuel|bahan bakar|pelumas|oil", q):
        if is_id:
            return (
                f"Pada {timestamp}: fuel-gas supply {_number(row[COL['fuel_supply_pressure']])} bar, lube-oil supply "
                f"{_number(row[COL['lube_oil_pressure']], 2)} bar, exhaust duct {_number(row[COL['exhaust_duct_pressure']])} bar, "
                f"dan differential pressure filter {_number(row[COL['filter_diff_pressure']])} bar. Semua PRESS memakai bar sesuai instruksi. "
                "Nilai duct dan filter perlu verifikasi scaling. Tanda negatif fuel flow dipertahankan dari data sumber."
            )
        return (
            f"At {timestamp}: fuel-gas supply {_number(row[COL['fuel_supply_pressure']])} bar, lube-oil supply "
            f"{_number(row[COL['lube_oil_pressure']], 2)} bar, exhaust duct {_number(row[COL['exhaust_duct_pressure']])} bar, "
            f"and filter differential {_number(row[COL['filter_diff_pressure']])} bar. All PRESS fields use bar as instructed. "
            "Duct and filter magnitudes need scaling verification. The negative fuel-flow sign is retained from the source."
        )

    if re.search(r"compressor|kompresor|discharge|filter|inlet", q):
        rise = row[COL["compressor_outlet_temp"]] - row[COL["compressor_inlet_temp"]]
        if is_id:
            return (
                f"Kompresor memiliki 17 stage geometri. Pada {timestamp}, inlet {_number(row[COL['compressor_inlet_temp']])} °C, "
                f"outlet {_number(row[COL['compressor_outlet_temp']])} °C, sehingga kenaikan temperatur {_number(rise)} °C. "
                "CSV hanya menyediakan pengukuran tingkat mesin, bukan tekanan atau temperatur setiap stage."
            )
        return (
            f"The compressor has 17 geometric stages. At {timestamp}, inlet is {_number(row[COL['compressor_inlet_temp']])} °C "
            f"and outlet {_number(row[COL['compressor_outlet_temp']])} °C, a measured rise of {_number(rise)} °C. "
            "The CSV contains machine-level measurements, not individual stage pressures or temperatures."
        )

    if re.search(r"exhaust|temperature|temperatur|suhu|combust|pembakar|thermal|cooling", q):
        minimum = float(frame[COL["exhaust_temp"]].min())
        maximum = float(frame[COL["exhaust_temp"]].max())
        if is_id:
            return (
                f"Pada {timestamp}: exhaust {_number(row[COL['exhaust_temp']])} °C, outlet kompresor "
                f"{_number(row[COL['compressor_outlet_temp']])} °C, cooling-air rotor {_number(row[COL['rotor_cooling_temp']])} °C, "
                f"dan lube oil {_number(row[COL['lube_oil_temp']])} °C. Exhaust tujuh hari: {_number(minimum)}–{_number(maximum)} °C. "
                "Tidak ada temperatur individual untuk 20 combustor."
            )
        return (
            f"At {timestamp}: exhaust {_number(row[COL['exhaust_temp']])} °C, compressor outlet "
            f"{_number(row[COL['compressor_outlet_temp']])} °C, rotor cooling air {_number(row[COL['rotor_cooling_temp']])} °C, "
            f"and lube oil {_number(row[COL['lube_oil_temp']])} °C. Seven-day exhaust range: {_number(minimum)}–{_number(maximum)} °C. "
            "No individual temperatures are available for the 20 combustors."
        )

    if re.search(r"load|beban|mw|output|power|daya|average|rata", q):
        average_load = float(daily[COL["load"]].mean())
        if is_id:
            return (
                f"{scope}: rata-rata beban {_number(average_load)} MW. Pada {timestamp}, beban {_number(row[COL['load']])} MW "
                f"pada {_number(row[COL['speed']], 0)} RPM. Interpretasikan perubahan beban bersama temperatur exhaust, fuel flow, dan mode operasi."
            )
        return (
            f"{scope}: average load {_number(average_load)} MW. At {timestamp}, load is {_number(row[COL['load']])} MW "
            f"at {_number(row[COL['speed']], 0)} RPM. Interpret load changes alongside exhaust temperature, fuel flow, and operating mode."
        )

    if is_id:
        return (
            f"GT-01 pada {timestamp}: beban {_number(row[COL['load']])} MW, shaft {_number(row[COL['speed']], 0)} RPM, "
            f"inlet/outlet kompresor {_number(row[COL['compressor_inlet_temp']])}/{_number(row[COL['compressor_outlet_temp']])} °C, "
            f"exhaust {_number(row[COL['exhaust_temp']])} °C, fuel gas {_number(row[COL['fuel_supply_pressure']])} bar, "
            f"dan label anomali {int(row[ANOMALY_COLUMN])}. Dataset memiliki {len(frame):,} rekaman dan {anomaly_count:,} label 1."
        )
    return (
        f"GT-01 at {timestamp}: load {_number(row[COL['load']])} MW, shaft {_number(row[COL['speed']], 0)} RPM, "
        f"compressor inlet/outlet {_number(row[COL['compressor_inlet_temp']])}/{_number(row[COL['compressor_outlet_temp']])} °C, "
        f"exhaust {_number(row[COL['exhaust_temp']])} °C, fuel gas {_number(row[COL['fuel_supply_pressure']])} bar, "
        f"and anomaly label {int(row[ANOMALY_COLUMN])}. The dataset contains {len(frame):,} records and {anomaly_count:,} label-1 rows."
    )


def ask_turbine_assistant(
    question: str,
    frame: pd.DataFrame,
    index: int,
    history: list[dict[str, str]] | None = None,
) -> AssistantResult:
    """Use OpenAI when configured, otherwise return the deterministic local answer."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return AssistantResult(local_engineering_answer(question, frame, index), "Local engineering rules")

    try:
        from openai import OpenAI

        snapshot = engineering_snapshot(frame, index)
        recent_history = (history or [])[-6:]
        prompt = (
            "ENGINEERING SNAPSHOT (authoritative for numerical claims):\n"
            + json.dumps(snapshot, ensure_ascii=False, indent=2)
            + "\n\nRECENT CHAT:\n"
            + json.dumps(recent_history, ensure_ascii=False)
            + "\n\nOPERATOR QUESTION:\n"
            + question
        )
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=(
                "You are a bilingual gas-turbine maintenance assistant for an industrial pilot digital twin. "
                "Answer in the operator's language (English or Bahasa Indonesia). Ground every numerical claim in the supplied snapshot. "
                "Treat Anomaly Prediction as a pre-existing binary CSV label, never as your own diagnosis. "
                "Do not invent OEM alarm limits, vibration units, fuel-flow units, individual stage sensors, or combustor temperatures. "
                "Clearly distinguish observation, inference, and recommended verification. Keep the answer practical and concise."
            ),
            input=prompt,
        )
        text = (response.output_text or "").strip()
        if not text:
            raise RuntimeError("The API returned an empty response.")
        return AssistantResult(text, f"OpenAI Responses API · {OPENAI_MODEL}")
    except Exception as error:  # The offline fallback keeps the pilot usable.
        fallback = local_engineering_answer(question, frame, index)
        return AssistantResult(
            fallback + f"\n\n_API unavailable; local fallback used ({type(error).__name__})._",
            "Local fallback",
        )

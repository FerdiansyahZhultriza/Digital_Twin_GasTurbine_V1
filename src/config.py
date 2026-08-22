from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "raw_data_gas_turbine.csv"

DATE_COLUMN = "Date"
ANOMALY_COLUMN = "Anomaly Prediction"

COL = {
    "speed": "GT SPEED",
    "load": "Actual Load (MW)",
    "compressor_inlet_temp": "COMP INLET AIR TEMP",
    "compressor_outlet_temp": "COMP OUTLET AIR TEMP",
    "exhaust_duct_pressure": "EXHAUST GAS DUCT PRESS",
    "fuel_flow": "GT FUEL GAS FLOW (AFTER CORRECT)",
    "fuel_supply_pressure": "FUEL GAS SUPPLY PRESS",
    "exhaust_temp": "EXHAUST GAS AVG TEMP(EXT)",
    "bearing_1": "No.1 BEARING ROTOR VIBRATION (TURB.SIDE) (X)",
    "bearing_2": "No.2 BEARING ROTOR VIBRATION (COMP.SIDE) (X)",
    "bearing_3": "No.3 BEARING ROTOR VIBRATION (TURB.SIDE) (X)",
    "rotor_cooling_temp": "ROTOR COOLING AIR TEMP AVE.",
    "lube_oil_pressure": "LUBE OIL SUPPLY PRESS",
    "lube_oil_temp": "LUBE OIL TEMP",
    "filter_diff_pressure": "INLET AIR FILTER DIFF PRESS",
}

TEMPERATURE_COLUMNS = {
    COL["compressor_inlet_temp"],
    COL["compressor_outlet_temp"],
    COL["exhaust_temp"],
    COL["rotor_cooling_temp"],
    COL["lube_oil_temp"],
}

PRESSURE_COLUMNS = {
    COL["exhaust_duct_pressure"],
    COL["fuel_supply_pressure"],
    COL["lube_oil_pressure"],
    COL["filter_diff_pressure"],
}

UNIT_BY_COLUMN = {
    COL["speed"]: "RPM",
    COL["load"]: "MW",
    **{column: "°C" for column in TEMPERATURE_COLUMNS},
    **{column: "bar" for column in PRESSURE_COLUMNS},
    COL["fuel_flow"]: "source unit",
    COL["bearing_1"]: "source unit",
    COL["bearing_2"]: "source unit",
    COL["bearing_3"]: "source unit",
    ANOMALY_COLUMN: "binary label",
}

REQUIRED_COLUMNS = [DATE_COLUMN, *COL.values(), ANOMALY_COLUMN]

# A cost-conscious default for a pilot project. Change this in .env if desired.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
MAX_CONTEXT_ROWS = int(os.getenv("MAX_CONTEXT_ROWS", "12"))

APP_TITLE = "Gas Turbine Digital Twin"
APP_SUBTITLE = "17-stage compressor · 20 can-annular combustors · 4-stage turbine"


![alt text](image.png)

![alt text](image-1.png)

![alt text](image-2.png)

# Gas Turbine Digital Twin — Python Pilot Project

This is a first pilot project for an industrial M701-F gas-turbine digital twin. It combines historical operating data, an interactive rotating 3D cutaway, trend analysis, source anomaly labels, and a bilingual English/Bahasa Indonesia maintenance assistant.

The packaged machine architecture contains:

- 17 axial-compressor rotor stages with stationary stator rows
- 20 can-annular combustors arranged circumferentially
- 4 turbine rotor stages with stationary stator rows
- One common rotating shaft, rotor drums, bearings, inlet casing, combustion casing, turbine casing, and exhaust diffuser

The geometry is a conceptual engineering visualization. It is not OEM CAD, a manufacturing drawing, or dimensionally accurate equipment geometry.

## Main features

- Runs locally in Visual Studio Code using Streamlit.
- Includes the original `raw_data_gas_turbine.csv` with 10,080 one-minute records.
- Interprets `TEMP` measurements as °C and `PRESS` measurements as bar.
- Replays every historical record with an optional automatic playback mode.
- Displays generator load, shaft speed, compressor discharge temperature, exhaust temperature, fuel-gas pressure, and the existing anomaly label.
- Provides load, thermal, bearing-vibration, daily-performance, and anomaly-label trends.
- Shows every original CSV channel in an instrument register.
- Rotates the compressor and turbine blades while the combustors, stators, and casing remain stationary.
- Answers questions in English or Bahasa Indonesia.
- Works without an API key using deterministic engineering rules.
- Optionally uses the OpenAI Responses API when `OPENAI_API_KEY` is configured.




If the browser does not open, copy the local address shown in the terminal, normally:

```text
http://localhost:8501
```

Never paste an API key directly into Python source code. The `.gitignore` file excludes `.env`, but if a key is ever pushed to GitHub, revoke it immediately and create a new key.


### The OpenAI assistant uses local fallback

Check that `.env` exists, `OPENAI_API_KEY` has a valid active key, your API project can access the selected model, billing is enabled, and Streamlit was restarted after editing `.env`.

## Project status

This repository is intentionally a pilot and learning project. Future development can add a trained anomaly-detection pipeline, OEM-configurable thresholds, database integration, per-sensor quality flags, role-based access, work-order recommendations, and connection to a validated historian API.


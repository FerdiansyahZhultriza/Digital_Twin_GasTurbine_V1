from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.assistant import ask_turbine_assistant
from src.config import (
    ANOMALY_COLUMN,
    APP_SUBTITLE,
    APP_TITLE,
    COL,
    DATE_COLUMN,
    DEFAULT_DATA_FILE,
    OPENAI_MODEL,
    UNIT_BY_COLUMN,
)
from src.data_loader import daily_summary, load_dataset, summarize, z_score
from src.turbine_model import render_turbine

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --dt-bg:#070b10; --dt-panel:#0c1118; --dt-raised:#111820;
        --dt-border:#1d2a33; --dt-text:#e9f0f2; --dt-muted:#80919a;
        --dt-cyan:#54dbe7; --dt-amber:#f3a44e; --dt-red:#f17372; --dt-green:#70d7a5;
    }
    .stApp { background:var(--dt-bg); color:var(--dt-text); }
    [data-testid="stSidebar"] { background:#090e14; border-right:1px solid var(--dt-border); }
    [data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:1680px; padding-top:1.6rem; padding-bottom:3rem; }
    h1,h2,h3 { letter-spacing:-.025em; }
    .dt-header { display:flex; justify-content:space-between; align-items:flex-end; gap:18px; padding:8px 0 17px; border-bottom:1px solid var(--dt-border); }
    .dt-eyebrow { color:var(--dt-cyan); font-size:.68rem; font-weight:650; letter-spacing:.16em; text-transform:uppercase; }
    .dt-title { margin:6px 0 2px; color:var(--dt-text); font-size:2rem; font-weight:540; }
    .dt-subtitle { color:var(--dt-muted); font-size:.82rem; }
    .dt-connected { display:flex; align-items:center; gap:8px; color:#a5cdb8; font-size:.68rem; letter-spacing:.08em; white-space:nowrap; }
    .dt-connected i { width:7px; height:7px; border-radius:50%; background:var(--dt-green); box-shadow:0 0 12px #70d7a588; }
    .dt-section { margin:16px 0 8px; color:#8fa0a8; font-size:.64rem; font-weight:650; letter-spacing:.14em; text-transform:uppercase; }
    .dt-status { display:inline-flex; align-items:center; gap:7px; padding:7px 10px; border:1px solid #2f503f; background:#0d1913; color:#8fd5aa; font-size:.67rem; letter-spacing:.06em; }
    .dt-status.warn { border-color:#67442c; background:#1b130d; color:#f0ad6e; }
    .dt-status i { width:7px; height:7px; border-radius:50%; background:currentColor; }
    [data-testid="stMetric"] { min-height:112px; padding:14px 14px 11px; border:1px solid var(--dt-border); background:var(--dt-panel); }
    [data-testid="stMetricLabel"] { color:#93a3aa; font-size:.7rem; letter-spacing:.04em; }
    [data-testid="stMetricValue"] { color:var(--dt-text); font-size:1.55rem; }
    [data-testid="stMetricDelta"] { font-size:.67rem; }
    [data-testid="stTabs"] button { color:#9aabb2; font-size:.75rem; letter-spacing:.04em; }
    [data-testid="stTabs"] button[aria-selected="true"] { color:var(--dt-cyan); }
    [data-testid="stPlotlyChart"] { border:1px solid var(--dt-border); background:var(--dt-panel); }
    [data-testid="stDataFrame"] { border:1px solid var(--dt-border); }
    [data-testid="stChatMessage"] { border:1px solid var(--dt-border); background:var(--dt-panel); }
    .dt-panel { padding:15px; border:1px solid var(--dt-border); background:var(--dt-panel); }
    .dt-panel h4 { margin:0 0 10px; font-size:.78rem; font-weight:600; }
    .dt-panel p { margin:5px 0; color:#aebbc1; font-size:.74rem; line-height:1.55; }
    .dt-reading { display:flex; justify-content:space-between; gap:12px; padding:8px 0; border-bottom:1px solid #18232b; }
    .dt-reading:last-child { border-bottom:0; }
    .dt-reading span { color:#91a0a7; font-size:.69rem; }
    .dt-reading strong { color:#e0e9e9; font-family:monospace; font-size:.71rem; font-weight:500; white-space:nowrap; }
    .dt-advisory { margin:10px 0; padding:12px 13px; border-left:3px solid var(--dt-amber); background:#17120e; color:#ccb49e; font-size:.72rem; line-height:1.55; }
    .dt-caption { color:#74858d; font-size:.66rem; line-height:1.5; }
    .dt-architecture { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:10px; }
    .dt-architecture div { padding:12px; border:1px solid var(--dt-border); background:var(--dt-raised); text-align:center; }
    .dt-architecture strong { display:block; color:var(--dt-text); font-size:1.4rem; }
    .dt-architecture span { color:#8fa0a8; font-size:.65rem; }
    @media(max-width:700px) {
        .dt-header { align-items:flex-start; flex-direction:column; }
        .dt-title { font-size:1.55rem; }
        .dt-architecture { grid-template-columns:1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_default_data(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return load_dataset(path)


def number(value: float, digits: int = 1) -> str:
    return f"{float(value):,.{digits}f}"


def process_trend(
    frame: pd.DataFrame,
    index: int,
    columns: list[str],
    names: list[str],
    colors: list[str],
    title: str,
    unit: str,
    hours: int = 12,
) -> go.Figure:
    points = max(30, hours * 60)
    start = max(0, index - points)
    window = frame.iloc[start : index + 1]
    if len(window) > 600:
        window = window.iloc[:: max(1, len(window) // 600)]

    figure = go.Figure()
    for column, name, color in zip(columns, names, colors, strict=True):
        figure.add_trace(
            go.Scatter(
                x=window[DATE_COLUMN],
                y=window[column],
                mode="lines",
                name=name,
                line={"color": color, "width": 1.8},
                hovertemplate=f"%{{x|%d %b %H:%M}}<br>{name}: %{{y:.2f}} {unit}<extra></extra>",
            )
        )
    flagged = window[window[ANOMALY_COLUMN].eq(1)]
    if not flagged.empty:
        first_column = columns[0]
        figure.add_trace(
            go.Scatter(
                x=flagged[DATE_COLUMN],
                y=flagged[first_column],
                mode="markers",
                name="Source label = 1",
                marker={"size": 4, "color": "#f17372", "opacity": 0.72},
                hovertemplate="%{x|%d %b %H:%M}<br>Source anomaly label = 1<extra></extra>",
            )
        )
    figure.update_layout(
        title={"text": title, "font": {"size": 13, "color": "#dce7e8"}, "x": 0.015},
        height=330,
        margin={"l": 20, "r": 15, "t": 48, "b": 20},
        paper_bgcolor="#0c1118",
        plot_bgcolor="#0c1118",
        font={"family": "Segoe UI, Arial", "size": 10, "color": "#81929b"},
        legend={"orientation": "h", "y": 1.08, "x": 0.32},
        hovermode="x unified",
        xaxis={"gridcolor": "#1b2730", "zeroline": False},
        yaxis={"title": unit, "gridcolor": "#1b2730", "zeroline": False},
    )
    return figure


with st.sidebar:
    st.markdown("### ⚙️ Block 2 GT 2.1")
    st.caption("M701F Gas Turbine Digital Twin")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Replace the packaged dataset",
        type=["csv"],
        help="Expected delimiter: semicolon (;). The original CSV remains packaged in data/.",
    )
    st.markdown("#### Playback")
    auto_play = st.toggle("Auto-play one-minute records", value=False)
    playback_delay = st.select_slider("Playback interval", options=[0.5, 1.0, 2.0, 5.0], value=1.0, format_func=lambda value: f"{value:g} s")
    st.markdown("---")
    st.markdown("#### Assistant")
    if os.getenv("OPENAI_API_KEY", "").strip():
        st.success(f"OpenAI enabled · {OPENAI_MODEL}")
    else:
        st.info("Offline engineering rules")
        st.caption("Add OPENAI_API_KEY to .env to enable the Responses API.")
    st.markdown("---")
    st.caption("TEMP → °C · PRESS → bar")
    st.caption("Detailed 3D engineering visualization. Geometry and dimensions are illustrative; OEM CAD is not supplied.")


try:
    if uploaded_file is not None:
        data = load_dataset(uploaded_file)
        source_label = uploaded_file.name
    else:
        data = read_default_data(str(DEFAULT_DATA_FILE), DEFAULT_DATA_FILE.stat().st_mtime)
        source_label = DEFAULT_DATA_FILE.name
    summary = summarize(data)
except Exception as error:
    st.error(f"Could not load the turbine dataset: {error}")
    st.stop()

st.session_state.setdefault("record_index", min(4320, len(data) - 1))
pending_record_index = st.session_state.pop("pending_record_index", None)
if pending_record_index is not None:
    st.session_state.record_index = pending_record_index
st.session_state.record_index = max(0, min(st.session_state.record_index, len(data) - 1))

st.markdown(
    f"""
    <div class="dt-header">
      <div>
        <div class="dt-eyebrow">Block 2 AI Assistant· GT 2.1· Gas Turbine Digital Twin</div>
        <div class="dt-title">{APP_TITLE}</div>
        <div class="dt-subtitle">{APP_SUBTITLE}</div>
      </div>
      <div class="dt-connected"><i></i> DATASET CONNECTED · {len(data):,} RECORDS</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="dt-section">Historical record selection</div>', unsafe_allow_html=True)
if len(data) > 1:
    selected_index = st.slider(
        "Historical record",
        min_value=0,
        max_value=len(data) - 1,
        format="%d",
        label_visibility="collapsed",
        key="record_index",
    )
else:
    selected_index = 0
    st.caption("The dataset contains one valid record.")
row = data.iloc[selected_index]
timestamp = row[DATE_COLUMN]

header_left, header_right = st.columns([4, 1])
with header_left:
    st.caption(
        f"{timestamp:%d %B %Y · %H:%M}  |  Record {selected_index + 1:,} of {len(data):,}  |  Source: {source_label}"
    )
with header_right:
    status_class = "warn" if int(row[ANOMALY_COLUMN]) == 1 else ""
    status_text = "SOURCE ANOMALY FLAG = 1" if int(row[ANOMALY_COLUMN]) == 1 else "SOURCE LABEL = 0"
    st.markdown(f'<div class="dt-status {status_class}"><i></i>{status_text}</div>', unsafe_allow_html=True)

metric_columns = st.columns(6)
metric_values = [
    ("Generator load", number(row[COL["load"]]), "MW", number(row[COL["load"]] - summary.average_load) + " vs 7-day mean"),
    ("Shaft speed", number(row[COL["speed"]], 0), "RPM", number(row[COL["speed"]] - 3000, 0) + " vs nominal"),
    ("Compressor discharge", number(row[COL["compressor_outlet_temp"]]), "°C", number(row[COL["compressor_outlet_temp"]] - data[COL["compressor_outlet_temp"]].mean()) + " vs mean"),
    ("Exhaust average", number(row[COL["exhaust_temp"]]), "°C", number(row[COL["exhaust_temp"]] - data[COL["exhaust_temp"]].mean()) + " vs mean"),
    ("Fuel-gas supply", number(row[COL["fuel_supply_pressure"]]), "bar", number(row[COL["fuel_supply_pressure"]] - data[COL["fuel_supply_pressure"]].mean(), 2) + " vs mean"),
    ("Anomaly prediction", number(row[ANOMALY_COLUMN], 0), "label", f"{summary.anomaly_rate * 100:.1f}% dataset rate"),
]
for container, (label, value, unit, delta) in zip(metric_columns, metric_values, strict=True):
    with container:
        st.metric(label, f"{value} {unit}", delta)

overview_tab, trends_tab, data_tab, assistant_tab, methodology_tab = st.tabs(
    ["3D TWIN", "OPERATING TRENDS", "INSTRUMENT DATA", "BILINGUAL ASSISTANT", "METHODOLOGY"]
)

with overview_tab:
    left, right = st.columns([3.3, 1])
    with left:
        render_turbine(row[COL["speed"]])
    with right:
        st.markdown('<div class="dt-section">Machine architecture</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="dt-architecture">
              <div><strong>17</strong><span>COMPRESSOR STAGES</span></div>
              <div><strong>20</strong><span>COMBUSTORS</span></div>
              <div><strong>4</strong><span>TURBINE STAGES</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="dt-section">Current thermal state</div>', unsafe_allow_html=True)
        thermal_rows = [
            ("Compressor inlet", row[COL["compressor_inlet_temp"]], "°C"),
            ("Compressor discharge", row[COL["compressor_outlet_temp"]], "°C"),
            ("Exhaust average", row[COL["exhaust_temp"]], "°C"),
            ("Rotor cooling air", row[COL["rotor_cooling_temp"]], "°C"),
            ("Lube-oil temperature", row[COL["lube_oil_temp"]], "°C"),
        ]
        reading_html = "".join(
            f'<div class="dt-reading"><span>{label}</span><strong>{number(value)} {unit}</strong></div>'
            for label, value, unit in thermal_rows
        )
        st.markdown(f'<div class="dt-panel">{reading_html}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="dt-advisory">Drag to orbit · scroll to zoom. Switch between cutaway and exterior, or inspect a section. Rotor speed is slowed for inspection.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="dt-section">Rotor dynamics</div>', unsafe_allow_html=True)
    st.caption("The three vibration channels retain their source labels; they are not mapped to the model's two physical bearing supports.")
    bearing_columns = st.columns(3)
    for container, (label, key) in zip(
        bearing_columns,
        [("Bearing #1 · turbine side", "bearing_1"), ("Bearing #2 · compressor side", "bearing_2"), ("Bearing #3 · turbine side", "bearing_3")],
        strict=True,
    ):
        with container:
            value = float(row[COL[key]])
            st.metric(label, number(value), f"z = {z_score(value, data[COL[key]]):+.2f} vs dataset")
            st.caption("Source unit; OEM limit not supplied")

with trends_tab:
    trend_hours = st.select_slider("Trailing window", options=[1, 3, 6, 12, 24, 48], value=12, format_func=lambda value: f"{value} hours")
    first_chart, second_chart = st.columns(2)
    with first_chart:
        st.plotly_chart(
            process_trend(
                data,
                selected_index,
                [COL["load"]],
                ["Generator load"],
                ["#54dbe7"],
                "Generator load and source anomaly labels",
                "MW",
                trend_hours,
            ),
            width="stretch",
            config={"displaylogo": False},
        )
    with second_chart:
        st.plotly_chart(
            process_trend(
                data,
                selected_index,
                [COL["compressor_outlet_temp"], COL["exhaust_temp"]],
                ["Compressor outlet", "Exhaust average"],
                ["#d7ba75", "#f3a44e"],
                "Hot-section and compressor temperatures",
                "°C",
                trend_hours,
            ),
            width="stretch",
            config={"displaylogo": False},
        )

    daily = daily_summary(data)
    daily_figure = go.Figure()
    daily_figure.add_bar(x=daily["Day"], y=daily["Average_Load_MW"], name="Average load (MW)", marker_color="#54dbe7")
    daily_figure.add_scatter(x=daily["Day"], y=daily["Anomaly_Rate_Percent"], name="Label-1 rate (%)", mode="lines+markers", line={"color": "#f17372", "width": 2}, yaxis="y2")
    daily_figure.update_layout(
        height=340,
        title={"text": "Daily performance and anomaly-label rate", "font": {"size": 13}},
        margin={"l": 20, "r": 20, "t": 55, "b": 25},
        paper_bgcolor="#0c1118",
        plot_bgcolor="#0c1118",
        font={"color": "#82939b", "size": 10},
        xaxis={"gridcolor": "#1b2730"},
        yaxis={"title": "Average load (MW)", "gridcolor": "#1b2730"},
        yaxis2={"title": "Label-1 rate (%)", "overlaying": "y", "side": "right", "range": [0, 100]},
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(daily_figure, width="stretch", config={"displaylogo": False})

with data_tab:
    st.markdown('<div class="dt-section">Complete current instrument register</div>', unsafe_allow_html=True)
    register_rows: list[dict[str, object]] = []
    for column in data.columns:
        value = row[column]
        if column == DATE_COLUMN:
            formatted = timestamp.strftime("%d/%m/%Y %H:%M")
        elif pd.isna(value):
            formatted = "—"
        elif column == ANOMALY_COLUMN:
            formatted = str(int(value))
        else:
            formatted = number(float(value), 2 if "LUBE OIL SUPPLY PRESS" in column else 1)
        register_rows.append({"Measurement": column, "Value": formatted, "Unit": UNIT_BY_COLUMN.get(column, "—")})
    st.dataframe(pd.DataFrame(register_rows), width="stretch", hide_index=True, height=610)

    st.markdown('<div class="dt-section">Data-quality advisories</div>', unsafe_allow_html=True)
    st.warning(
        "All TEMP columns are treated as °C and PRESS columns as bar. Exhaust-duct and inlet-filter differential-pressure magnitudes look unusual if interpreted literally as bar; verify the source instrument scaling."
    )
    st.warning(
        "Corrected fuel-gas flow is negative in the source and its engineering unit is not defined. Bearing-vibration units and OEM alarm/trip thresholds are also not supplied."
    )
    cleaned_csv = data.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button("Download validated CSV", cleaned_csv, "validated_raw_data_gas_turbine.csv", "text/csv")

with assistant_tab:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Dataset ready. Ask about the selected operating condition, compressor temperatures, bearing vibration, "
                    "pressures, anomaly labels, or a specific day. Anda juga dapat bertanya dalam Bahasa Indonesia."
                ),
                "mode": "System",
            }
        ]
    top_assistant, top_clear = st.columns([5, 1])
    with top_assistant:
        mode_text = f"OpenAI Responses API · {OPENAI_MODEL}" if os.getenv("OPENAI_API_KEY", "").strip() else "Offline local engineering rules"
        st.caption(f"Active mode: {mode_text}. Only compact engineering context is sent when OpenAI is enabled—not the full CSV.")
    with top_clear:
        if st.button("Clear conversation", width="stretch"):
            st.session_state.chat_messages = []
            st.rerun()

    quick_questions = st.columns(4)
    presets = ["Analyze anomaly on 20 Aug", "Cek vibrasi bearing", "Explain compressor condition", "Berapa tekanan fuel gas?"]
    pending_question = None
    for container, preset in zip(quick_questions, presets, strict=True):
        with container:
            if st.button(preset, width="stretch"):
                pending_question = preset

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("mode") and message["role"] == "assistant":
                st.caption(message["mode"])

    typed_question = st.chat_input("Ask in English or Bahasa Indonesia…")
    question = typed_question or pending_question
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        history = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.chat_messages[-7:]
        ]
        with st.spinner("Analyzing the selected turbine record…"):
            result = ask_turbine_assistant(question, data, selected_index, history)
        st.session_state.chat_messages.append({"role": "assistant", "content": result.text, "mode": result.mode})
        st.rerun()

with methodology_tab:
    st.markdown("### What this pilot project does")
    st.markdown(
        """
        - Loads and validates all **10,080 one-minute records** from the packaged semicolon-delimited CSV.
        - Replays historical readings with an interactive, detailed gas-turbine visualization and smoothly animated shaft rotation.
        - Shows **17 compressor rotor stages**, **20 circumferential can-annular combustors**, **4 turbine rotor stages**, solid airfoil blades, and **2 bearing supports**.
        - Provides cutaway and exterior views, section inspection, and realistic metal materials and lighting. Shaft animation is slowed for inspection; the RPM reading preserves the selected record.
        - Uses the existing `Anomaly Prediction` column as a source label. It does **not** retrain or claim a new machine-learning model.
        - Answers offline using deterministic data-grounded rules. When an API key is present, it optionally calls the OpenAI Responses API with a compact snapshot.
        """
    )
    st.markdown("### Important engineering boundaries")
    st.markdown(
        """
        - The 3D geometry is an illustrative engineering visualization. OEM CAD, exact dimensions, blade profiles, and manufacturing details were not supplied.
        - The three source vibration channels are not assigned to the two bearing supports shown in the visualization.
        - The CSV contains machine-level measurements, not individual values for each compressor stage or combustor.
        - No vibration unit, fuel-flow unit, alarm limit, trip limit, or OEM acceptance criterion was included.
        - A label of `1` is treated as *flagged by the source*, not automatically as a confirmed mechanical fault.
        - The application is read-only and does not connect to or control a DCS, PLC, or protection system.
        """
    )

st.caption(
    f"Pilot dataset: {summary.start:%d %b %Y %H:%M}–{summary.end:%d %b %Y %H:%M} · "
    f"{summary.rows:,} records · {summary.anomaly_count:,} label-1 rows · source-unit validation required"
)

if auto_play and len(data) > 1:
    time.sleep(float(playback_delay))
    st.session_state.pending_record_index = (selected_index + 1) % len(data)
    st.rerun()

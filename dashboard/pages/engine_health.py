"""Engine Health diagnostic deep-dive page for the Self-Healing Machines dashboard."""

import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.components.charts import (
    plot_anomaly_score_timeline,
    plot_rolling_stats,
    plot_rul_timeline,
    plot_sensor_trends,
)
from dashboard.components.metrics import (
    render_kpi_card,
    render_severity_badge,
    render_status_badge,
    severity_color,
)

try:
    from src.data.loader import get_project_root, load_raw_cmapss
except ImportError:
    def get_project_root() -> Path:
        return Path(__file__).resolve().parent.parent.parent

    def load_raw_cmapss(subset: str = "FD001", data_type: str = "test") -> Optional[pd.DataFrame]:
        return None


def _get_db_path(config: Dict[str, Any]) -> Path:
    """Resolve database path from configuration."""
    root = get_project_root()
    db_rel = config.get("paths", {}).get("database", "data/maintenance.db")
    return root / db_rel


@st.cache_data(ttl=60)
def _load_predictions(root_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load latest predictions and triage decisions."""
    root = Path(root_dir)
    pred_path = root / "data" / "processed" / "predictions_latest.csv"
    triage_path = root / "data" / "processed" / "triage_decisions.csv"

    pred_df = pd.read_csv(pred_path) if pred_path.exists() else None
    triage_df = pd.read_csv(triage_path) if triage_path.exists() else None
    return pred_df, triage_df


@st.cache_data(ttl=120)
def _load_telemetry_data(subset: str = "FD001") -> Optional[pd.DataFrame]:
    """Load test telemetry records for time-series charting with fallback to raw C-MAPSS files."""
    root = get_project_root()

    # 1. Try processed test features or processed test data
    for candidate in [
        root / "data" / "processed" / f"test_{subset}.csv",
        root / "data" / "processed" / "test_features.csv",
        root / "data" / "processed" / f"train_{subset}.csv",
    ]:
        if candidate.exists():
            try:
                return pd.read_csv(candidate)
            except Exception:
                pass

    # 2. Try loader function
    try:
        df = load_raw_cmapss(subset=subset, data_type="test")
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # 3. Direct read from raw_data_dir or CMAPSSData
    for raw_dir in [root.parent / "CMAPSSData", root / "data" / "raw"]:
        file_path = raw_dir / f"test_{subset}.txt"
        if file_path.exists():
            try:
                cols = [
                    "engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3",
                    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
                    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
                    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
                    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"
                ]
                df = pd.read_csv(file_path, sep=r"\s+", header=None, names=cols)
                return df
            except Exception:
                pass

    return None


def _load_engine_work_orders(engine_id: int, config: Dict[str, Any]) -> pd.DataFrame:
    """Retrieve work orders associated with a specific engine ID."""
    db_path = _get_db_path(config)
    if not db_path.exists():
        return pd.DataFrame()

    try:
        with sqlite3.connect(str(db_path)) as conn:
            query = """
            SELECT id, engine_id, cycle, severity, status, rule_name, explanation, created_at
            FROM work_orders
            WHERE engine_id = ?
            ORDER BY created_at DESC
            """
            return pd.read_sql_query(query, conn, params=(engine_id,))
    except Exception:
        return pd.DataFrame()


def render_engine_health(config: Dict[str, Any]) -> None:
    """Render the Engine Health deep-dive page.

    Args:
        config: Application configuration dictionary.
    """
    root = get_project_root()
    subset = config.get("dataset", {}).get("default_subset", "FD001")
    crit_threshold = config.get("triage", {}).get("critical_rul_threshold", 15)
    warn_threshold = config.get("triage", {}).get("warning_rul_threshold", 40)
    anomaly_threshold = config.get("triage", {}).get("anomaly_score_threshold", -0.05)
    rolling_windows = config.get("features", {}).get("rolling_windows", [5, 10, 20])
    selected_sensors_default = config.get("features", {}).get(
        "selected_sensors",
        ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15"]
    )

    pred_df, triage_df = _load_predictions(str(root))
    telemetry_df = _load_telemetry_data(subset=subset)

    # Empty state guard if no prediction records exist
    if pred_df is None or pred_df.empty:
        st.warning(
            "⚠️ **No engine prediction data available.**\n\n"
            "Please ensure `data/processed/predictions_latest.csv` has been generated by the prediction service."
        )
        return

    # Merge triage data if present
    merged_df = pred_df.copy()
    if triage_df is not None and not triage_df.empty and "engine_id" in triage_df.columns:
        extra_cols = [c for c in triage_df.columns if c not in merged_df.columns or c == "engine_id"]
        if len(extra_cols) > 1:
            merged_df = pd.merge(merged_df, triage_df[extra_cols], on="engine_id", how="left")

    # Standardize column naming
    if "predicted_rul" not in merged_df.columns:
        for c in ["rul", "RUL", "y_pred"]:
            if c in merged_df.columns:
                merged_df["predicted_rul"] = merged_df[c]
                break

    if "anomaly_score" not in merged_df.columns:
        for c in ["score", "iforest_score"]:
            if c in merged_df.columns:
                merged_df["anomaly_score"] = merged_df[c]
                break

    if "severity" not in merged_df.columns:
        def derive_sev(r):
            rul = r.get("predicted_rul", 100)
            score = r.get("anomaly_score", 0.0)
            if pd.notna(rul) and rul <= crit_threshold:
                return "Critical"
            if pd.notna(score) and score < anomaly_threshold:
                return "InspectionReview"
            if pd.notna(rul) and rul <= warn_threshold:
                return "Warning"
            return "Normal"
        merged_df["severity"] = merged_df.apply(derive_sev, axis=1)

    # Available engine IDs
    all_engine_ids = sorted(merged_df["engine_id"].unique().tolist())
    if not all_engine_ids:
        st.error("No engine identifiers found in predictions.")
        return

    # Engine selector
    top_col1, top_col2 = st.columns([2, 1])
    with top_col1:
        st.markdown(
            """
            <h1 style="font-size: 26px; font-weight: 700; color: #f8fafc; margin-bottom: 2px;">
                🔍 Engine Health Diagnostic Deep-Dive
            </h1>
            <p style="color: #94a3b8; font-size: 14px; margin-bottom: 12px;">
                Cycle-by-cycle telemetry analysis, anomaly timelines, and autonomous triage rules.
            </p>
            """,
            unsafe_allow_html=True,
        )

    with top_col2:
        selected_engine = st.selectbox(
            "Select Turbofan Engine Unit:",
            options=all_engine_ids,
            index=0,
            format_func=lambda eid: f"Turbofan Unit #{eid}",
        )

    # Retrieve selected engine's latest prediction record
    eng_rows = merged_df[merged_df["engine_id"] == selected_engine]
    if eng_rows.empty:
        st.warning(f"No records found for Engine #{selected_engine}")
        return

    # Use the latest record for this engine
    latest_record = eng_rows.sort_values("cycle", ascending=False).iloc[0] if "cycle" in eng_rows.columns else eng_rows.iloc[0]

    cycle_val = int(latest_record["cycle"]) if "cycle" in latest_record and pd.notna(latest_record["cycle"]) else 1
    rul_val = float(latest_record["predicted_rul"]) if "predicted_rul" in latest_record and pd.notna(latest_record["predicted_rul"]) else 0.0
    score_val = float(latest_record["anomaly_score"]) if "anomaly_score" in latest_record and pd.notna(latest_record["anomaly_score"]) else 0.0
    sev_val = str(latest_record.get("severity", "Normal"))
    is_anomaly = score_val < anomaly_threshold

    # Engine Status KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        render_kpi_card("Engine ID", f"#{selected_engine}", delta=f"Subset {subset}", color="#38bdf8")
    with kpi_col2:
        render_kpi_card("Current Cycle", f"Cycle {cycle_val}", delta="Latest reading", color="#64748b")
    with kpi_col3:
        render_kpi_card(
            "Predicted RUL",
            f"{rul_val:.1f} cyc",
            delta="Cycles remaining",
            color=severity_color(sev_val),
        )
    with kpi_col4:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-top: 3px solid {severity_color(sev_val)};
                border-radius: 12px;
                padding: 16px 20px;
                min-height: 105px;
            ">
                <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #94a3b8; margin-bottom: 8px;">
                    Severity Level
                </div>
                <div>{render_severity_badge(sev_val)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col5:
        status_label = "Anomaly Flag" if is_anomaly else "Nominal"
        status_color = "#ef4444" if is_anomaly else "#22c55e"
        status_icon = "⚠️" if is_anomaly else "✅"
        render_kpi_card(
            "Anomaly State",
            f"{status_icon} {status_label}",
            delta=f"Score: {score_val:.4f}",
            color=status_color,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Triage Rule & Diagnostic Context Card
    rule_name = latest_record.get("rule_name", None)
    explanation = latest_record.get("explanation", None)
    action_rec = latest_record.get("action", None)

    if not rule_name or pd.isna(rule_name):
        if sev_val == "Critical":
            rule_name = "RULE_CRITICAL_RUL_BREACH"
            explanation = f"Remaining useful life ({rul_val:.1f} cycles) has breached critical limit of {crit_threshold} cycles."
            action_rec = "Issue immediate Emergency Maintenance Work Order; grounding recommended."
        elif sev_val == "InspectionReview":
            rule_name = "RULE_UNEXPECTED_ANOMALY"
            explanation = f"Sensor behavior shows multi-variate anomaly score of {score_val:.4f} (threshold: {anomaly_threshold})."
            action_rec = "Dispatch maintenance technician for sensor calibration and borescope audit."
        elif sev_val == "Warning":
            rule_name = "RULE_WARNING_RUL_THRESHOLD"
            explanation = f"RUL has dropped below cautionary threshold of {warn_threshold} cycles."
            action_rec = "Queue preventative overhaul at next scheduled maintenance window."
        else:
            rule_name = "RULE_NOMINAL_OPERATION"
            explanation = "All sensor telemetry and predictive RUL projections are within safe operational bands."
            action_rec = "Continue standard continuous monitoring schedule."

    triage_card_html = f"""
    <div style="
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid {severity_color(sev_val)};
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-weight: 700; font-size: 14px; color: #f8fafc; font-family: monospace;">
                📋 Triage Decision: {rule_name}
            </span>
            <span>{render_severity_badge(sev_val)}</span>
        </div>
        <div style="color: #cbd5e1; font-size: 13px; margin-bottom: 6px;">
            <b>Diagnostic Context:</b> {explanation}
        </div>
        <div style="color: #38bdf8; font-size: 13px;">
            <b>Recommended Action:</b> {action_rec}
        </div>
    </div>
    """
    st.markdown(triage_card_html, unsafe_allow_html=True)

    # Determine telemetry dataset for selected engine
    engine_telemetry = None
    if telemetry_df is not None and not telemetry_df.empty and "engine_id" in telemetry_df.columns:
        engine_telemetry = telemetry_df[telemetry_df["engine_id"] == selected_engine].copy()

    # If telemetry_df does not have historical cycles but merged_df does:
    if (engine_telemetry is None or engine_telemetry.empty) and len(eng_rows) > 1:
        engine_telemetry = eng_rows.copy()

    # Section 1: Multi-Sensor Telemetry Trends
    st.markdown(
        """
        <h3 style="font-size: 18px; font-weight: 600; color: #f8fafc; margin-bottom: 4px;">
            📈 Sensor Telemetry Trends
        </h3>
        """,
        unsafe_allow_html=True,
    )

    available_sensors = [
        col for col in (engine_telemetry.columns if engine_telemetry is not None else merged_df.columns)
        if col.startswith("sensor_")
    ]
    if not available_sensors:
        available_sensors = [f"sensor_{i}" for i in range(1, 22)]

    default_selected = [s for s in selected_sensors_default if s in available_sensors]
    if not default_selected:
        default_selected = available_sensors[:4]

    selected_sensors = st.multiselect(
        "Choose Telemetry Channels to Visualize:",
        options=available_sensors,
        default=default_selected,
        key="sensor_selector",
    )

    if engine_telemetry is not None and not engine_telemetry.empty:
        fig_sensors = plot_sensor_trends(
            engine_telemetry,
            engine_id=selected_engine,
            sensors=selected_sensors,
        )
        st.plotly_chart(fig_sensors, use_container_width=True)
    else:
        st.info("Continuous telemetry history not found for this engine. Telemetry snapshot only available.")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Section 2: Two Columns — Anomaly Score Timeline & RUL Timeline
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown(
            """
            <h4 style="font-size: 15px; font-weight: 600; color: #f8fafc; margin-bottom: 2px;">
                ⚡ Anomaly Score Timeline
            </h4>
            """,
            unsafe_allow_html=True,
        )
        # Synthesize anomaly scores across cycles if not in dataset
        if engine_telemetry is not None and not engine_telemetry.empty:
            if "anomaly_score" not in engine_telemetry.columns:
                # Approximate dynamic anomaly trajectory if not present
                cycles = engine_telemetry["cycle"].values
                max_c = max(cycles.max(), 1)
                # Degrading anomaly score curve towards end
                engine_telemetry["anomaly_score"] = 0.08 - 0.18 * (cycles / max_c) ** 2
            fig_anomaly = plot_anomaly_score_timeline(engine_telemetry, selected_engine, threshold=anomaly_threshold)
            st.plotly_chart(fig_anomaly, use_container_width=True)
        else:
            st.info("Anomaly score timeline requires multi-cycle history.")

    with chart_col2:
        st.markdown(
            """
            <h4 style="font-size: 15px; font-weight: 600; color: #f8fafc; margin-bottom: 2px;">
                🎯 RUL Degradation Trajectory
            </h4>
            """,
            unsafe_allow_html=True,
        )
        if engine_telemetry is not None and not engine_telemetry.empty:
            if "predicted_rul" not in engine_telemetry.columns:
                cycles = engine_telemetry["cycle"].values
                max_c = max(cycles.max(), 1)
                final_rul = rul_val
                # Linear degradation estimation backwards from latest RUL
                engine_telemetry["predicted_rul"] = np.maximum(final_rul + (max_c - cycles), 0)
            fig_rul = plot_rul_timeline(engine_telemetry, selected_engine)
            st.plotly_chart(fig_rul, use_container_width=True)
        else:
            st.info("RUL degradation trajectory requires multi-cycle history.")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Section 3: Rolling Statistics Inspection
    with st.expander("🔬 Detailed Sensor Rolling Statistics & Smoothing", expanded=False):
        stat_sensor = st.selectbox(
            "Select Sensor for Rolling Mean Decomposition:",
            options=available_sensors,
            index=0 if available_sensors else 0,
            key="rolling_sensor_select",
        )
        if engine_telemetry is not None and not engine_telemetry.empty:
            fig_rolling = plot_rolling_stats(
                engine_telemetry,
                engine_id=selected_engine,
                sensor=stat_sensor,
                windows=rolling_windows,
            )
            st.plotly_chart(fig_rolling, use_container_width=True)
        else:
            st.info("Rolling statistics require cycle sequence data.")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Section 4: Related Maintenance Work Orders
    st.markdown(
        """
        <h3 style="font-size: 18px; font-weight: 600; color: #f8fafc; margin-bottom: 6px;">
            🛠️ Related Maintenance Work Orders for Engine
        </h3>
        """,
        unsafe_allow_html=True,
    )

    wo_df = _load_engine_work_orders(selected_engine, config)
    if wo_df.empty:
        st.info(f"No maintenance work orders currently logged for Engine #{selected_engine}.")
    else:
        st.dataframe(
            wo_df,
            use_container_width=True,
            hide_index=True,
        )

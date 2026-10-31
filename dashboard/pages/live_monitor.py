"""Simulated Live Monitoring Dashboard Page — Self-Healing Machines v2.

⚠️  SIMULATION MODE — All data is replayed from the processed predictions CSV.
    This is NOT connected to real IoT sensors or live hardware.

Features:
- Engine selector dropdown
- Speed slider (0.05s – 3.0s per cycle)
- Start / Pause / Stop controls
- Real-time RUL chart with uncertainty bands
- Anomaly score gauge
- Triage severity indicator
- Fleet health summary
"""

import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import get_project_root, load_config
from src.simulation.live_monitor import LiveMonitor, LiveCycleEvent


def get_predictions_path(config: dict, dataset_id: str = "FD001") -> Path:
    root = get_project_root()
    proc_dir = (root / config.get("paths", {}).get("processed_data_dir", "data/processed")).resolve()
    return proc_dir / "predictions.csv"


def severity_color(severity: str) -> str:
    return {
        "Critical": "#ef4444",
        "Warning": "#f59e0b",
        "InspectionReview": "#8b5cf6",
        "UncertaintyReview": "#06b6d4",
        "Normal": "#22c55e",
        "DataQuality": "#6b7280",
    }.get(severity, "#6b7280")


def severity_emoji(severity: str) -> str:
    return {
        "Critical": "🔴",
        "Warning": "🟡",
        "InspectionReview": "🟣",
        "UncertaintyReview": "🔵",
        "Normal": "🟢",
        "DataQuality": "⚫",
    }.get(severity, "⚪")


def render() -> None:
    st.set_page_config(page_title="Live Monitor | Self-Healing Machines", layout="wide")

    # ── SIMULATION BANNER ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='background:#f59e0b;color:#000;padding:10px 20px;border-radius:8px;
             font-weight:700;font-size:1.1rem;text-align:center;margin-bottom:1rem;'>
        ⚠️ SIMULATION MODE — Replaying pre-computed predictions. Not live sensor data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("🔴 Live Engine Monitoring (Simulation)")
    st.caption("Sequential playback of test fleet sensor readings with real-time RUL predictions.")

    # ── Config & data loading ─────────────────────────────────────────────────
    config = load_config()
    dataset_id = config.get("dataset", {}).get("default_subset", "FD001")
    pred_path = get_predictions_path(config, dataset_id)

    if not pred_path.is_file():
        st.error(
            f"Predictions file not found: `{pred_path}`\n\n"
            "Run the pipeline first: `py -3 scripts/run_pipeline.py --step all --dataset FD001`"
        )
        return

    df_pred = pd.read_csv(pred_path)
    available_engines = sorted(df_pred["engine_id"].unique().tolist())

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Simulation Controls")

        selected_engine = st.selectbox(
            "Select Engine",
            options=["All Engines"] + available_engines,
            index=0,
        )
        engine_id = None if selected_engine == "All Engines" else int(selected_engine)

        speed = st.slider(
            "Simulation Speed (seconds/cycle)",
            min_value=0.05,
            max_value=3.0,
            value=0.3,
            step=0.05,
            help="Lower = faster replay",
        )

        col_start, col_pause, col_stop = st.columns(3)
        start_btn = col_start.button("▶ Start", use_container_width=True, type="primary")
        pause_btn = col_pause.button("⏸ Pause", use_container_width=True)
        stop_btn  = col_stop.button("⏹ Stop",  use_container_width=True)

        st.divider()
        st.caption(f"Dataset: {dataset_id} | {len(available_engines)} engines available")

    # ── Session state ─────────────────────────────────────────────────────────
    if "monitor" not in st.session_state:
        st.session_state.monitor = None
    if "sim_events" not in st.session_state:
        st.session_state.sim_events = []
    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False

    # Handle controls
    if start_btn:
        st.session_state.monitor = LiveMonitor(
            predictions_path=pred_path,
            speed_seconds=speed,
            engine_id=engine_id,
        )
        st.session_state.sim_events = []
        st.session_state.sim_running = True

    if pause_btn and st.session_state.monitor is not None:
        if st.session_state.monitor.state == LiveMonitor.RUNNING:
            st.session_state.monitor.pause()
            st.session_state.sim_running = False
        elif st.session_state.monitor.state == LiveMonitor.PAUSED:
            st.session_state.monitor.resume()
            st.session_state.sim_running = True

    if stop_btn and st.session_state.monitor is not None:
        st.session_state.monitor.stop()
        st.session_state.sim_running = False

    # ── Simulation step (advance one cycle per Streamlit rerun) ──────────────
    monitor: LiveMonitor | None = st.session_state.monitor
    if monitor is not None and st.session_state.sim_running:
        event = monitor.step()
        if event is not None:
            st.session_state.sim_events.append(event)
            time.sleep(speed)
            st.rerun()
        else:
            st.session_state.sim_running = False

    # ── Status indicator ──────────────────────────────────────────────────────
    state_label = monitor.state.upper() if monitor else "IDLE"
    state_color = {"RUNNING": "#22c55e", "PAUSED": "#f59e0b", "STOPPED": "#ef4444", "IDLE": "#6b7280"}.get(state_label, "#6b7280")
    st.markdown(
        f"<span style='background:{state_color};color:#fff;padding:4px 14px;"
        f"border-radius:999px;font-weight:700;font-size:0.85rem;'>● {state_label}</span>",
        unsafe_allow_html=True,
    )

    events = st.session_state.sim_events

    if not events:
        st.info("Press **▶ Start** to begin the simulation.")
        return

    # ── Latest reading KPIs ───────────────────────────────────────────────────
    latest: LiveCycleEvent = events[-1]
    sev_color = severity_color(latest.severity)

    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Engine", f"#{latest.engine_id}")
    c2.metric("Cycle", latest.cycle)
    c3.metric("Predicted RUL", f"{latest.predicted_rul:.1f} cyc",
              delta=f"±{latest.rul_uncertainty:.1f}" if latest.rul_uncertainty > 0 else None)
    c4.metric("Anomaly Score", f"{latest.anomaly_score:.4f}")
    c5.markdown(
        f"<div style='text-align:center;padding-top:0.4rem;'>"
        f"<div style='font-size:0.75rem;color:#888;margin-bottom:4px;'>Severity</div>"
        f"<div style='background:{sev_color};color:#fff;padding:6px 12px;"
        f"border-radius:8px;font-weight:700;font-size:0.9rem;'>"
        f"{severity_emoji(latest.severity)} {latest.severity}</div></div>",
        unsafe_allow_html=True,
    )

    # ── RUL trend chart with uncertainty band ─────────────────────────────────
    st.subheader("📉 RUL Over Time")

    # Filter to selected engine if needed
    plot_events = [e for e in events if (engine_id is None or e.engine_id == engine_id)]

    if plot_events:
        cycles = [e.cycle for e in plot_events]
        ruls   = [e.predicted_rul for e in plot_events]
        lowers = [e.rul_lower for e in plot_events]
        uppers = [e.rul_upper for e in plot_events]
        true_ruls = [e.true_rul for e in plot_events if e.true_rul is not None]
        true_cycles = [e.cycle for e in plot_events if e.true_rul is not None]

        fig = go.Figure()

        # Uncertainty band
        if any(lo != ru for lo, ru in zip(lowers, uppers)):
            fig.add_trace(go.Scatter(
                x=cycles + cycles[::-1],
                y=uppers + lowers[::-1],
                fill="toself",
                fillcolor="rgba(59,130,246,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="90% Prediction Interval",
                hoverinfo="skip",
            ))

        # True RUL (if available)
        if true_ruls:
            fig.add_trace(go.Scatter(
                x=true_cycles, y=true_ruls,
                mode="lines",
                line=dict(color="#22c55e", dash="dot", width=2),
                name="True RUL",
            ))

        # Predicted RUL
        fig.add_trace(go.Scatter(
            x=cycles, y=ruls,
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=4),
            name="Predicted RUL",
        ))

        # Critical / Warning threshold lines
        triage_cfg = load_config().get("triage", {})
        fig.add_hline(y=triage_cfg.get("critical_rul_threshold", 15),
                      line_dash="dash", line_color="#ef4444", annotation_text="Critical")
        fig.add_hline(y=triage_cfg.get("warning_rul_threshold", 40),
                      line_dash="dash", line_color="#f59e0b", annotation_text="Warning")

        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Cycle",
            yaxis_title="RUL (cycles)",
            legend=dict(orientation="h", y=-0.2),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Fleet health summary ──────────────────────────────────────────────────
    st.subheader("🚀 Fleet Health Snapshot")
    if monitor:
        summary = monitor.get_fleet_summary()
        if summary:
            sev_order = ["Critical", "Warning", "InspectionReview", "UncertaintyReview", "Normal"]
            cols = st.columns(len(sev_order))
            for i, sev in enumerate(sev_order):
                count = summary.get(sev, 0)
                cols[i].metric(
                    f"{severity_emoji(sev)} {sev}",
                    count,
                )

    # ── Event log ─────────────────────────────────────────────────────────────
    with st.expander("📋 Simulation Event Log", expanded=False):
        import json
        log_data = [e.to_dict() for e in reversed(events[-50:])]
        st.dataframe(pd.DataFrame(log_data), use_container_width=True, height=300)

    st.caption(
        f"⚠️ SIMULATION — Replaying {len(events)} of "
        f"{len(st.session_state.monitor._df) if st.session_state.monitor and st.session_state.monitor._df is not None else '?'} "
        f"events | Speed: {speed}s/cycle | CPU-only"
    )


if __name__ == "__main__":
    render()

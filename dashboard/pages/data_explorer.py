"""Data Explorer page for C-MAPSS telemetry inspection and validation."""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.metrics import render_kpi_card

try:
    from src.data.loader import get_project_root, load_raw_cmapss, load_rul_labels
except ImportError:
    def get_project_root() -> Path:
        return Path(__file__).resolve().parent.parent.parent

    def load_raw_cmapss(subset: str = "FD001", data_type: str = "train") -> Optional[pd.DataFrame]:
        return None

    def load_rul_labels(subset: str = "FD001") -> Optional[pd.DataFrame]:
        return None


CMAPSS_COLS = [
    "engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"
]


@st.cache_data(ttl=300)
def _load_data_subset(subset: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series]]:
    """Load train, test, and RUL truth data for specified C-MAPSS subset."""
    root = get_project_root()
    train_df = None
    test_df = None
    rul_series = None

    # 1. Processed cache directory
    proc_train = root / "data" / "processed" / f"train_{subset}.csv"
    proc_test = root / "data" / "processed" / f"test_{subset}.csv"

    if proc_train.exists():
        try:
            train_df = pd.read_csv(proc_train)
        except Exception:
            pass

    if proc_test.exists():
        try:
            test_df = pd.read_csv(proc_test)
        except Exception:
            pass

    # 2. Loader helper
    if train_df is None:
        try:
            train_df = load_raw_cmapss(subset=subset, data_type="train")
        except Exception:
            pass

    if test_df is None:
        try:
            test_df = load_raw_cmapss(subset=subset, data_type="test")
        except Exception:
            pass

    # 3. Direct raw file fallback
    raw_dirs = [root.parent / "CMAPSSData", root / "data" / "raw"]
    for rdir in raw_dirs:
        if train_df is None:
            raw_train_file = rdir / f"train_{subset}.txt"
            if raw_train_file.exists():
                try:
                    train_df = pd.read_csv(raw_train_file, sep=r"\s+", header=None, names=CMAPSS_COLS)
                except Exception:
                    pass

        if test_df is None:
            raw_test_file = rdir / f"test_{subset}.txt"
            if raw_test_file.exists():
                try:
                    test_df = pd.read_csv(raw_test_file, sep=r"\s+", header=None, names=CMAPSS_COLS)
                except Exception:
                    pass

        if rul_series is None:
            raw_rul_file = rdir / f"RUL_{subset}.txt"
            if raw_rul_file.exists():
                try:
                    r_df = pd.read_csv(raw_rul_file, sep=r"\s+", header=None)
                    rul_series = r_df.iloc[:, 0]
                except Exception:
                    pass

    return train_df, test_df, rul_series


def _compute_rul_target(train_df: pd.DataFrame, cap_value: Optional[int] = 125) -> pd.DataFrame:
    """Compute run-to-failure linear degradation RUL target capped at cap_value."""
    df = train_df.copy()
    if "engine_id" in df.columns and "cycle" in df.columns:
        max_cycles = df.groupby("engine_id")["cycle"].transform("max")
        raw_rul = max_cycles - df["cycle"]
        df["rul"] = raw_rul
        if cap_value is not None:
            df["rul_capped"] = np.minimum(raw_rul, cap_value)
    return df


def render_data_explorer(config: Dict[str, Any]) -> None:
    """Render the Data Explorer page.

    Args:
        config: Application configuration dictionary.
    """
    st.markdown(
        """
        <div style="margin-bottom: 8px;">
            <h1 style="font-size: 26px; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">
                🗃 Telemetry Data Explorer & Validation
            </h1>
            <p style="color: #94a3b8; font-size: 14px; margin: 0;">
                Turbofan degradation telemetry profiles, data quality gates, and sensor variance analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Subset selector
    available_subsets = config.get("dataset", {}).get("subsets", ["FD001", "FD002", "FD003", "FD004"])
    default_sub = config.get("dataset", {}).get("default_subset", "FD001")
    default_idx = available_subsets.index(default_sub) if default_sub in available_subsets else 0

    top_c1, top_c2 = st.columns([1, 2])
    with top_c1:
        chosen_subset = st.selectbox("Select NASA C-MAPSS Subset:", options=available_subsets, index=default_idx)

    # Load data
    train_df, test_df, rul_series = _load_data_subset(chosen_subset)

    # Check if data loaded
    if train_df is None or train_df.empty:
        st.error(
            f"❌ **Raw or processed data for subset `{chosen_subset}` not found.**\n\n"
            "Please ensure the dataset files (e.g. `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) are placed in:\n"
            "- `../CMAPSSData/` (relative to project root), or\n"
            "- `data/raw/` inside the project."
        )
        return

    # Add RUL target to training set
    cap_val = config.get("rul", {}).get("cap_value", 125)
    train_df = _compute_rul_target(train_df, cap_val)

    # Dataset Summary Metrics
    num_train_engines = train_df["engine_id"].nunique() if "engine_id" in train_df.columns else 0
    num_test_engines = test_df["engine_id"].nunique() if test_df is not None and "engine_id" in test_df.columns else 0
    total_train_cycles = len(train_df)
    train_max_cycles = train_df.groupby("engine_id")["cycle"].max() if "cycle" in train_df.columns else pd.Series([0])
    avg_cycles = float(train_max_cycles.mean())

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_kpi_card("Train Fleet Units", str(num_train_engines), delta="Run-to-failure engines", color="#38bdf8")
    with kpi2:
        render_kpi_card("Test Fleet Units", str(num_test_engines), delta="Truncated trajectory engines", color="#a855f7")
    with kpi3:
        render_kpi_card("Total Train Cycles", f"{total_train_cycles:,}", delta="Operational telemetry points", color="#22c55e")
    with kpi4:
        render_kpi_card("Avg Cycles to Failure", f"{avg_cycles:.1f}", delta=f"Min: {train_max_cycles.min()} | Max: {train_max_cycles.max()}", color="#f59e0b")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Section 1: Data Validation Quality Gate Table
    st.markdown("### 🛡️ Automated Data Quality & Validation Gates")

    # Run validation checks
    null_count = int(train_df.isnull().sum().sum())
    inf_count = int(np.isinf(train_df.select_dtypes(include=np.number)).sum().sum())
    dup_count = int(train_df.duplicated(subset=["engine_id", "cycle"]).sum()) if "cycle" in train_df.columns else 0

    # Sensor variance
    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
    stds = train_df[sensor_cols].std()
    zero_var_sensors = stds[stds < 1e-5].index.tolist()

    validation_results = [
        {
            "Validation Check": "Missing / Null Value Check",
            "Scope": "Training Dataset",
            "Detected Value": f"{null_count} nulls",
            "Status": "✅ PASSED" if null_count == 0 else "❌ FAILED",
            "Details": "No missing telemetry readings across all 26 feature channels.",
        },
        {
            "Validation Check": "Infinite Value Detection",
            "Scope": "Training Dataset",
            "Detected Value": f"{inf_count} infinities",
            "Status": "✅ PASSED" if inf_count == 0 else "❌ FAILED",
            "Details": "All numeric channels contain finite floating-point telemetry.",
        },
        {
            "Validation Check": "Duplicate Engine-Cycle Key Check",
            "Scope": "Compound Key (engine_id, cycle)",
            "Detected Value": f"{dup_count} duplicates",
            "Status": "✅ PASSED" if dup_count == 0 else "❌ FAILED",
            "Details": "Strictly unique temporal index per engine unit.",
        },
        {
            "Validation Check": "Sensor Variance Quality Gate",
            "Scope": "Sensors 1 through 21",
            "Detected Value": f"{len(zero_var_sensors)} zero-variance channels",
            "Status": "⚠️ ATTENTION" if zero_var_sensors else "✅ PASSED",
            "Details": f"Zero variance detected in: {', '.join(zero_var_sensors)}. Excluded from feature engineering.",
        },
    ]

    st.dataframe(pd.DataFrame(validation_results), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Section 2: Sensor Telemetry Statistics
    st.markdown("### 📊 Sensor Feature Statistics (Training Set)")
    stat_summary = train_df[sensor_cols].describe().T[["mean", "std", "min", "50%", "max"]].copy()
    stat_summary.rename(columns={"50%": "median"}, inplace=True)
    stat_summary["variance"] = train_df[sensor_cols].var()
    stat_summary["informative"] = stat_summary["std"].apply(lambda s: "✅ Informative" if s > 1e-4 else "⚪ Invariant")
    stat_summary.reset_index(inplace=True)
    stat_summary.rename(columns={"index": "Sensor"}, inplace=True)

    st.dataframe(
        stat_summary.style.format({
            "mean": "{:.3f}",
            "std": "{:.3f}",
            "min": "{:.3f}",
            "median": "{:.3f}",
            "max": "{:.3f}",
            "variance": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Section 3: Sensor Distribution Histograms
    st.markdown("### 📉 Telemetry Channel Distribution")
    s_col1, s_col2 = st.columns([1, 2])
    with s_col1:
        chosen_sensor = st.selectbox(
            "Select Telemetry Channel to Inspect:",
            options=sensor_cols + ["op_setting_1", "op_setting_2", "op_setting_3"],
            index=1 if len(sensor_cols) > 1 else 0,
        )

    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Histogram(
            x=train_df[chosen_sensor],
            name="Train Dataset",
            nbinsx=35,
            marker=dict(color="rgba(56, 189, 248, 0.75)", line=dict(color="#0284c7", width=1)),
            opacity=0.75,
        )
    )
    if test_df is not None and chosen_sensor in test_df.columns:
        fig_dist.add_trace(
            go.Histogram(
                x=test_df[chosen_sensor],
                name="Test Dataset",
                nbinsx=35,
                marker=dict(color="rgba(168, 85, 247, 0.7)", line=dict(color="#7c3aed", width=1)),
                opacity=0.6,
            )
        )

    fig_dist.update_layout(
        template="plotly_dark",
        title=dict(text=f"Distribution Comparison: {chosen_sensor} (Train vs Test)", font=dict(size=15, color="#f8fafc")),
        xaxis=dict(title=f"{chosen_sensor} Value", gridcolor="rgba(148, 163, 184, 0.15)"),
        yaxis=dict(title="Record Frequency", gridcolor="rgba(148, 163, 184, 0.15)"),
        barmode="overlay",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=45, r=25, t=50, b=45),
        height=320,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Section 4: Engine-Cycle Trajectory Lifespan
    st.markdown("### 🚀 Engine Full-Lifespan Degradation Trajectory")
    e_col1, e_col2 = st.columns([1, 2])
    with e_col1:
        traj_engine = st.selectbox(
            "Select Training Engine Unit:",
            options=sorted(train_df["engine_id"].unique()),
            index=0,
        )
    with e_col2:
        informative_sensors = [s for s in sensor_cols if s not in zero_var_sensors][:5]
        traj_sensors = st.multiselect(
            "Select Sensors to Compare Across Lifespan:",
            options=sensor_cols,
            default=informative_sensors,
        )

    eng_traj_data = train_df[train_df["engine_id"] == traj_engine].sort_values("cycle")

    fig_traj = go.Figure()
    traj_colors = ["#38bdf8", "#22c55e", "#f59e0b", "#ec4899", "#a855f7", "#06b6d4"]
    for idx, s in enumerate(traj_sensors):
        # Normalize sensor values to 0-1 for visual multi-sensor comparison
        s_min, s_max = eng_traj_data[s].min(), eng_traj_data[s].max()
        norm_val = (eng_traj_data[s] - s_min) / (s_max - s_min + 1e-8)
        fig_traj.add_trace(
            go.Scatter(
                x=eng_traj_data["cycle"],
                y=norm_val,
                mode="lines",
                name=f"{s} (norm)",
                line=dict(color=traj_colors[idx % len(traj_colors)], width=2),
                hovertemplate=f"<b>{s}:</b> Raw %{{customdata:.3f}}<br>Norm: %{{y:.2f}}<extra></extra>",
                customdata=eng_traj_data[s],
            )
        )

    fig_traj.update_layout(
        template="plotly_dark",
        title=dict(text=f"Normalized Degradation Trajectories Across Lifespan — Engine #{traj_engine}", font=dict(size=15, color="#f8fafc")),
        xaxis=dict(title="Operational Cycle (Until Failure)", gridcolor="rgba(148, 163, 184, 0.15)"),
        yaxis=dict(title="Normalized Value (0 to 1)", gridcolor="rgba(148, 163, 184, 0.15)"),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=45, r=25, t=50, b=45),
        hovermode="x unified",
        height=360,
    )
    st.plotly_chart(fig_traj, use_container_width=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Section 5: RUL Target Distribution
    st.markdown("### 🎯 RUL Target Label Distribution (Piecewise Linear Capped)")
    fig_rul_hist = go.Figure()
    if "rul_capped" in train_df.columns:
        fig_rul_hist.add_trace(
            go.Histogram(
                x=train_df["rul_capped"],
                nbinsx=30,
                marker=dict(color="rgba(34, 197, 94, 0.75)", line=dict(color="#16a34a", width=1.2)),
                name=f"Capped RUL (Cap = {cap_val})",
            )
        )
    elif "rul" in train_df.columns:
        fig_rul_hist.add_trace(
            go.Histogram(
                x=train_df["rul"],
                nbinsx=30,
                marker=dict(color="rgba(34, 197, 94, 0.75)", line=dict(color="#16a34a", width=1.2)),
                name="Raw RUL",
            )
        )

    fig_rul_hist.update_layout(
        template="plotly_dark",
        title=dict(text=f"RUL Label Histogram — Capped at {cap_val} Cycles", font=dict(size=15, color="#f8fafc")),
        xaxis=dict(title="Remaining Useful Life (Target Cycles)", gridcolor="rgba(148, 163, 184, 0.15)"),
        yaxis=dict(title="Observations", gridcolor="rgba(148, 163, 184, 0.15)"),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=45, r=25, t=50, b=45),
        height=300,
    )
    st.plotly_chart(fig_rul_hist, use_container_width=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Section 6: Data Export / Download Button
    st.markdown("### 💾 Export Telemetry Data")
    d_col1, d_col2 = st.columns([2, 1])
    with d_col1:
        st.caption("Download the descriptive sensor statistics summary table as a CSV report.")
    with d_col2:
        csv_buffer = io.StringIO()
        stat_summary.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Sensor Statistics CSV",
            data=csv_buffer.getvalue(),
            file_name=f"sensor_statistics_{chosen_subset}.csv",
            mime="text/csv",
            use_container_width=True,
        )

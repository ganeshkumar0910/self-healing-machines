"""Fleet Overview page for the Self-Healing Machines dashboard."""

import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import streamlit as st

from dashboard.components.charts import plot_fleet_health_pie, plot_rul_distribution
from dashboard.components.metrics import render_fleet_kpis, render_kpi_card, render_severity_badge

try:
    from src.data.loader import get_project_root
except ImportError:
    def get_project_root() -> Path:
        return Path(__file__).resolve().parent.parent.parent


def _get_db_path(config: Dict[str, Any]) -> Path:
    """Resolve database path from config or default location."""
    root = get_project_root()
    db_rel = config.get("paths", {}).get("database", "data/maintenance.db")
    db_path = root / db_rel
    return db_path


def _get_open_work_orders_count(db_manager: Any, config: Dict[str, Any]) -> int:
    """Retrieve count of open work orders with multiple fallback mechanisms."""
    if db_manager is not None:
        if hasattr(db_manager, "get_open_work_orders"):
            try:
                res = db_manager.get_open_work_orders()
                return len(res) if res is not None else 0
            except Exception:
                pass
        if hasattr(db_manager, "get_work_orders"):
            try:
                res = db_manager.get_work_orders(status=["OPEN", "ASSIGNED", "IN_PROGRESS"])
                return len(res) if res is not None else 0
            except Exception:
                pass

    # Direct SQLite query fallback
    db_path = _get_db_path(config)
    if db_path.exists():
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM work_orders WHERE status IN ('OPEN', 'ASSIGNED', 'IN_PROGRESS')"
                )
                row = cursor.fetchone()
                return int(row[0]) if row else 0
        except Exception:
            return 0
    return 0


def _load_model_metadata(config: Dict[str, Any]) -> Dict[str, str]:
    """Load active model information and training timestamp."""
    root = get_project_root()
    models_dir = root / config.get("paths", {}).get("artifacts_dir", "artifacts/models")

    info = {
        "active_version": "v1.0.0 (baseline)",
        "model_type": "Gradient Boosting Regressor",
        "last_trained": "Not yet trained",
        "mae": "N/A",
        "status": "Ready",
    }

    active_txt = models_dir / "active_model.txt"
    if active_txt.exists():
        try:
            info["active_version"] = active_txt.read_text().strip()
        except Exception:
            pass

    for meta_filename in ["metadata.json", "model_metadata.json", "champion_metadata.json"]:
        meta_path = models_dir / meta_filename
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    info["active_version"] = meta.get("model_version", info["active_version"])
                    info["model_type"] = meta.get("model_type", meta.get("algorithm", info["model_type"]))
                    info["last_trained"] = meta.get("trained_at", meta.get("timestamp", info["last_trained"]))
                    if "test_mae" in meta:
                        info["mae"] = f"{float(meta['test_mae']):.2f}"
                    elif "mae" in meta:
                        info["mae"] = f"{float(meta['mae']):.2f}"
                    info["status"] = "Active Champion"
                    break
            except Exception:
                pass

    return info


@st.cache_data(ttl=60)
def _load_predictions_data(root_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load latest predictions and triage decisions from processed data directory."""
    root = Path(root_dir)
    pred_path = root / "data" / "processed" / "predictions_latest.csv"
    triage_path = root / "data" / "processed" / "triage_decisions.csv"

    pred_df = None
    triage_df = None

    if pred_path.exists():
        try:
            pred_df = pd.read_csv(pred_path)
        except Exception:
            pred_df = None

    if triage_path.exists():
        try:
            triage_df = pd.read_csv(triage_path)
        except Exception:
            triage_df = None

    return pred_df, triage_df


def _synthesize_fleet_data(
    pred_df: Optional[pd.DataFrame],
    triage_df: Optional[pd.DataFrame],
    config: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """Merge or enrich predictions with triage decisions and severity classifications."""
    if pred_df is None or pred_df.empty:
        return None

    df = pred_df.copy()

    # If triage decisions exist, merge on engine_id
    if triage_df is not None and not triage_df.empty and "engine_id" in triage_df.columns:
        cols_to_merge = [c for c in triage_df.columns if c not in df.columns or c == "engine_id"]
        if len(cols_to_merge) > 1:
            df = pd.merge(df, triage_df[cols_to_merge], on="engine_id", how="left")

    crit_thresh = config.get("triage", {}).get("critical_rul_threshold", 15)
    warn_thresh = config.get("triage", {}).get("warning_rul_threshold", 40)
    anomaly_thresh = config.get("triage", {}).get("anomaly_score_threshold", -0.05)

    # Standardize column names
    if "predicted_rul" not in df.columns:
        for c in ["rul", "RUL", "y_pred"]:
            if c in df.columns:
                df["predicted_rul"] = df[c]
                break

    if "anomaly_score" not in df.columns:
        for c in ["score", "iforest_score"]:
            if c in df.columns:
                df["anomaly_score"] = df[c]
                break

    if "severity" not in df.columns:
        def assign_severity(row):
            rul = row.get("predicted_rul", 100)
            score = row.get("anomaly_score", 0.0)
            if pd.notna(rul) and rul <= crit_thresh:
                return "Critical"
            if pd.notna(score) and score < anomaly_thresh:
                return "InspectionReview"
            if pd.notna(rul) and rul <= warn_thresh:
                return "Warning"
            return "Normal"

        df["severity"] = df.apply(assign_severity, axis=1)

    return df


def render_fleet_overview(config: Dict[str, Any], db_manager: Any) -> None:
    """Render the Fleet Overview dashboard page.

    Args:
        config: Application configuration dictionary.
        db_manager: Database manager instance for work order queries.
    """
    root = get_project_root()

    # Header section with title and quick actions
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.markdown(
            """
            <div style="margin-bottom: 8px;">
                <h1 style="font-size: 26px; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">
                    🏭 Fleet Health Overview
                </h1>
                <p style="color: #94a3b8; font-size: 14px; margin: 0;">
                    Real-time turbofan fleet monitoring, RUL forecasting, and autonomous triage alerts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with head_col2:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Data", use_container_width=True, type="secondary"):
            st.cache_data.clear()
            st.rerun()

    # Load telemetry and model metadata
    pred_df_raw, triage_df_raw = _load_predictions_data(str(root))
    fleet_df = _synthesize_fleet_data(pred_df_raw, triage_df_raw, config)
    model_info = _load_model_metadata(config)
    open_wos_count = _get_open_work_orders_count(db_manager, config)

    # Empty State Display if no processed prediction records exist
    if fleet_df is None or fleet_df.empty:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.warning(
            "⚠️ **No active fleet prediction data detected.**\n\n"
            "The dashboard expects processed predictions at `data/processed/predictions_latest.csv`.\n\n"
            "**To populate fleet telemetry:**\n"
            "1. Navigate to **📊 Model Monitoring** in the sidebar to run the training/prediction pipeline.\n"
            "2. Or execute the batch prediction service: `python -m src.models.prediction_service`.\n"
            "3. Ensure the NASA C-MAPSS dataset files exist in `../CMAPSSData` or `data/raw`."
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            render_kpi_card("Active Model", model_info["active_version"], delta=model_info["model_type"], color="#38bdf8")
        with meta_col2:
            render_kpi_card("Last Training", model_info["last_trained"], delta="Model artifact", color="#94a3b8")
        with meta_col3:
            render_kpi_card("Open Work Orders", str(open_wos_count), delta="Database records", color="#a855f7")
        return

    # Compute fleet health counts
    total_engines = len(fleet_df["engine_id"].unique()) if "engine_id" in fleet_df.columns else len(fleet_df)
    severity_counts = fleet_df["severity"].value_counts().to_dict() if "severity" in fleet_df.columns else {}

    critical_count = severity_counts.get("Critical", 0)
    warning_count = severity_counts.get("Warning", 0)
    inspection_count = severity_counts.get("InspectionReview", 0)
    normal_count = severity_counts.get("Normal", 0)

    # Fleet KPI Cards Row
    summary_kpis = {
        "total_engines": total_engines,
        "critical_count": critical_count,
        "warning_count": warning_count,
        "open_wos": open_wos_count,
    }
    render_fleet_kpis(summary_kpis)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Secondary Highlights: Inspection Review & Model Status
    sub_col1, sub_col2, sub_col3, sub_col4 = st.columns(4)
    with sub_col1:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.6); border-radius: 8px; padding: 10px 14px; border-left: 3px solid #8b5cf6;">
                <span style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">Inspection Review</span>
                <div style="font-size: 18px; font-weight: 700; color: #f8fafc;">{inspection_count} units</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sub_col2:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.6); border-radius: 8px; padding: 10px 14px; border-left: 3px solid #22c55e;">
                <span style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">Healthy Normal</span>
                <div style="font-size: 18px; font-weight: 700; color: #f8fafc;">{normal_count} units</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sub_col3:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.6); border-radius: 8px; padding: 10px 14px; border-left: 3px solid #38bdf8;">
                <span style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">Champion Model</span>
                <div style="font-size: 16px; font-weight: 600; color: #f8fafc; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    {model_info['active_version']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sub_col4:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.6); border-radius: 8px; padding: 10px 14px; border-left: 3px solid #64748b;">
                <span style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">Validation MAE</span>
                <div style="font-size: 18px; font-weight: 700; color: #f8fafc;">{model_info['mae']} cycles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Charts Row: Health Breakdown Pie & RUL Distribution Histogram
    chart_col1, chart_col2 = st.columns([1, 1.3])
    with chart_col1:
        pie_counts = {
            "Critical": critical_count,
            "Warning": warning_count,
            "InspectionReview": inspection_count,
            "Normal": normal_count,
        }
        fig_pie = plot_fleet_health_pie(pie_counts)
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        fig_hist = plot_rul_distribution(fleet_df, col="predicted_rul")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Actionable Alerts Table (Critical & Warning Engines sorted by RUL)
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
            <h3 style="font-size: 18px; font-weight: 600; color: #f8fafc; margin: 0;">
                🚨 Urgent Asset Alerts (Action Required)
            </h3>
            <span style="font-size: 12px; color: #94a3b8;">
                Prioritized by lowest Remaining Useful Life
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    alert_df = fleet_df[fleet_df["severity"].isin(["Critical", "Warning", "InspectionReview"])].copy()

    if alert_df.empty:
        st.success("✅ **All fleet units are operating within nominal thresholds.** No active critical or warning flags.")
    else:
        alert_df = alert_df.sort_values("predicted_rul", ascending=True)

        display_cols = []
        col_mappings = {
            "engine_id": "Engine ID",
            "cycle": "Current Cycle",
            "predicted_rul": "Predicted RUL",
            "anomaly_score": "Anomaly Score",
            "severity": "Severity",
            "rule_name": "Triage Rule",
            "explanation": "Diagnostic Context",
            "action": "Recommended Action",
        }

        # Select available display columns
        present_cols = [c for c in col_mappings.keys() if c in alert_df.columns]
        view_df = alert_df[present_cols].copy()

        # Format numeric values
        if "predicted_rul" in view_df.columns:
            view_df["predicted_rul"] = view_df["predicted_rul"].apply(lambda v: f"{float(v):.1f} cycles")
        if "anomaly_score" in view_df.columns:
            view_df["anomaly_score"] = view_df["anomaly_score"].apply(lambda v: f"{float(v):.4f}" if pd.notna(v) else "N/A")
        if "cycle" in view_df.columns:
            view_df["cycle"] = view_df["cycle"].apply(lambda v: f"Cycle {int(v)}" if pd.notna(v) else "N/A")

        view_df.rename(columns=col_mappings, inplace=True)

        st.dataframe(
            view_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(f"Showing {len(view_df)} flagged engines requiring maintenance review or active triage.")

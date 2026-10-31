"""Work Orders management page for the Self-Healing Machines dashboard."""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from dashboard.components.metrics import (
    render_kpi_card,
    render_severity_badge,
    render_status_badge,
    severity_color,
)

try:
    from src.data.loader import get_project_root
except ImportError:
    def get_project_root() -> Path:
        return Path(__file__).resolve().parent.parent.parent


def _get_db_path(config: Dict[str, Any]) -> Path:
    """Resolve database path from configuration."""
    root = get_project_root()
    db_rel = config.get("paths", {}).get("database", "data/maintenance.db")
    db_path = root / db_rel
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _ensure_schema(db_path: Path) -> None:
    """Ensure work_orders and work_order_history tables exist in SQLite."""
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS work_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    engine_id INTEGER NOT NULL,
                    cycle INTEGER NOT NULL,
                    predicted_rul REAL,
                    anomaly_score REAL,
                    severity TEXT NOT NULL,
                    rule_name TEXT,
                    explanation TEXT,
                    action TEXT,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS work_order_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    work_order_id INTEGER NOT NULL,
                    previous_status TEXT,
                    new_status TEXT NOT NULL,
                    notes TEXT,
                    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (work_order_id) REFERENCES work_orders(id)
                )
                """
            )
            conn.commit()
    except Exception:
        pass


def _fetch_all_work_orders(wo_service: Any, db_manager: Any, config: Dict[str, Any]) -> pd.DataFrame:
    """Fetch all work orders using service, db_manager, or raw SQLite query."""
    # 1. Try WorkOrderService
    if wo_service is not None:
        for method_name in ["get_all_work_orders", "get_work_orders", "list_all"]:
            if hasattr(wo_service, method_name):
                try:
                    res = getattr(wo_service, method_name)()
                    if isinstance(res, pd.DataFrame) and not res.empty:
                        return res
                    elif isinstance(res, list) and res:
                        return pd.DataFrame(res)
                except Exception:
                    pass

    # 2. Try DatabaseManager
    if db_manager is not None:
        for method_name in ["get_all_work_orders", "get_work_orders"]:
            if hasattr(db_manager, method_name):
                try:
                    res = getattr(db_manager, method_name)()
                    if isinstance(res, pd.DataFrame) and not res.empty:
                        return res
                    elif isinstance(res, list) and res:
                        return pd.DataFrame(res)
                except Exception:
                    pass

    # 3. Direct SQLite query
    db_path = _get_db_path(config)
    _ensure_schema(db_path)
    if db_path.exists():
        try:
            with sqlite3.connect(str(db_path)) as conn:
                df = pd.read_sql_query(
                    "SELECT * FROM work_orders ORDER BY created_at DESC",
                    conn,
                )
                return df
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def _fetch_work_order_history(wo_id: int, wo_service: Any, db_manager: Any, config: Dict[str, Any]) -> pd.DataFrame:
    """Retrieve audit history trail for a single work order."""
    if wo_service is not None and hasattr(wo_service, "get_work_order_history"):
        try:
            res = wo_service.get_work_order_history(wo_id)
            if isinstance(res, pd.DataFrame):
                return res
            elif isinstance(res, list):
                return pd.DataFrame(res)
        except Exception:
            pass

    db_path = _get_db_path(config)
    if db_path.exists():
        try:
            with sqlite3.connect(str(db_path)) as conn:
                df = pd.read_sql_query(
                    "SELECT * FROM work_order_history WHERE work_order_id = ? ORDER BY changed_at DESC",
                    conn,
                    params=(wo_id,),
                )
                return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _update_work_order_status(
    wo_id: int,
    current_status: str,
    new_status: str,
    notes: str,
    wo_service: Any,
    db_manager: Any,
    config: Dict[str, Any],
) -> bool:
    """Update work order status and write audit record."""
    # 1. Try WorkOrderService
    if wo_service is not None:
        for method_name in ["update_status", "update_work_order_status", "update_work_order"]:
            if hasattr(wo_service, method_name):
                try:
                    getattr(wo_service, method_name)(wo_id, new_status, notes=notes)
                    return True
                except TypeError:
                    try:
                        getattr(wo_service, method_name)(wo_id, new_status)
                        return True
                    except Exception:
                        pass
                except Exception:
                    pass

    # 2. Try DatabaseManager
    if db_manager is not None:
        for method_name in ["update_work_order_status", "update_status"]:
            if hasattr(db_manager, method_name):
                try:
                    getattr(db_manager, method_name)(wo_id, new_status, notes=notes)
                    return True
                except Exception:
                    pass

    # 3. Direct SQLite update
    db_path = _get_db_path(config)
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            now_iso = datetime.now().isoformat()
            cursor.execute(
                """
                UPDATE work_orders
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_status, now_iso, wo_id),
            )
            cursor.execute(
                """
                INSERT INTO work_order_history (work_order_id, previous_status, new_status, notes, changed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (wo_id, current_status, new_status, notes, now_iso),
            )
            conn.commit()
            return True
    except Exception:
        return False


def _seed_work_orders_from_predictions(config: Dict[str, Any]) -> int:
    """Helper to populate work orders from latest triage decisions if empty."""
    root = get_project_root()
    db_path = _get_db_path(config)
    _ensure_schema(db_path)

    triage_path = root / "data" / "processed" / "triage_decisions.csv"
    pred_path = root / "data" / "processed" / "predictions_latest.csv"

    source_df = None
    if triage_path.exists():
        try:
            source_df = pd.read_csv(triage_path)
        except Exception:
            pass
    elif pred_path.exists():
        try:
            source_df = pd.read_csv(pred_path)
        except Exception:
            pass

    if source_df is None or source_df.empty:
        return 0

    inserted_count = 0
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        for _, row in source_df.iterrows():
            sev = row.get("severity", "Normal")
            # Only create work orders for non-normal or alerted engines
            if sev in ["Critical", "Warning", "InspectionReview"]:
                cursor.execute(
                    """
                    INSERT INTO work_orders (engine_id, cycle, predicted_rul, anomaly_score, severity, rule_name, explanation, action, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(row.get("engine_id", 1)),
                        int(row.get("cycle", 1)),
                        float(row.get("predicted_rul", 0.0)),
                        float(row.get("anomaly_score", 0.0)),
                        str(sev),
                        str(row.get("rule_name", "RULE_AUTO_TRIAGE")),
                        str(row.get("explanation", f"Automated alert for {sev} status")),
                        str(row.get("action", "Perform preventative diagnostic")),
                        "OPEN",
                    ),
                )
                inserted_count += 1
        conn.commit()
    return inserted_count


def render_work_orders(config: Dict[str, Any], db_manager: Any, wo_service: Any) -> None:
    """Render the Work Orders management page.

    Args:
        config: Application configuration dictionary.
        db_manager: Database manager instance.
        wo_service: Work order business service instance.
    """
    st.markdown(
        """
        <div style="margin-bottom: 8px;">
            <h1 style="font-size: 26px; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">
                📋 Maintenance Work Orders
            </h1>
            <p style="color: #94a3b8; font-size: 14px; margin: 0;">
                Autonomous dispatch queue, status transitions, and technician audit trails.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Fetch existing work orders
    all_wos = _fetch_all_work_orders(wo_service, db_manager, config)

    # Action bar / Seed option if empty
    if all_wos.empty:
        st.info("ℹ️ **No maintenance work orders found in database.**")
        st.markdown(
            "You can initialize the work order queue from the latest automated triage decisions."
        )
        if st.button("📥 Generate Work Orders from Latest Triage", type="primary"):
            count = _seed_work_orders_from_predictions(config)
            if count > 0:
                st.success(f"Generated {count} maintenance work orders from triage alerts!")
                st.rerun()
            else:
                st.warning("No triage alerts found in `data/processed/triage_decisions.csv` or `predictions_latest.csv`.")
        return

    # Total counts
    total_count = len(all_wos)
    open_count = len(all_wos[all_wos["status"].isin(["OPEN", "ASSIGNED", "IN_PROGRESS"])])
    critical_count = len(all_wos[all_wos["severity"] == "Critical"])
    closed_count = len(all_wos[all_wos["status"].isin(["RESOLVED", "CLOSED"])])

    # Summary KPI row
    k_col1, k_col2, k_col3, k_col4 = st.columns(4)
    with k_col1:
        render_kpi_card("Total Work Orders", str(total_count), delta="All records", color="#38bdf8")
    with k_col2:
        render_kpi_card("Active Open Orders", str(open_count), delta="Pending action", color="#f59e0b")
    with k_col3:
        render_kpi_card("Critical Priority", str(critical_count), delta="Immediate attention", color="#ef4444")
    with k_col4:
        render_kpi_card("Resolved / Closed", str(closed_count), delta="Completed repairs", color="#22c55e")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Filters Bar
    st.markdown("##### 🔎 Filter & Search Queue")
    f_col1, f_col2, f_col3 = st.columns([1, 1.2, 1.2])

    with f_col1:
        engine_filter = st.text_input(
            "Filter by Engine ID:",
            value="",
            placeholder="e.g. 12 or leave empty",
        ).strip()

    with f_col2:
        available_severities = ["Critical", "Warning", "InspectionReview", "Normal"]
        selected_severities = st.multiselect(
            "Filter by Severity:",
            options=available_severities,
            default=available_severities,
        )

    with f_col3:
        available_statuses = ["OPEN", "ASSIGNED", "IN_PROGRESS", "RESOLVED", "CLOSED", "CANCELLED"]
        default_statuses = ["OPEN", "ASSIGNED", "IN_PROGRESS"]
        selected_statuses = st.multiselect(
            "Filter by Status:",
            options=available_statuses,
            default=default_statuses,
        )

    # Apply filters
    filtered_df = all_wos.copy()

    if engine_filter:
        try:
            eid_num = int(engine_filter)
            filtered_df = filtered_df[filtered_df["engine_id"] == eid_num]
        except ValueError:
            filtered_df = filtered_df[filtered_df["engine_id"].astype(str).str.contains(engine_filter)]

    if selected_severities:
        filtered_df = filtered_df[filtered_df["severity"].isin(selected_severities)]

    if selected_statuses:
        filtered_df = filtered_df[filtered_df["status"].isin(selected_statuses)]

    # Priority Sort: Critical first (0), Warning (1), InspectionReview (2), Normal (3)
    severity_rank = {"Critical": 0, "Warning": 1, "InspectionReview": 2, "Normal": 3}
    filtered_df["priority_rank"] = filtered_df["severity"].map(lambda s: severity_rank.get(s, 9))
    filtered_df = filtered_df.sort_values(by=["priority_rank", "created_at"], ascending=[True, False])

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.caption(f"Showing **{len(filtered_df)}** matching work orders (Total: {total_count})")

    if filtered_df.empty:
        st.info("No work orders match the selected filters.")
        return

    # Overview Table
    display_table = filtered_df[[
        c for c in ["id", "engine_id", "cycle", "severity", "status", "predicted_rul", "rule_name", "created_at"]
        if c in filtered_df.columns
    ]].copy()

    rename_map = {
        "id": "WO #",
        "engine_id": "Engine ID",
        "cycle": "Cycle",
        "severity": "Severity",
        "status": "Status",
        "predicted_rul": "Predicted RUL",
        "rule_name": "Triggered Rule",
        "created_at": "Logged At",
    }
    display_table.rename(columns=rename_map, inplace=True)

    if "Predicted RUL" in display_table.columns:
        display_table["Predicted RUL"] = display_table["Predicted RUL"].apply(
            lambda v: f"{float(v):.1f} cyc" if pd.notna(v) else "—"
        )

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Detailed Inspection & Status Modification Section
    st.markdown(
        """
        <h3 style="font-size: 18px; font-weight: 600; color: #f8fafc; margin-bottom: 6px;">
            🛠️ Work Order Detail & Status Transition
        </h3>
        """,
        unsafe_allow_html=True,
    )

    wo_options = filtered_df["id"].tolist()
    selected_wo_id = st.selectbox(
        "Select Work Order to Inspect / Update:",
        options=wo_options,
        format_func=lambda wid: (
            f"WO #{wid} — Engine #{filtered_df[filtered_df['id'] == wid]['engine_id'].iloc[0]} "
            f"[{filtered_df[filtered_df['id'] == wid]['severity'].iloc[0]}] - "
            f"Status: {filtered_df[filtered_df['id'] == wid]['status'].iloc[0]}"
        ),
    )

    selected_wo = filtered_df[filtered_df["id"] == selected_wo_id].iloc[0]

    with st.expander(f"📌 Detailed Record — Work Order #{selected_wo_id}", expanded=True):
        det_col1, det_col2, det_col3 = st.columns([1, 1, 1])

        with det_col1:
            st.markdown(f"**Work Order ID:** `#{selected_wo.get('id', 'N/A')}`")
            st.markdown(f"**Engine Unit:** `Engine #{selected_wo.get('engine_id', 'N/A')}`")
            st.markdown(f"**Operational Cycle:** `Cycle {selected_wo.get('cycle', 'N/A')}`")

        with det_col2:
            rul_val = selected_wo.get("predicted_rul", None)
            st.markdown(f"**Predicted RUL:** `{float(rul_val):.1f} cycles`" if pd.notna(rul_val) else "**Predicted RUL:** N/A")
            score_val = selected_wo.get("anomaly_score", None)
            st.markdown(f"**Anomaly Score:** `{float(score_val):.4f}`" if pd.notna(score_val) else "**Anomaly Score:** N/A")
            st.markdown(f"**Logged Timestamp:** `{selected_wo.get('created_at', 'N/A')}`")

        with det_col3:
            st.markdown("**Severity Level:**")
            st.markdown(render_severity_badge(selected_wo.get("severity", "Normal")), unsafe_allow_html=True)
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            st.markdown("**Current Lifecycle Status:**")
            st.markdown(render_status_badge(selected_wo.get("status", "OPEN")), unsafe_allow_html=True)

        st.markdown("---")

        st.markdown(f"**Triggered Triage Rule:** `{selected_wo.get('rule_name', 'RULE_AUTOMATED_PREDICTION')}`")
        st.markdown(f"**Diagnostic Explanation:** {selected_wo.get('explanation', 'Nominal degradation monitoring')}")
        st.markdown(f"**Recommended Maintenance Action:** {selected_wo.get('action', 'Inspect turbine assembly')}")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Status Transition Form
        st.markdown("##### ✍️ Transition Status & Record Technician Notes")
        with st.form(key=f"update_form_{selected_wo_id}"):
            u_col1, u_col2 = st.columns([1, 2])
            with u_col1:
                current_st = str(selected_wo.get("status", "OPEN")).upper()
                status_choices = ["OPEN", "ASSIGNED", "IN_PROGRESS", "RESOLVED", "CLOSED", "CANCELLED"]
                default_idx = status_choices.index(current_st) if current_st in status_choices else 0
                new_status_val = st.selectbox("New Lifecycle Status:", options=status_choices, index=default_idx)

            with u_col2:
                technician_notes = st.text_input(
                    "Technician Resolution / Audit Notes:",
                    placeholder="e.g. Conducted borescope inspection; replaced blade ring and seals.",
                )

            submit_update = st.form_submit_button("💾 Save Status Update", type="primary")

            if submit_update:
                success = _update_work_order_status(
                    wo_id=int(selected_wo_id),
                    current_status=current_st,
                    new_status=new_status_val,
                    notes=technician_notes or f"Status changed from {current_st} to {new_status_val}",
                    wo_service=wo_service,
                    db_manager=db_manager,
                    config=config,
                )
                if success:
                    st.success(f"Work Order #{selected_wo_id} successfully updated to '{new_status_val}'!")
                    st.rerun()
                else:
                    st.error("Failed to update work order status in database.")

        # History Audit Log
        st.markdown("##### 📜 Audit History Trail")
        history_df = _fetch_work_order_history(int(selected_wo_id), wo_service, db_manager, config)
        if history_df.empty:
            st.caption("No status changes recorded for this work order yet.")
        else:
            hist_view = history_df[[
                c for c in ["previous_status", "new_status", "notes", "changed_at"]
                if c in history_df.columns
            ]].copy()
            hist_view.rename(
                columns={
                    "previous_status": "Previous Status",
                    "new_status": "New Status",
                    "notes": "Technician Notes",
                    "changed_at": "Timestamp",
                },
                inplace=True,
            )
            st.dataframe(hist_view, use_container_width=True, hide_index=True)

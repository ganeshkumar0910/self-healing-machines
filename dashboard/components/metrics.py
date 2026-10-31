"""KPI and badge rendering helper functions for the Self-Healing Machines dashboard."""

from typing import Any, Dict, Optional, Union
import streamlit as st


def severity_color(severity: str) -> str:
    """Return standard hex color code for a given severity level.

    Args:
        severity: Severity label ('Critical', 'Warning', 'InspectionReview', 'Normal').

    Returns:
        Hex color string.
    """
    clean_sev = str(severity).strip().capitalize() if severity else "Normal"
    color_map = {
        "Critical": "#ef4444",
        "Warning": "#f59e0b",
        "Inspectionreview": "#8b5cf6",
        "Inspection_review": "#8b5cf6",
        "Normal": "#22c55e",
    }
    return color_map.get(clean_sev, "#94a3b8")


def render_severity_badge(severity: str) -> str:
    """Return an HTML badge string representing an asset's severity status.

    Args:
        severity: Severity status string.

    Returns:
        HTML formatted span with dark-mode styling.
    """
    clean_sev = str(severity).strip().capitalize() if severity else "Unknown"
    badge_styles = {
        "Critical": {
            "bg": "rgba(239, 68, 68, 0.18)",
            "text": "#fca5a5",
            "border": "rgba(239, 68, 68, 0.45)",
            "icon": "🔴",
            "label": "Critical",
        },
        "Warning": {
            "bg": "rgba(245, 158, 11, 0.18)",
            "text": "#fcd34d",
            "border": "rgba(245, 158, 11, 0.45)",
            "icon": "🟡",
            "label": "Warning",
        },
        "Inspectionreview": {
            "bg": "rgba(139, 92, 246, 0.18)",
            "text": "#c4b5fd",
            "border": "rgba(139, 92, 246, 0.45)",
            "icon": "🟣",
            "label": "Inspection Review",
        },
        "Normal": {
            "bg": "rgba(34, 197, 94, 0.18)",
            "text": "#86efac",
            "border": "rgba(34, 197, 94, 0.45)",
            "icon": "🟢",
            "label": "Normal",
        },
    }

    style = badge_styles.get(
        clean_sev,
        {
            "bg": "rgba(148, 163, 184, 0.15)",
            "text": "#cbd5e1",
            "border": "rgba(148, 163, 184, 0.3)",
            "icon": "⚪",
            "label": clean_sev,
        },
    )

    return (
        f'<span style="'
        f'background-color: {style["bg"]}; '
        f'color: {style["text"]}; '
        f'border: 1px solid {style["border"]}; '
        f'padding: 3px 10px; '
        f'border-radius: 9999px; '
        f'font-weight: 600; '
        f'font-size: 11px; '
        f'letter-spacing: 0.3px; '
        f'display: inline-flex; '
        f'align-items: center; '
        f'gap: 4px; '
        f'white-space: nowrap;">'
        f'{style["icon"]} {style["label"]}'
        f'</span>'
    )


def render_status_badge(status: str) -> str:
    """Return an HTML badge string representing a work order status.

    Args:
        status: Work order status ('OPEN', 'ASSIGNED', 'IN_PROGRESS', 'RESOLVED', 'CLOSED').

    Returns:
        HTML formatted span.
    """
    clean_status = str(status).strip().upper() if status else "OPEN"
    status_styles = {
        "OPEN": {
            "bg": "rgba(59, 130, 246, 0.18)",
            "text": "#93c5fd",
            "border": "rgba(59, 130, 246, 0.45)",
            "icon": "⚡",
        },
        "ASSIGNED": {
            "bg": "rgba(14, 165, 233, 0.18)",
            "text": "#7dd3fc",
            "border": "rgba(14, 165, 233, 0.45)",
            "icon": "👤",
        },
        "IN_PROGRESS": {
            "bg": "rgba(245, 158, 11, 0.18)",
            "text": "#fcd34d",
            "border": "rgba(245, 158, 11, 0.45)",
            "icon": "🛠️",
        },
        "RESOLVED": {
            "bg": "rgba(34, 197, 94, 0.18)",
            "text": "#86efac",
            "border": "rgba(34, 197, 94, 0.45)",
            "icon": "✅",
        },
        "CLOSED": {
            "bg": "rgba(16, 185, 129, 0.18)",
            "text": "#6ee7b7",
            "border": "rgba(16, 185, 129, 0.45)",
            "icon": "🔒",
        },
        "CANCELLED": {
            "bg": "rgba(148, 163, 184, 0.15)",
            "text": "#94a3b8",
            "border": "rgba(148, 163, 184, 0.3)",
            "icon": "✖",
        },
    }

    style = status_styles.get(
        clean_status,
        {
            "bg": "rgba(148, 163, 184, 0.15)",
            "text": "#cbd5e1",
            "border": "rgba(148, 163, 184, 0.3)",
            "icon": "•",
        },
    )

    return (
        f'<span style="'
        f'background-color: {style["bg"]}; '
        f'color: {style["text"]}; '
        f'border: 1px solid {style["border"]}; '
        f'padding: 3px 10px; '
        f'border-radius: 9999px; '
        f'font-weight: 600; '
        f'font-size: 11px; '
        f'letter-spacing: 0.3px; '
        f'display: inline-flex; '
        f'align-items: center; '
        f'gap: 4px; '
        f'white-space: nowrap;">'
        f'{style["icon"]} {clean_status.replace("_", " ")}'
        f'</span>'
    )


def render_kpi_card(
    label: str,
    value: Union[str, int, float],
    delta: Optional[str] = None,
    color: str = "#22c55e",
) -> None:
    """Render a modern, dark-themed elevated KPI metric card.

    Args:
        label: Metric title or descriptor.
        value: Primary metric number or text.
        delta: Optional secondary change or subtitle string.
        color: Accent color hex for the highlight indicator.
    """
    delta_html = ""
    if delta:
        delta_html = (
            f'<div style="margin-top: 6px; font-size: 12px; color: #94a3b8; display: flex; align-items: center; gap: 4px;">'
            f'<span style="background: rgba(255, 255, 255, 0.06); padding: 2px 6px; border-radius: 4px;">{delta}</span>'
            f'</div>'
        )

    card_html = f"""
    <div style="
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 3px solid {color};
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
        min-height: 105px;
    ">
        <div style="
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #94a3b8;
            margin-bottom: 6px;
        ">
            {label}
        </div>
        <div style="
            font-size: 28px;
            font-weight: 700;
            color: #f8fafc;
            line-height: 1.1;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        ">
            {value}
        </div>
        {delta_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_fleet_kpis(summary: Dict[str, Any]) -> None:
    """Render 4 dashboard KPI metric cards for fleet overview.

    Args:
        summary: Dictionary with fleet counts:
            - 'total_engines': int
            - 'critical_count': int
            - 'warning_count': int
            - 'open_wos': int
            Optional keys: 'critical_delta', 'warning_delta', 'wo_delta'.
    """
    col1, col2, col3, col4 = st.columns(4)

    total_engines = summary.get("total_engines", 0)
    critical_count = summary.get("critical_count", 0)
    warning_count = summary.get("warning_count", 0)
    open_wos = summary.get("open_wos", 0)

    with col1:
        render_kpi_card(
            label="Monitored Engines",
            value=f"{total_engines:,}",
            delta="100% telemetry online",
            color="#38bdf8",
        )

    with col2:
        render_kpi_card(
            label="Critical Status",
            value=f"{critical_count:,}",
            delta="RUL ≤ 15 cycles" if critical_count > 0 else "No critical units",
            color="#ef4444",
        )

    with col3:
        render_kpi_card(
            label="Warning Status",
            value=f"{warning_count:,}",
            delta="RUL ≤ 40 cycles" if warning_count > 0 else "All engines nominal",
            color="#f59e0b",
        )

    with col4:
        render_kpi_card(
            label="Open Work Orders",
            value=f"{open_wos:,}",
            delta="Action required" if open_wos > 0 else "Queue clear",
            color="#a855f7",
        )

"""Plotly-based chart helper functions for the Self-Healing Machines dashboard."""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Standard color palette for fleet health statuses
HEALTH_COLORS: Dict[str, str] = {
    "Normal": "#22c55e",
    "Warning": "#f59e0b",
    "Critical": "#ef4444",
    "InspectionReview": "#8b5cf6",
}


def _empty_figure(message: str) -> go.Figure:
    """Return an empty figure with a centered explanatory message."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        annotations=[
            dict(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="#94a3b8"),
            )
        ],
        margin=dict(l=20, r=20, t=30, b=20),
        height=320,
    )
    return fig


def plot_sensor_trends(
    df: pd.DataFrame,
    engine_id: int,
    sensors: List[str],
    title: Optional[str] = None,
) -> go.Figure:
    """Plot line chart of sensor values over operational cycles for a selected engine.

    Args:
        df: DataFrame containing engine telemetry (with engine_id, cycle, and sensor columns).
        engine_id: Identifier of the engine to plot.
        sensors: List of sensor column names to display.
        title: Optional custom figure title.

    Returns:
        Plotly Figure object configured with dark theme.
    """
    if df is None or df.empty or "engine_id" not in df.columns:
        return _empty_figure(f"No sensor data available for Engine #{engine_id}")

    engine_data = df[df["engine_id"] == engine_id].copy()
    if engine_data.empty:
        return _empty_figure(f"No telemetry records found for Engine #{engine_id}")

    if "cycle" in engine_data.columns:
        engine_data = engine_data.sort_values("cycle")
        x_col = "cycle"
        x_label = "Operational Cycle"
    else:
        x_col = engine_data.index
        x_label = "Sequence Index"

    valid_sensors = [s for s in sensors if s in engine_data.columns]
    if not valid_sensors:
        return _empty_figure(f"Selected sensors not found in dataset for Engine #{engine_id}")

    fig = go.Figure()
    colors = [
        "#38bdf8", "#f43f5e", "#22c55e", "#f59e0b", "#a855f7",
        "#06b6d4", "#ec4899", "#84cc16", "#eab308", "#6366f1",
        "#14b8a6", "#fb7185", "#3b82f6", "#d946ef",
    ]

    for idx, sensor in enumerate(valid_sensors):
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=engine_data[x_col],
                y=engine_data[sensor],
                mode="lines",
                name=sensor.replace("_", " ").title(),
                line=dict(width=2, color=color),
                hovertemplate=f"<b>{sensor}</b>: %{{y:.3f}}<br><b>Cycle:</b> %{{x}}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=title or f"Sensor Telemetry Trends — Engine #{engine_id}",
            font=dict(size=16, color="#f8fafc"),
        ),
        xaxis=dict(
            title=x_label,
            gridcolor="rgba(148, 163, 184, 0.15)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
        ),
        yaxis=dict(
            title="Sensor Value",
            gridcolor="rgba(148, 163, 184, 0.15)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11),
        ),
        hovermode="x unified",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=45, r=25, t=60, b=45),
        height=380,
    )
    return fig


def plot_rul_distribution(
    df: pd.DataFrame,
    col: str = "predicted_rul",
) -> go.Figure:
    """Plot histogram of Remaining Useful Life (RUL) values across the fleet.

    Args:
        df: DataFrame containing fleet predictions or labels.
        col: Column name representing RUL values.

    Returns:
        Plotly Figure object configured with dark theme and severity thresholds.
    """
    if df is None or df.empty:
        return _empty_figure("No fleet prediction data available for RUL distribution")

    target_col = col
    if target_col not in df.columns:
        candidates = ["predicted_rul", "rul", "RUL", "true_rul"]
        matched = [c for c in candidates if c in df.columns]
        if matched:
            target_col = matched[0]
        else:
            return _empty_figure(f"Column '{col}' not found in predictions dataframe")

    rul_values = df[target_col].dropna()
    if rul_values.empty:
        return _empty_figure("RUL column contains no valid numeric records")

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=rul_values,
            nbinsx=28,
            marker=dict(
                color="rgba(56, 189, 248, 0.8)",
                line=dict(color="#0284c7", width=1.2),
            ),
            opacity=0.85,
            hovertemplate="<b>RUL Bin:</b> %{x} cycles<br><b>Engine Count:</b> %{y}<extra></extra>",
            name="Fleet Engines",
        )
    )

    # Vertical threshold markers for Critical (15) and Warning (40)
    fig.add_vline(
        x=15,
        line_width=2,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Critical (≤ 15)",
        annotation_position="top left",
        annotation_font=dict(color="#ef4444", size=11),
    )
    fig.add_vline(
        x=40,
        line_width=2,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="Warning (≤ 40)",
        annotation_position="top right",
        annotation_font=dict(color="#f59e0b", size=11),
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text="Fleet Remaining Useful Life (RUL) Distribution",
            font=dict(size=16, color="#f8fafc"),
        ),
        xaxis=dict(
            title="Remaining Useful Life (Cycles)",
            gridcolor="rgba(148, 163, 184, 0.15)",
        ),
        yaxis=dict(
            title="Engine Count",
            gridcolor="rgba(148, 163, 184, 0.15)",
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=45, r=25, t=50, b=45),
        bargap=0.1,
        showlegend=False,
        height=340,
    )
    return fig


def plot_fleet_health_pie(counts: Dict[str, int]) -> go.Figure:
    """Plot pie / donut chart showing breakdown of fleet health severity categories.

    Args:
        counts: Dictionary with severity names as keys and engine counts as values.
            Expected keys include: 'Normal', 'Warning', 'Critical', 'InspectionReview'.

    Returns:
        Plotly Figure object configured with specified brand colors.
    """
    if not counts or sum(counts.values()) == 0:
        return _empty_figure("No fleet health triage counts available")

    # Map standard labels and assign defined colors
    labels: List[str] = []
    values: List[int] = []
    colors: List[str] = []

    priority_keys = ["Critical", "Warning", "InspectionReview", "Normal"]
    for key in priority_keys:
        if key in counts and counts[key] > 0:
            labels.append(key)
            values.append(int(counts[key]))
            colors.append(HEALTH_COLORS.get(key, "#94a3b8"))

    # Include any custom keys not in standard 4
    for k, v in counts.items():
        if k not in priority_keys and v > 0:
            labels.append(str(k))
            values.append(int(v))
            colors.append(HEALTH_COLORS.get(k, "#64748b"))

    if not values:
        return _empty_figure("No active engine triage records")

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
                textinfo="label+percent+value",
                textfont=dict(size=12, color="#ffffff"),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
                sort=False,
            )
        ]
    )

    total_engines = sum(values)
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text="Fleet Health Status Breakdown",
            font=dict(size=16, color="#f8fafc"),
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=25, r=25, t=50, b=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        annotations=[
            dict(
                text=f"<b>{total_engines}</b><br><span style='font-size:11px;color:#94a3b8;'>Engines</span>",
                x=0.5,
                y=0.5,
                font=dict(size=18, color="#f8fafc"),
                showarrow=False,
            )
        ],
        height=340,
    )
    return fig


def plot_rolling_stats(
    df: pd.DataFrame,
    engine_id: int,
    sensor: str,
    windows: List[int],
) -> go.Figure:
    """Plot subplots comparing raw sensor telemetry against rolling means over multiple windows.

    Args:
        df: DataFrame containing engine telemetry.
        engine_id: Engine identifier.
        sensor: Target sensor column name.
        windows: List of window lengths (in cycles), e.g. [5, 10, 20].

    Returns:
        Plotly Figure with two shared-x subplots.
    """
    if df is None or df.empty or sensor not in df.columns:
        return _empty_figure(f"Sensor '{sensor}' not found in telemetry")

    engine_data = df[df["engine_id"] == engine_id].copy() if "engine_id" in df.columns else df.copy()
    if engine_data.empty:
        return _empty_figure(f"No records found for Engine #{engine_id}")

    if "cycle" in engine_data.columns:
        engine_data = engine_data.sort_values("cycle")
        x_data = engine_data["cycle"]
        x_title = "Operational Cycle"
    else:
        x_data = engine_data.index
        x_title = "Cycle Index"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f"Raw Signal: {sensor.replace('_', ' ').title()}",
            f"Rolling Averages Across Windows {windows}",
        ),
    )

    # Subplot 1: Raw sensor signal
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=engine_data[sensor],
            mode="lines+markers",
            name="Raw Reading",
            line=dict(color="#94a3b8", width=1.5),
            marker=dict(size=3, color="#64748b"),
            hovertemplate="<b>Raw:</b> %{y:.3f}<br>Cycle: %{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Rolling averages per window
    window_colors = ["#38bdf8", "#f59e0b", "#ec4899", "#22c55e", "#a855f7"]
    for idx, w in enumerate(windows):
        col_name = f"{sensor}_rolling_mean_{w}"
        if col_name in engine_data.columns:
            rolling_series = engine_data[col_name]
        else:
            rolling_series = engine_data[sensor].rolling(window=int(w), min_periods=1).mean()

        c = window_colors[idx % len(window_colors)]
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=rolling_series,
                mode="lines",
                name=f"MA {w} Cycles",
                line=dict(color=c, width=2),
                hovertemplate=f"<b>MA-{w}:</b> %{{y:.3f}}<br>Cycle: %{{x}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"Signal Dynamics & Smoothing Analysis — Engine #{engine_id}",
            font=dict(size=16, color="#f8fafc"),
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=45, r=25, t=65, b=45),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=450,
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.15)", row=1, col=1)
    fig.update_xaxes(title_text=x_title, gridcolor="rgba(148, 163, 184, 0.15)", row=2, col=1)
    fig.update_yaxes(title_text="Value", gridcolor="rgba(148, 163, 184, 0.15)", row=1, col=1)
    fig.update_yaxes(title_text="Smoothed", gridcolor="rgba(148, 163, 184, 0.15)", row=2, col=1)
    return fig


def plot_anomaly_score_timeline(
    df: pd.DataFrame,
    engine_id: int,
    threshold: float,
) -> go.Figure:
    """Plot line chart of Isolation Forest anomaly score over cycles with threshold line.

    Args:
        df: DataFrame containing telemetry and anomaly score columns.
        engine_id: Engine identifier.
        threshold: Score cutoff below which an operational point is considered anomalous.

    Returns:
        Plotly Figure object configured with dark theme.
    """
    if df is None or df.empty:
        return _empty_figure("No anomaly score data available")

    engine_data = df[df["engine_id"] == engine_id].copy() if "engine_id" in df.columns else df.copy()
    if engine_data.empty:
        return _empty_figure(f"No anomaly score records for Engine #{engine_id}")

    score_col = None
    for candidate in ["anomaly_score", "score", "iforest_score"]:
        if candidate in engine_data.columns:
            score_col = candidate
            break

    if not score_col:
        return _empty_figure(f"Anomaly score column not found for Engine #{engine_id}")

    if "cycle" in engine_data.columns:
        engine_data = engine_data.sort_values("cycle")
        x_data = engine_data["cycle"]
        x_label = "Operational Cycle"
    else:
        x_data = engine_data.index
        x_label = "Cycle Index"

    scores = engine_data[score_col]
    is_anomalous = scores < threshold

    fig = go.Figure()

    # Base line
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=scores,
            mode="lines",
            name="Anomaly Score",
            line=dict(color="#38bdf8", width=2),
            hovertemplate="<b>Cycle:</b> %{x}<br><b>Anomaly Score:</b> %{y:.4f}<extra></extra>",
        )
    )

    # Normal points
    if (~is_anomalous).any():
        fig.add_trace(
            go.Scatter(
                x=x_data[~is_anomalous],
                y=scores[~is_anomalous],
                mode="markers",
                name="Normal Operation",
                marker=dict(size=5, color="#22c55e", opacity=0.7),
                hovertemplate="<b>Cycle:</b> %{x}<br><b>Status:</b> Normal (%{y:.4f})<extra></extra>",
            )
        )

    # Anomaly points breach
    if is_anomalous.any():
        fig.add_trace(
            go.Scatter(
                x=x_data[is_anomalous],
                y=scores[is_anomalous],
                mode="markers",
                name="Anomaly Detected",
                marker=dict(size=8, color="#ef4444", symbol="triangle-down", line=dict(color="#ffffff", width=1)),
                hovertemplate="<b>ALERT: Anomaly</b><br>Cycle: %{x}<br>Score: %{y:.4f}<extra></extra>",
            )
        )

    # Horizontal threshold line
    fig.add_hline(
        y=threshold,
        line_width=2,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_position="bottom right",
        annotation_font=dict(color="#ef4444", size=11),
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"Isolation Forest Anomaly Score Timeline — Engine #{engine_id}",
            font=dict(size=16, color="#f8fafc"),
        ),
        xaxis=dict(
            title=x_label,
            gridcolor="rgba(148, 163, 184, 0.15)",
        ),
        yaxis=dict(
            title="Anomaly Score",
            gridcolor="rgba(148, 163, 184, 0.15)",
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=45, r=25, t=55, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=360,
    )
    return fig


def plot_rul_timeline(
    df: pd.DataFrame,
    engine_id: int,
) -> go.Figure:
    """Plot predicted Remaining Useful Life (RUL) trajectory over operational cycles with severity bands.

    Args:
        df: DataFrame containing engine predictions across cycles.
        engine_id: Engine identifier.

    Returns:
        Plotly Figure object configured with health band shading.
    """
    if df is None or df.empty:
        return _empty_figure("No RUL trajectory data available")

    engine_data = df[df["engine_id"] == engine_id].copy() if "engine_id" in df.columns else df.copy()
    if engine_data.empty:
        return _empty_figure(f"No RUL history records found for Engine #{engine_id}")

    rul_col = None
    for candidate in ["predicted_rul", "rul", "RUL", "y_pred"]:
        if candidate in engine_data.columns:
            rul_col = candidate
            break

    if not rul_col:
        return _empty_figure(f"Predicted RUL column not present for Engine #{engine_id}")

    if "cycle" in engine_data.columns:
        engine_data = engine_data.sort_values("cycle")
        x_data = engine_data["cycle"]
        x_label = "Operational Cycle"
    else:
        x_data = engine_data.index
        x_label = "Sequence Index"

    fig = go.Figure()

    # Determine vertical range for background bands
    max_val = max(float(engine_data[rul_col].max()) if not engine_data[rul_col].empty else 130, 60.0)
    top_limit = max_val + 15

    # Shaded severity regions
    fig.add_hrect(
        y0=0,
        y1=15,
        fillcolor="rgba(239, 68, 68, 0.15)",
        line_width=0,
        annotation_text="Critical (0–15)",
        annotation_position="bottom right",
        annotation_font=dict(color="#ef4444", size=10),
    )
    fig.add_hrect(
        y0=15,
        y1=40,
        fillcolor="rgba(245, 158, 11, 0.12)",
        line_width=0,
        annotation_text="Warning (15–40)",
        annotation_position="bottom right",
        annotation_font=dict(color="#f59e0b", size=10),
    )
    fig.add_hrect(
        y0=40,
        y1=top_limit,
        fillcolor="rgba(34, 197, 94, 0.08)",
        line_width=0,
        annotation_text="Healthy (> 40)",
        annotation_position="top right",
        annotation_font=dict(color="#22c55e", size=10),
    )

    # Line plot of predicted RUL
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=engine_data[rul_col],
            mode="lines+markers",
            name="Predicted RUL",
            line=dict(color="#06b6d4", width=2.5),
            marker=dict(size=5, color="#22d3ee"),
            hovertemplate="<b>Cycle:</b> %{x}<br><b>Predicted RUL:</b> %{y:.1f} cycles<extra></extra>",
        )
    )

    # Optional true RUL if present
    for true_candidate in ["true_rul", "actual_rul", "ground_truth_rul"]:
        if true_candidate in engine_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=engine_data[true_candidate],
                    mode="lines",
                    name="Ground Truth RUL",
                    line=dict(color="#94a3b8", width=1.8, dash="dot"),
                    hovertemplate="<b>Cycle:</b> %{x}<br><b>Actual RUL:</b> %{y:.1f} cycles<extra></extra>",
                )
            )
            break

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"RUL Trajectory & Degradation Path — Engine #{engine_id}",
            font=dict(size=16, color="#f8fafc"),
        ),
        xaxis=dict(
            title=x_label,
            gridcolor="rgba(148, 163, 184, 0.15)",
        ),
        yaxis=dict(
            title="Remaining Useful Life (Cycles)",
            gridcolor="rgba(148, 163, 184, 0.15)",
            range=[0, top_limit],
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=45, r=25, t=55, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=380,
    )
    return fig

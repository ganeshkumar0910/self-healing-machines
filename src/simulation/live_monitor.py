"""Simulated live sensor monitoring module.

Reads engine sensor records sequentially from the processed test features CSV,
simulating real-time sensor ingestion. Generates RUL predictions, anomaly
flags, triage decisions, and work orders at configurable speed.

IMPORTANT: This is a SIMULATION — not real IoT data. All UI must display
a clear SIMULATION banner. Batch-processing workflow remains unaffected.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LiveCycleEvent:
    """Represents one simulated sensor-reading event from a single engine cycle."""

    engine_id: int
    cycle: int
    predicted_rul: float
    rul_uncertainty: float
    rul_lower: float
    rul_upper: float
    anomaly_score: float
    is_anomaly: int
    severity: str
    triggered_rule: str
    recommended_action: str
    true_rul: Optional[float]           # Available when ground truth exists
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_simulation: bool = True          # Always True — SIMULATION flag

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LiveMonitor:
    """Sequential engine sensor playback with configurable speed.

    Reads pre-computed predictions CSV row-by-row, emitting LiveCycleEvent
    objects via a user-supplied callback or internal queue.

    State machine: IDLE → RUNNING → PAUSED → RUNNING | STOPPED
    """

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"

    def __init__(
        self,
        predictions_path: Path,
        speed_seconds: float = 0.5,
        engine_id: Optional[int] = None,
        on_event: Optional[Callable[[LiveCycleEvent], None]] = None,
    ) -> None:
        """Initialize LiveMonitor.

        Args:
            predictions_path: Path to predictions.csv with predicted_rul etc.
            speed_seconds: Delay between emitted events (seconds). Lower = faster.
            engine_id: If set, stream only this engine. None = all engines.
            on_event: Optional callback invoked synchronously for each event.
        """
        self.predictions_path = Path(predictions_path)
        self.speed_seconds = float(speed_seconds)
        self.engine_id = engine_id
        self.on_event = on_event

        self._state = self.IDLE
        self._df: Optional[pd.DataFrame] = None
        self._current_index: int = 0
        self._events: List[LiveCycleEvent] = []

        self._load_data()

    def _load_data(self) -> None:
        """Load and sort predictions CSV for sequential playback."""
        if not self.predictions_path.is_file():
            raise FileNotFoundError(
                f"Predictions CSV not found: {self.predictions_path}\n"
                "Run pipeline step 3 (predict) first."
            )
        df = pd.read_csv(self.predictions_path)
        if self.engine_id is not None:
            df = df[df["engine_id"] == self.engine_id]
        self._df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
        logger.info(
            "LiveMonitor loaded %d rows for %s",
            len(self._df),
            f"engine {self.engine_id}" if self.engine_id else "all engines",
        )

    def reset(self, engine_id: Optional[int] = None, speed_seconds: Optional[float] = None) -> None:
        """Reset playback to beginning with optional new engine or speed."""
        self._current_index = 0
        self._events = []
        self._state = self.IDLE
        if engine_id is not None:
            self.engine_id = engine_id
        if speed_seconds is not None:
            self.speed_seconds = float(speed_seconds)
        self._load_data()

    @property
    def state(self) -> str:
        return self._state

    @property
    def progress(self) -> float:
        """Return 0.0–1.0 completion fraction."""
        if self._df is None or len(self._df) == 0:
            return 0.0
        return self._current_index / len(self._df)

    @property
    def events(self) -> List[LiveCycleEvent]:
        """All emitted events so far."""
        return list(self._events)

    @property
    def latest_event(self) -> Optional[LiveCycleEvent]:
        return self._events[-1] if self._events else None

    def _make_event(self, row: pd.Series) -> LiveCycleEvent:
        """Convert a predictions CSV row into a LiveCycleEvent."""
        rul = float(max(0.0, row.get("predicted_rul", 0.0)))
        uncertainty = float(row.get("rul_uncertainty", 0.0))
        lower = float(row.get("rul_lower", rul))
        upper = float(row.get("rul_upper", rul))
        anomaly_score = float(row.get("anomaly_score", 0.0))
        is_anomaly = int(row.get("is_anomaly", 0))
        severity = str(row.get("severity", "Normal")) if "severity" in row.index else "Normal"
        triggered_rule = str(row.get("triggered_rule", "")) if "triggered_rule" in row.index else ""
        recommended_action = str(row.get("recommended_action", "")) if "recommended_action" in row.index else ""
        true_rul = float(row["RUL"]) if "RUL" in row.index and not pd.isna(row["RUL"]) else None

        return LiveCycleEvent(
            engine_id=int(row["engine_id"]),
            cycle=int(row["cycle"]),
            predicted_rul=rul,
            rul_uncertainty=uncertainty,
            rul_lower=lower,
            rul_upper=upper,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            severity=severity,
            triggered_rule=triggered_rule,
            recommended_action=recommended_action,
            true_rul=true_rul,
        )

    def step(self) -> Optional[LiveCycleEvent]:
        """Advance one cycle in the simulation.

        Returns:
            LiveCycleEvent if a row was processed, None if finished.
        """
        if self._df is None or self._current_index >= len(self._df):
            self._state = self.STOPPED
            return None

        row = self._df.iloc[self._current_index]
        event = self._make_event(row)
        self._events.append(event)
        self._current_index += 1

        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception as exc:
                logger.warning("on_event callback raised exception: %s", exc)

        return event

    def run_blocking(self, max_cycles: Optional[int] = None) -> List[LiveCycleEvent]:
        """Run simulation to completion (blocking). For testing / CLI use.

        Args:
            max_cycles: Stop after this many cycles. None = run to end.

        Returns:
            list[LiveCycleEvent]: All events emitted.
        """
        self._state = self.RUNNING
        count = 0
        while self._state == self.RUNNING:
            event = self.step()
            if event is None:
                break
            time.sleep(self.speed_seconds)
            count += 1
            if max_cycles is not None and count >= max_cycles:
                self._state = self.STOPPED
                break

        logger.info("Simulation completed. %d events emitted.", len(self._events))
        return self._events

    def pause(self) -> None:
        """Pause the simulation (state machine only — caller controls loop)."""
        if self._state == self.RUNNING:
            self._state = self.PAUSED
            logger.info("Simulation paused at index %d.", self._current_index)

    def resume(self) -> None:
        """Resume from paused state."""
        if self._state == self.PAUSED:
            self._state = self.RUNNING
            logger.info("Simulation resumed from index %d.", self._current_index)

    def stop(self) -> None:
        """Stop simulation."""
        self._state = self.STOPPED
        logger.info("Simulation stopped.")

    def get_engine_history(self, engine_id: int) -> List[Dict[str, Any]]:
        """Return all events for a specific engine as list of dicts."""
        return [e.to_dict() for e in self._events if e.engine_id == engine_id]

    def get_fleet_summary(self) -> Dict[str, int]:
        """Return current severity count across latest events per engine."""
        if not self._events:
            return {}
        # Latest event per engine
        latest: Dict[int, LiveCycleEvent] = {}
        for ev in self._events:
            latest[ev.engine_id] = ev
        counts: Dict[str, int] = {}
        for ev in latest.values():
            counts[ev.severity] = counts.get(ev.severity, 0) + 1
        return counts

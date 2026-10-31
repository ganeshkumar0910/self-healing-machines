"""Autonomous rule-based maintenance triage agent — v2 with uncertainty awareness.

Improvements over v1:
- All thresholds read from config (no hardcoded defaults except safe fallbacks)
- Prediction uncertainty field added to TriageDecision
- RULE_UNCERTAINTY_RISK: high-uncertainty engines flagged for inspection
- Signed error and late-prediction tracking
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TriageDecision:
    """Structured decision output from the autonomous triage agent."""

    engine_id: int
    cycle: int
    predicted_rul: float
    anomaly_score: float
    anomaly_threshold: float
    severity: str
    recommended_action: str
    triggered_rule: str
    explanation: str
    model_version: str
    timestamp: str
    # v2 additions
    rul_uncertainty: float = 0.0          # ensemble std proxy; 0 if not available
    rul_lower: float = 0.0                # 90% lower bound; 0 if not available
    rul_upper: float = 0.0                # 90% upper bound; 0 if not available

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary representation."""
        return asdict(self)


class TriageAgent:
    """Automated operational triage agent that maps RUL and anomaly metrics to actions.

    Applies deterministic priority rules to classify machine health and prescribe
    targeted maintenance actions. All thresholds are configurable via settings.yaml.

    Rule priority (evaluated in order — first match wins):
    1. RULE_INVALID_INPUT       — reject bad observations
    2. RULE_CRITICAL_RUL        — predicted_rul <= critical_rul
    3. RULE_WARNING_RUL         — predicted_rul <= warning_rul
    4. RULE_ANOMALY_OVERRIDE    — anomaly_score < anomaly_threshold
    5. RULE_UNCERTAINTY_RISK    — rul_uncertainty > uncertainty_threshold (new in v2)
    6. RULE_NORMAL              — nominal operation
    """

    def __init__(
        self,
        critical_rul: int = 15,
        warning_rul: int = 40,
        anomaly_threshold: float = -0.05,
        uncertainty_threshold: float = 25.0,
        model_version: str = "unknown",
    ) -> None:
        """Initialize TriageAgent.

        Args:
            critical_rul: RUL <= this triggers Emergency Maintenance.
            warning_rul: RUL <= this triggers Planned Maintenance.
            anomaly_threshold: Score below this triggers Inspection Review.
            uncertainty_threshold: Prediction std above this triggers Uncertainty Review.
                                   Set to inf to disable this rule.
            model_version: Model identifier producing the predictions.
        """
        self.critical_rul = critical_rul
        self.warning_rul = warning_rul
        self.anomaly_threshold = anomaly_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.model_version = model_version

    def evaluate(
        self,
        engine_id: Optional[int],
        cycle: Optional[int],
        predicted_rul: Optional[float],
        anomaly_score: Optional[float],
        rul_uncertainty: float = 0.0,
        rul_lower: float = 0.0,
        rul_upper: float = 0.0,
    ) -> TriageDecision:
        """Evaluate a single engine observation against triage decision rules.

        Args:
            engine_id: Engine unit identifier.
            cycle: Current operating cycle.
            predicted_rul: Model-estimated remaining cycles.
            anomaly_score: Isolation Forest anomaly score.
            rul_uncertainty: Ensemble std of RUL predictions (0 if unavailable).
            rul_lower: Lower bound of prediction interval (0 if unavailable).
            rul_upper: Upper bound of prediction interval (0 if unavailable).

        Returns:
            TriageDecision: Full structured maintenance triage record.
        """
        iso_timestamp = datetime.now(timezone.utc).isoformat()

        # ── Rule 1: Validate input integrity ─────────────────────────────────
        is_invalid = False
        reasons: List[str] = []

        if engine_id is None or (isinstance(engine_id, float) and np.isnan(engine_id)) or engine_id < 1:
            is_invalid = True
            reasons.append(f"Invalid engine_id: {engine_id}")

        if cycle is None or (isinstance(cycle, float) and np.isnan(cycle)) or cycle < 1:
            is_invalid = True
            reasons.append(f"Invalid cycle: {cycle}")

        if predicted_rul is None or (isinstance(predicted_rul, float) and np.isnan(predicted_rul)):
            is_invalid = True
            reasons.append(f"Invalid predicted_rul: {predicted_rul}")

        if anomaly_score is None or (isinstance(anomaly_score, float) and np.isnan(anomaly_score)):
            is_invalid = True
            reasons.append(f"Invalid anomaly_score: {anomaly_score}")

        if is_invalid:
            clean_engine = int(engine_id) if (engine_id is not None and not (isinstance(engine_id, float) and np.isnan(engine_id))) else -1
            clean_cycle = int(cycle) if (cycle is not None and not (isinstance(cycle, float) and np.isnan(cycle))) else -1
            clean_rul = float(predicted_rul) if (predicted_rul is not None and not (isinstance(predicted_rul, float) and np.isnan(predicted_rul))) else -1.0
            clean_score = float(anomaly_score) if (anomaly_score is not None and not (isinstance(anomaly_score, float) and np.isnan(anomaly_score))) else 0.0
            return TriageDecision(
                engine_id=clean_engine, cycle=clean_cycle,
                predicted_rul=clean_rul, anomaly_score=clean_score,
                anomaly_threshold=self.anomaly_threshold,
                severity="DataQuality",
                recommended_action="Reject observation — data quality issue",
                triggered_rule="RULE_INVALID_INPUT",
                explanation=f"Data quality defect: {'; '.join(reasons)}.",
                model_version=self.model_version, timestamp=iso_timestamp,
                rul_uncertainty=float(rul_uncertainty),
                rul_lower=float(rul_lower), rul_upper=float(rul_upper),
            )

        # Safe casts
        eng = int(engine_id)
        cyc = int(cycle)
        rul = float(max(0.0, predicted_rul))   # floor at zero
        score = float(anomaly_score)
        uncertainty = float(rul_uncertainty)
        lower = float(rul_lower)
        upper = float(rul_upper)

        # ── Rule 2: Critical RUL ─────────────────────────────────────────────
        if rul <= self.critical_rul:
            return TriageDecision(
                engine_id=eng, cycle=cyc, predicted_rul=rul, anomaly_score=score,
                anomaly_threshold=self.anomaly_threshold, severity="Critical",
                recommended_action="Schedule emergency maintenance immediately",
                triggered_rule="RULE_CRITICAL_RUL",
                explanation=(
                    f"Engine {eng} predicted RUL ({rul:.1f} cycles) is at or below "
                    f"critical threshold ({self.critical_rul} cycles). "
                    "Emergency maintenance required immediately."
                ),
                model_version=self.model_version, timestamp=iso_timestamp,
                rul_uncertainty=uncertainty, rul_lower=lower, rul_upper=upper,
            )

        # ── Rule 3: Warning RUL ──────────────────────────────────────────────
        if rul <= self.warning_rul:
            return TriageDecision(
                engine_id=eng, cycle=cyc, predicted_rul=rul, anomaly_score=score,
                anomaly_threshold=self.anomaly_threshold, severity="Warning",
                recommended_action="Schedule planned maintenance within 2 maintenance windows",
                triggered_rule="RULE_WARNING_RUL",
                explanation=(
                    f"Engine {eng} predicted RUL ({rul:.1f} cycles) is within warning "
                    f"threshold (<= {self.warning_rul} cycles). Schedule planned maintenance."
                ),
                model_version=self.model_version, timestamp=iso_timestamp,
                rul_uncertainty=uncertainty, rul_lower=lower, rul_upper=upper,
            )

        # ── Rule 4: Anomaly override ─────────────────────────────────────────
        if score < self.anomaly_threshold:
            return TriageDecision(
                engine_id=eng, cycle=cyc, predicted_rul=rul, anomaly_score=score,
                anomaly_threshold=self.anomaly_threshold, severity="InspectionReview",
                recommended_action="Perform targeted sensor inspection and manual review",
                triggered_rule="RULE_ANOMALY_OVERRIDE",
                explanation=(
                    f"Engine {eng} anomaly score ({score:.4f}) below threshold "
                    f"({self.anomaly_threshold:.4f}) despite high RUL ({rul:.1f} cycles). "
                    "Sensor drift or abnormal operating state suspected."
                ),
                model_version=self.model_version, timestamp=iso_timestamp,
                rul_uncertainty=uncertainty, rul_lower=lower, rul_upper=upper,
            )

        # ── Rule 5: Uncertainty-based inspection (new in v2) ─────────────────
        if uncertainty > 0.0 and uncertainty > self.uncertainty_threshold:
            return TriageDecision(
                engine_id=eng, cycle=cyc, predicted_rul=rul, anomaly_score=score,
                anomaly_threshold=self.anomaly_threshold, severity="UncertaintyReview",
                recommended_action="Verify sensor calibration — high prediction uncertainty",
                triggered_rule="RULE_UNCERTAINTY_RISK",
                explanation=(
                    f"Engine {eng} has high prediction uncertainty "
                    f"(std={uncertainty:.1f} cycles, interval [{lower:.1f}, {upper:.1f}]). "
                    f"Mean RUL={rul:.1f} cycles — recommend additional inspection."
                ),
                model_version=self.model_version, timestamp=iso_timestamp,
                rul_uncertainty=uncertainty, rul_lower=lower, rul_upper=upper,
            )

        # ── Rule 6: Normal operation ─────────────────────────────────────────
        return TriageDecision(
            engine_id=eng, cycle=cyc, predicted_rul=rul, anomaly_score=score,
            anomaly_threshold=self.anomaly_threshold, severity="Normal",
            recommended_action="Continue standard monitoring",
            triggered_rule="RULE_NORMAL",
            explanation=(
                f"Engine {eng} operating normally (predicted RUL: {rul:.1f} cycles, "
                f"anomaly score: {score:.4f}). No maintenance action needed."
            ),
            model_version=self.model_version, timestamp=iso_timestamp,
            rul_uncertainty=uncertainty, rul_lower=lower, rul_upper=upper,
        )

    def evaluate_batch(self, df: pd.DataFrame) -> List[TriageDecision]:
        """Evaluate a batch of engine observations.

        Args:
            df: DataFrame with engine_id, cycle, predicted_rul, anomaly_score.
                Optional: rul_uncertainty, rul_lower, rul_upper.

        Returns:
            list[TriageDecision]: One decision per row.
        """
        required_cols = ["engine_id", "cycle", "predicted_rul", "anomaly_score"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Input DataFrame missing required columns: {missing}")

        decisions: List[TriageDecision] = []
        for _, row in df.iterrows():
            decision = self.evaluate(
                engine_id=row.get("engine_id"),
                cycle=row.get("cycle"),
                predicted_rul=row.get("predicted_rul"),
                anomaly_score=row.get("anomaly_score"),
                rul_uncertainty=float(row.get("rul_uncertainty", 0.0) or 0.0),
                rul_lower=float(row.get("rul_lower", 0.0) or 0.0),
                rul_upper=float(row.get("rul_upper", 0.0) or 0.0),
            )
            decisions.append(decision)

        severity_counts = {}
        for d in decisions:
            severity_counts[d.severity] = severity_counts.get(d.severity, 0) + 1
        logger.info(
            "Batch triage complete for %d observations: %s",
            len(decisions), severity_counts,
        )
        return decisions

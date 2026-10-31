"""Autonomous rule-based maintenance triage agent for operational decision making."""

from dataclasses import asdict, dataclass
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary representation."""
        return asdict(self)


class TriageAgent:
    """Automated operational triage agent that maps RUL and anomaly metrics to actions.

    Applies deterministic priority rules to classify machine health and prescribe
    targeted maintenance actions.
    """

    def __init__(
        self,
        critical_rul: int = 15,
        warning_rul: int = 40,
        anomaly_threshold: float = -0.05,
        model_version: str = "unknown",
    ) -> None:
        """Initialize TriageAgent.

        Args:
            critical_rul: RUL cycles threshold below which Emergency Maintenance is triggered.
            warning_rul: RUL cycles threshold below which Planned Maintenance is scheduled.
            anomaly_threshold: Anomaly score below which Inspection Review is triggered.
            model_version: Model identifier or version producing the predictions.
        """
        self.critical_rul = critical_rul
        self.warning_rul = warning_rul
        self.anomaly_threshold = anomaly_threshold
        self.model_version = model_version

    def evaluate(
        self,
        engine_id: Optional[int],
        cycle: Optional[int],
        predicted_rul: Optional[float],
        anomaly_score: Optional[float],
    ) -> TriageDecision:
        """Evaluate a single engine telemetry observation against triage decision rules.

        Evaluation Priority:
        1. RULE_INVALID_INPUT: Missing, NaN, or non-physical parameters.
        2. RULE_CRITICAL_RUL: predicted_rul <= critical_rul -> Critical.
        3. RULE_WARNING_RUL: predicted_rul <= warning_rul -> Warning.
        4. RULE_ANOMALY_OVERRIDE: anomaly_score < anomaly_threshold -> InspectionReview.
        5. RULE_NORMAL: Nominal operation -> Normal.

        Args:
            engine_id: Engine unit identifier.
            cycle: Current operating cycle.
            predicted_rul: Model-estimated remaining cycles.
            anomaly_score: Isolation Forest anomaly score.

        Returns:
            TriageDecision: Full structured maintenance triage record.
        """
        iso_timestamp = datetime.now(timezone.utc).isoformat()

        # Rule 1: Validate input integrity
        is_invalid = False
        reasons = []

        if engine_id is None or pd.isna(engine_id) or engine_id < 1:
            is_invalid = True
            reasons.append(f"Invalid engine_id: {engine_id}")

        if cycle is None or pd.isna(cycle) or cycle < 1:
            is_invalid = True
            reasons.append(f"Invalid cycle: {cycle}")

        if predicted_rul is None or pd.isna(predicted_rul):
            is_invalid = True
            reasons.append(f"Invalid predicted_rul: {predicted_rul}")

        if anomaly_score is None or pd.isna(anomaly_score):
            is_invalid = True
            reasons.append(f"Invalid anomaly_score: {anomaly_score}")

        if is_invalid:
            clean_engine = int(engine_id) if (engine_id is not None and not pd.isna(engine_id)) else -1
            clean_cycle = int(cycle) if (cycle is not None and not pd.isna(cycle)) else -1
            clean_rul = float(predicted_rul) if (predicted_rul is not None and not pd.isna(predicted_rul)) else -1.0
            clean_score = float(anomaly_score) if (anomaly_score is not None and not pd.isna(anomaly_score)) else 0.0

            return TriageDecision(
                engine_id=clean_engine,
                cycle=clean_cycle,
                predicted_rul=clean_rul,
                anomaly_score=clean_score,
                anomaly_threshold=self.anomaly_threshold,
                severity="DataQuality",
                recommended_action="Reject observation — data quality issue",
                triggered_rule="RULE_INVALID_INPUT",
                explanation=f"Data quality defect: {'; '.join(reasons)}.",
                model_version=self.model_version,
                timestamp=iso_timestamp,
            )

        # Cast safe types
        eng = int(engine_id)
        cyc = int(cycle)
        rul = float(predicted_rul)
        score = float(anomaly_score)

        # Rule 2: Critical RUL
        if rul <= self.critical_rul:
            return TriageDecision(
                engine_id=eng,
                cycle=cyc,
                predicted_rul=rul,
                anomaly_score=score,
                anomaly_threshold=self.anomaly_threshold,
                severity="Critical",
                recommended_action="Schedule emergency maintenance immediately",
                triggered_rule="RULE_CRITICAL_RUL",
                explanation=(
                    f"Engine {eng} predicted RUL ({rul:.1f} cycles) is at or below critical threshold "
                    f"({self.critical_rul} cycles). Emergency maintenance required immediately."
                ),
                model_version=self.model_version,
                timestamp=iso_timestamp,
            )

        # Rule 3: Warning RUL
        if rul <= self.warning_rul:
            return TriageDecision(
                engine_id=eng,
                cycle=cyc,
                predicted_rul=rul,
                anomaly_score=score,
                anomaly_threshold=self.anomaly_threshold,
                severity="Warning",
                recommended_action="Schedule planned maintenance within 2 maintenance windows",
                triggered_rule="RULE_WARNING_RUL",
                explanation=(
                    f"Engine {eng} predicted RUL ({rul:.1f} cycles) is within warning threshold "
                    f"(<= {self.warning_rul} cycles). Schedule planned maintenance."
                ),
                model_version=self.model_version,
                timestamp=iso_timestamp,
            )

        # Rule 4: Anomaly Override
        if score < self.anomaly_threshold:
            return TriageDecision(
                engine_id=eng,
                cycle=cyc,
                predicted_rul=rul,
                anomaly_score=score,
                anomaly_threshold=self.anomaly_threshold,
                severity="InspectionReview",
                recommended_action="Perform targeted sensor inspection and manual review",
                triggered_rule="RULE_ANOMALY_OVERRIDE",
                explanation=(
                    f"Engine {eng} anomaly score ({score:.4f}) dropped below threshold "
                    f"({self.anomaly_threshold:.4f}) despite high RUL ({rul:.1f} cycles). "
                    f"Sensor drift or abnormal state suspected."
                ),
                model_version=self.model_version,
                timestamp=iso_timestamp,
            )

        # Rule 5: Nominal Operation
        return TriageDecision(
            engine_id=eng,
            cycle=cyc,
            predicted_rul=rul,
            anomaly_score=score,
            anomaly_threshold=self.anomaly_threshold,
            severity="Normal",
            recommended_action="Continue standard monitoring",
            triggered_rule="RULE_NORMAL",
            explanation=(
                f"Engine {eng} operating normally (predicted RUL: {rul:.1f} cycles, "
                f"anomaly score: {score:.4f}). No maintenance action needed."
            ),
            model_version=self.model_version,
            timestamp=iso_timestamp,
        )

    def evaluate_batch(self, df: pd.DataFrame) -> List[TriageDecision]:
        """Evaluate a batch of engine observations.

        Args:
            df: DataFrame containing engine_id, cycle, predicted_rul, and anomaly_score.

        Returns:
            list[TriageDecision]: List of evaluation decisions corresponding to each row.
        """
        required_cols = ["engine_id", "cycle", "predicted_rul", "anomaly_score"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Input DataFrame is missing required columns: {missing}")

        decisions = []
        for _, row in df.iterrows():
            decision = self.evaluate(
                engine_id=row.get("engine_id"),
                cycle=row.get("cycle"),
                predicted_rul=row.get("predicted_rul"),
                anomaly_score=row.get("anomaly_score"),
            )
            decisions.append(decision)

        logger.info("Batch triage evaluation completed for %d observations.", len(decisions))
        return decisions

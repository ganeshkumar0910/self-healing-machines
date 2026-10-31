"""Unit tests for the autonomous TriageAgent and rule-based decision engine."""

from pathlib import Path
import sys

# Ensure project root is available on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.agent.triage_agent import TriageAgent, TriageDecision


@pytest.fixture
def default_agent() -> TriageAgent:
    """Fixture providing a default TriageAgent instance."""
    return TriageAgent(
        critical_rul=15,
        warning_rul=40,
        anomaly_threshold=-0.05,
        model_version="v1.0.0-test",
    )


def test_critical_rul_boundary(default_agent: TriageAgent) -> None:
    """Test boundary behavior around critical RUL threshold (15 cycles)."""
    # At boundary
    d_at = default_agent.evaluate(engine_id=1, cycle=100, predicted_rul=15.0, anomaly_score=0.1)
    assert d_at.severity == "Critical"
    assert d_at.triggered_rule == "RULE_CRITICAL_RUL"

    # Below boundary
    d_below = default_agent.evaluate(engine_id=1, cycle=100, predicted_rul=14.0, anomaly_score=0.1)
    assert d_below.severity == "Critical"
    assert d_below.triggered_rule == "RULE_CRITICAL_RUL"

    # Just above boundary
    d_above = default_agent.evaluate(engine_id=1, cycle=100, predicted_rul=16.0, anomaly_score=0.1)
    assert d_above.severity == "Warning"
    assert d_above.triggered_rule == "RULE_WARNING_RUL"


def test_warning_rul_boundary(default_agent: TriageAgent) -> None:
    """Test boundary behavior around warning RUL threshold (40 cycles)."""
    # At warning boundary
    d_at = default_agent.evaluate(engine_id=2, cycle=80, predicted_rul=40.0, anomaly_score=0.1)
    assert d_at.severity == "Warning"
    assert d_at.triggered_rule == "RULE_WARNING_RUL"

    # Just above warning boundary with nominal anomaly score
    d_above = default_agent.evaluate(engine_id=2, cycle=80, predicted_rul=41.0, anomaly_score=0.1)
    assert d_above.severity == "Normal"
    assert d_above.triggered_rule == "RULE_NORMAL"


def test_normal_status(default_agent: TriageAgent) -> None:
    """Test nominal engine operation with high RUL and non-anomalous score."""
    decision = default_agent.evaluate(
        engine_id=3, cycle=20, predicted_rul=100.0, anomaly_score=0.15
    )
    assert decision.severity == "Normal"
    assert decision.recommended_action == "Continue standard monitoring"
    assert decision.triggered_rule == "RULE_NORMAL"


def test_anomaly_override(default_agent: TriageAgent) -> None:
    """Test sensor anomaly override condition when RUL appears healthy."""
    decision = default_agent.evaluate(
        engine_id=4, cycle=50, predicted_rul=50.0, anomaly_score=-0.10
    )
    assert decision.severity == "InspectionReview"
    assert "Perform targeted sensor inspection" in decision.recommended_action
    assert decision.triggered_rule == "RULE_ANOMALY_OVERRIDE"


def test_invalid_input_nan(default_agent: TriageAgent) -> None:
    """Test rejection when predicted RUL is NaN."""
    decision = default_agent.evaluate(
        engine_id=5, cycle=10, predicted_rul=float("nan"), anomaly_score=0.05
    )
    assert decision.severity == "DataQuality"
    assert decision.triggered_rule == "RULE_INVALID_INPUT"
    assert "Reject observation" in decision.recommended_action


def test_invalid_input_negative_cycle(default_agent: TriageAgent) -> None:
    """Test rejection when operating cycle is non-physical (< 1)."""
    decision = default_agent.evaluate(
        engine_id=6, cycle=-1, predicted_rul=80.0, anomaly_score=0.05
    )
    assert decision.severity == "DataQuality"
    assert decision.triggered_rule == "RULE_INVALID_INPUT"


def test_invalid_input_none(default_agent: TriageAgent) -> None:
    """Test rejection when predicted RUL is None."""
    decision = default_agent.evaluate(
        engine_id=7, cycle=15, predicted_rul=None, anomaly_score=0.05
    )
    assert decision.severity == "DataQuality"
    assert decision.triggered_rule == "RULE_INVALID_INPUT"


def test_decision_fields_complete(default_agent: TriageAgent) -> None:
    """Verify that all required dataclass attributes are populated on decision output."""
    decision = default_agent.evaluate(
        engine_id=1, cycle=10, predicted_rul=12.0, anomaly_score=-0.02
    )
    required_fields = [
        "engine_id",
        "cycle",
        "predicted_rul",
        "anomaly_score",
        "anomaly_threshold",
        "severity",
        "recommended_action",
        "triggered_rule",
        "explanation",
        "model_version",
        "timestamp",
    ]
    for field in required_fields:
        assert hasattr(decision, field), f"Missing field: {field}"
        val = getattr(decision, field)
        assert val is not None, f"Field {field} is None"

    assert len(decision.explanation) > 0
    assert len(decision.timestamp) > 0


def test_batch_evaluation(default_agent: TriageAgent) -> None:
    """Verify batch evaluation over a DataFrame of multiple engines."""
    df = pd.DataFrame(
        [
            {"engine_id": 1, "cycle": 150, "predicted_rul": 10.0, "anomaly_score": 0.05},
            {"engine_id": 2, "cycle": 120, "predicted_rul": 30.0, "anomaly_score": 0.05},
            {"engine_id": 3, "cycle": 80, "predicted_rul": 60.0, "anomaly_score": -0.08},
            {"engine_id": 4, "cycle": 40, "predicted_rul": 90.0, "anomaly_score": 0.10},
            {"engine_id": 5, "cycle": -5, "predicted_rul": 75.0, "anomaly_score": 0.02},
        ]
    )
    decisions = default_agent.evaluate_batch(df)
    assert len(decisions) == 5

    severities = [d.severity for d in decisions]
    assert severities == ["Critical", "Warning", "InspectionReview", "Normal", "DataQuality"]


def test_custom_thresholds() -> None:
    """Verify customized threshold parameters are correctly respected by agent."""
    custom_agent = TriageAgent(
        critical_rul=20,
        warning_rul=50,
        anomaly_threshold=-0.02,
        model_version="custom-v2",
    )
    # RUL=18 was Warning under default (15), now Critical under custom (20)
    decision = custom_agent.evaluate(engine_id=10, cycle=50, predicted_rul=18.0, anomaly_score=0.1)
    assert decision.severity == "Critical"
    assert decision.model_version == "custom-v2"

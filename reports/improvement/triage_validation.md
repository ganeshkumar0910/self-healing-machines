# Triage Agent Validation Report — v2
# Self-Healing Machines | Rule-Based Deterministic Triage

---

## Overview

The triage agent applies a deterministic, explainable rule-based decision system to classify
engine health and prescribe targeted maintenance actions. No LLM or probabilistic model is used.

All thresholds are configurable via `config/settings.yaml`. The agent evaluates rules in
strict priority order — first matching rule wins.

---

## Rule Priority Table (v2)

| Priority | Rule ID | Condition | Severity | Action |
|----------|---------|-----------|----------|--------|
| 1 | RULE_INVALID_INPUT | Missing/null/negative inputs | DataQuality | Reject observation |
| 2 | RULE_CRITICAL_RUL | predicted_rul ≤ 15 cycles | Critical | Emergency maintenance immediately |
| 3 | RULE_WARNING_RUL | predicted_rul ≤ 40 cycles | Warning | Planned maintenance within 2 windows |
| 4 | RULE_ANOMALY_OVERRIDE | anomaly_score < −0.05 | InspectionReview | Sensor inspection + manual review |
| 5 | RULE_UNCERTAINTY_RISK *(NEW v2)* | rul_uncertainty > 25 cycles | UncertaintyReview | Verify calibration — high uncertainty |
| 6 | RULE_NORMAL | None above triggered | Normal | Continue standard monitoring |

---

## Configuration (config/settings.yaml)

```yaml
triage:
  critical_rul_threshold: 15      # RUL <= this -> Critical
  warning_rul_threshold: 40       # RUL <= this -> Warning
  anomaly_score_threshold: -0.05  # Score below this -> InspectionReview
  uncertainty_threshold: 25.0     # Ensemble std above this -> UncertaintyReview
                                  # Set to 999 to disable
```

---

## New in v2: RULE_UNCERTAINTY_RISK

### Motivation
The ensemble of 6 CPU models produces uncertainty estimates (std of predictions).
When models disagree significantly (std > 25 cycles), the individual prediction may be
unreliable despite a nominal RUL value. The UncertaintyReview rule catches these cases.

### Conditions for triggering
- predicted_rul > warning_rul_threshold (otherwise absorbed by Rules 2/3)
- rul_uncertainty > 25.0 cycles (configurable)
- anomaly_score >= anomaly_score_threshold (otherwise absorbed by Rule 4)

### Example output
```
Engine 47: predicted RUL=82.1 cycles, std=31.4 cycles
→ Triggered: RULE_UNCERTAINTY_RISK
→ Severity: UncertaintyReview
→ Action: Verify sensor calibration — high prediction uncertainty
→ Interval: [30.5, 133.7] cycles (90% prediction interval)
```

---

## TriageDecision v2 Schema

```python
@dataclass
class TriageDecision:
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
    rul_uncertainty: float = 0.0   # ensemble std
    rul_lower: float = 0.0         # 90% lower bound
    rul_upper: float = 0.0         # 90% upper bound
```

---

## Test Coverage

All triage rules tested in `tests/test_improvements.py::TestTriageAgentV2`:

| Test | Rule | Status |
|------|------|--------|
| test_critical_rul_rule | RULE_CRITICAL_RUL | PASSED |
| test_warning_rul_rule | RULE_WARNING_RUL | PASSED |
| test_anomaly_override_rule | RULE_ANOMALY_OVERRIDE | PASSED |
| test_uncertainty_risk_rule | RULE_UNCERTAINTY_RISK | PASSED |
| test_normal_rule | RULE_NORMAL | PASSED |
| test_invalid_input_rule | RULE_INVALID_INPUT | PASSED |
| test_negative_rul_floored_to_zero | Prediction floor | PASSED |
| test_uncertainty_fields_in_decision | v2 schema | PASSED |
| test_batch_returns_one_per_row | Batch evaluation | PASSED |

Plus 10 original tests from `tests/test_triage_agent.py` — all passing.

---

## Prediction Floor (Safety)

All predictions are floored at 0 before rule evaluation:
```python
rul = float(max(0.0, predicted_rul))
```

A model predicting −5 cycles (impossible) is clipped to 0 → triggers RULE_CRITICAL_RUL.
This prevents unsafe "negative RUL" states from bypassing emergency rules.

---

## Design Principles

1. **Deterministic** — same inputs always produce same outputs. No randomness.
2. **Explainable** — every decision includes plain-English `explanation` field.
3. **Configurable** — all thresholds in `config/settings.yaml`, no hardcoded values.
4. **Fail-safe** — invalid inputs trigger DataQuality rejection rather than silent failure.
5. **Auditable** — complete `TriageDecision` record saved to database and CSV.
6. **No LLM** — fully rule-based as per user requirement.

---

## Severity Distribution (FD001 Test Fleet Run)

Actual output from pipeline run on 100 test engines (October 31, 2026):

| Severity | Count | % | Root Cause |
|----------|-------|---|------------|
| Critical | 100 | 100% | RF overfits val endpoints → predicts near-zero RUL on test |
| Warning | 0 | 0% | — |
| Normal | 0 | 0% | — |

> **Note:** 100% Critical is a consequence of Random Forest overfitting 20 validation
> endpoints (achieving MAE=0.000 on val but predicting near-zero for all test engines).
> Linear Regression (best generalization, test EP-MAE=35.91) would produce a healthier
> triage distribution. Champion selection should incorporate a minimum generalization check
> in future versions.

"""Unit and integration tests for WorkOrderService and DatabaseManager."""

from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure project root is available on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.agent.triage_agent import TriageDecision
from src.database.db_manager import DatabaseManager
from src.maintenance.work_order_service import WorkOrderService


@pytest.fixture
def service(tmp_path: Path) -> WorkOrderService:
    """Fixture providing a fresh WorkOrderService backed by an isolated temporary database."""
    db_file = tmp_path / "test_maintenance.db"
    db_manager = DatabaseManager(db_file)
    return WorkOrderService(db_manager)


def create_sample_decision(
    engine_id: int = 1,
    cycle: int = 100,
    severity: str = "Critical",
    predicted_rul: float = 12.0,
    anomaly_score: float = 0.05,
) -> TriageDecision:
    """Helper to instantiate mock TriageDecision instances."""
    return TriageDecision(
        engine_id=engine_id,
        cycle=cycle,
        predicted_rul=predicted_rul,
        anomaly_score=anomaly_score,
        anomaly_threshold=-0.05,
        severity=severity,
        recommended_action="Emergency inspection",
        triggered_rule=f"RULE_{severity.upper()}",
        explanation="Automated test explanation",
        model_version="test-v1",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def test_create_work_order_critical(service: WorkOrderService) -> None:
    """Verify successful creation of a Critical severity work order."""
    decision = create_sample_decision(engine_id=1, cycle=120, severity="Critical")
    res = service.create_work_order(decision)

    assert res["created"] is True
    assert res["wo_id"].startswith("WO-1-120-CRI-")

    fetched = service.get_work_order_by_id(res["wo_id"])
    assert fetched is not None
    assert fetched["engine_id"] == 1
    assert fetched["cycle"] == 120
    assert fetched["severity"] == "Critical"
    assert fetched["status"] == "Open"


def test_no_wo_for_normal(service: WorkOrderService) -> None:
    """Verify that work orders are not created for Normal or DataQuality severities."""
    d_normal = create_sample_decision(engine_id=2, cycle=50, severity="Normal")
    res_normal = service.create_work_order(d_normal)
    assert res_normal["created"] is False
    assert res_normal["wo_id"] == ""

    d_dq = create_sample_decision(engine_id=3, cycle=10, severity="DataQuality")
    res_dq = service.create_work_order(d_dq)
    assert res_dq["created"] is False
    assert res_dq["wo_id"] == ""

    orders = service.get_work_orders()
    assert len(orders) == 0


def test_duplicate_prevention(service: WorkOrderService) -> None:
    """Verify duplicate work orders for same engine, cycle, and severity are suppressed."""
    decision = create_sample_decision(engine_id=4, cycle=200, severity="Warning")

    res1 = service.create_work_order(decision)
    assert res1["created"] is True

    res2 = service.create_work_order(decision)
    assert res2["created"] is False
    assert res2["wo_id"] == res1["wo_id"]

    all_orders = service.get_work_orders(engine_id=4)
    assert len(all_orders) == 1


def test_status_update(service: WorkOrderService) -> None:
    """Verify valid progression of work order status from Open -> InProgress -> Closed."""
    decision = create_sample_decision(engine_id=5, cycle=110, severity="Critical")
    res = service.create_work_order(decision)
    wo_id = res["wo_id"]

    # Open -> InProgress
    ok = service.update_status(wo_id, "InProgress", notes="Technician dispatched.")
    assert ok is True
    wo = service.get_work_order_by_id(wo_id)
    assert wo["status"] == "InProgress"

    # InProgress -> Closed
    ok2 = service.update_status(wo_id, "Closed", notes="Component replaced and validated.")
    assert ok2 is True
    wo2 = service.get_work_order_by_id(wo_id)
    assert wo2["status"] == "Closed"


def test_invalid_status_rejected(service: WorkOrderService) -> None:
    """Verify that unsupported status values raise a ValueError."""
    decision = create_sample_decision(engine_id=6, cycle=100, severity="Warning")
    res = service.create_work_order(decision)
    wo_id = res["wo_id"]

    with pytest.raises(ValueError, match="Invalid status"):
        service.update_status(wo_id, "PendingApproval")


def test_filter_by_severity(service: WorkOrderService) -> None:
    """Verify filtering work orders by severity."""
    service.create_work_order(create_sample_decision(engine_id=1, cycle=10, severity="Critical"))
    service.create_work_order(create_sample_decision(engine_id=2, cycle=20, severity="Warning"))
    service.create_work_order(create_sample_decision(engine_id=3, cycle=30, severity="InspectionReview"))

    crit_list = service.get_work_orders(severity="Critical")
    assert len(crit_list) == 1
    assert crit_list[0]["engine_id"] == 1

    warn_list = service.get_work_orders(severity="Warning")
    assert len(warn_list) == 1
    assert warn_list[0]["engine_id"] == 2


def test_filter_by_engine(service: WorkOrderService) -> None:
    """Verify filtering work orders by specific engine ID."""
    service.create_work_order(create_sample_decision(engine_id=7, cycle=10, severity="Critical"))
    service.create_work_order(create_sample_decision(engine_id=7, cycle=20, severity="Warning"))
    service.create_work_order(create_sample_decision(engine_id=8, cycle=15, severity="Critical"))

    engine_7 = service.get_work_orders(engine_id=7)
    assert len(engine_7) == 2

    engine_8 = service.get_work_orders(engine_id=8)
    assert len(engine_8) == 1


def test_audit_history_recorded(service: WorkOrderService) -> None:
    """Verify immutable audit transitions are stored in wo_history."""
    decision = create_sample_decision(engine_id=9, cycle=50, severity="Critical")
    res = service.create_work_order(decision)
    wo_id = res["wo_id"]

    service.update_status(wo_id, "InProgress", notes="Initial diagnosis")
    service.update_status(wo_id, "Closed", notes="Fixed turbine blades")

    history = service.get_work_order_history(wo_id)
    assert len(history) == 3

    assert history[0]["old_status"] is None
    assert history[0]["new_status"] == "Open"

    assert history[1]["old_status"] == "Open"
    assert history[1]["new_status"] == "InProgress"
    assert history[1]["notes"] == "Initial diagnosis"

    assert history[2]["old_status"] == "InProgress"
    assert history[2]["new_status"] == "Closed"
    assert history[2]["notes"] == "Fixed turbine blades"


def test_fleet_summary(service: WorkOrderService) -> None:
    """Verify fleet-wide status and severity summary aggregation."""
    service.create_work_order(create_sample_decision(engine_id=1, cycle=10, severity="Critical"))
    service.create_work_order(create_sample_decision(engine_id=2, cycle=20, severity="Critical"))
    res_w = service.create_work_order(create_sample_decision(engine_id=3, cycle=30, severity="Warning"))
    res_i = service.create_work_order(create_sample_decision(engine_id=4, cycle=40, severity="InspectionReview"))

    service.update_status(res_w["wo_id"], "InProgress")
    service.update_status(res_i["wo_id"], "Closed")

    summary = service.get_fleet_summary()
    assert summary["total"] == 4
    assert summary["open"] == 2
    assert summary["in_progress"] == 1
    assert summary["closed"] == 1

    assert summary["by_severity"]["Critical"] == 2
    assert summary["by_severity"]["Warning"] == 1
    assert summary["by_severity"]["InspectionReview"] == 1

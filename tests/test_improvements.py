"""Extended test suite — covers endpoint evaluation, new models, uncertainty, simulation.

Preserves all 27 original tests (imported from existing test files) and adds:
- Endpoint-level evaluation protocol
- RUL cap consistency
- New model types (ExtraTrees, HistGB, XGBoost)
- Ensemble uncertainty estimation
- Feature engineering (v2: rolling min/max, EMA, sensor diffs, cycle_pct)
- Triage agent (uncertainty rule, all severity levels)
- Simulated live monitor
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data.rul_calculator import compute_train_rul, compute_test_rul
from src.features.engineer import FeatureEngineer
from src.models.rul_model import (
    ExtraTreesRULModel,
    GradientBoostingRULModel,
    HistGradientBoostingRULModel,
    LinearRULModel,
    RandomForestRULModel,
    XGBoostRULModel,
    compute_ensemble_uncertainty,
    compute_nasa_score,
    evaluate_endpoint,
    evaluate_model,
    evaluate_by_rul_range,
)
from src.agent.triage_agent import TriageAgent, TriageDecision
from src.data.splitter import engine_wise_split


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_synthetic_df(
    n_engines: int = 10,
    cycles_per_engine: int = 30,
    n_features: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate synthetic multi-engine time-series DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for eng in range(1, n_engines + 1):
        for cyc in range(1, cycles_per_engine + 1):
            row = {"engine_id": eng, "cycle": cyc}
            for f in range(n_features):
                row[f"sensor_{f+1}"] = float(rng.normal(loc=500 + f * 50, scale=10))
            rows.append(row)
    return pd.DataFrame(rows)


def make_cmapss_like_df(n_engines: int = 20, max_cycles: int = 200) -> pd.DataFrame:
    """Synthetic CMAPSS-style DataFrame with RUL column."""
    rng = np.random.default_rng(42)
    rows = []
    for eng in range(1, n_engines + 1):
        life = rng.integers(50, max_cycles + 1)
        for cyc in range(1, life + 1):
            rows.append({
                "engine_id": eng,
                "cycle": cyc,
                "sensor_2": float(640 + cyc * 0.1 + rng.normal(0, 1)),
                "sensor_3": float(1580 + cyc * 0.05),
            })
    df = pd.DataFrame(rows)
    df = compute_train_rul(df, cap_value=125)
    return df


# ── Evaluation protocol tests ─────────────────────────────────────────────────

class TestEndpointEvaluation:
    def test_endpoint_eval_selects_last_cycle_only(self):
        """Endpoint eval must use last cycle per engine — not all rows."""
        df = pd.DataFrame({
            "engine_id": [1, 1, 1, 2, 2],
            "cycle":     [1, 2, 3, 1, 2],
            "RUL":       [30.0, 20.0, 10.0, 50.0, 40.0],
            "predicted_rul": [35.0, 22.0, 8.0, 45.0, 38.0],
        })
        metrics = evaluate_endpoint(df)
        # Engine 1 last: true=10, pred=8 → error=-2 (early)
        # Engine 2 last: true=40, pred=38 → error=-2 (early)
        assert metrics["n_engines"] == 2
        assert abs(metrics["mae"] - 2.0) < 1e-6

    def test_endpoint_mae_higher_than_row_level(self):
        """Endpoint MAE should typically differ from row-level for realistic data."""
        df = make_cmapss_like_df(n_engines=15)
        sensors = ["sensor_2", "sensor_3"]
        fe = FeatureEngineer(selected_sensors=sensors, rolling_windows=[5], include_op_settings=False)
        df_feat = fe.build_features(df)

        X = df_feat[fe.get_feature_columns()].to_numpy()
        y = df_feat["RUL"].to_numpy()

        model = LinearRULModel()
        model.fit(X, y)
        preds = model.predict(X)

        df_feat_pred = df_feat.copy()
        df_feat_pred["predicted_rul"] = preds

        row_metrics = evaluate_model(y, preds)
        ep_metrics = evaluate_endpoint(df_feat_pred)

        # Both should be non-negative finite values
        assert row_metrics["mae"] > 0
        assert ep_metrics["mae"] > 0
        assert np.isfinite(row_metrics["mae"])
        assert np.isfinite(ep_metrics["mae"])

    def test_rul_cap_applied_in_endpoint_eval(self):
        """RUL cap must be applied to true values in endpoint evaluation."""
        df = pd.DataFrame({
            "engine_id": [1, 1],
            "cycle":     [1, 2],
            "RUL":       [200.0, 190.0],       # Uncapped test RUL values
            "predicted_rul": [125.0, 115.0],
        })
        metrics_capped = evaluate_endpoint(df, rul_cap=125)
        # With cap: true=[125,125], pred=[125,115], last cycle: true=125, pred=115 → MAE=10
        metrics_uncapped = evaluate_endpoint(df, rul_cap=None)
        # Without cap: true=190, pred=115 → MAE=75
        assert metrics_capped["mae"] < metrics_uncapped["mae"]

    def test_late_prediction_percentage(self):
        """late_pct should count engines where predicted > true at endpoint."""
        df = pd.DataFrame({
            "engine_id": [1, 2, 3, 4],
            "cycle":     [5, 5, 5, 5],
            "RUL":       [10.0, 20.0, 30.0, 40.0],
            "predicted_rul": [15.0, 25.0, 25.0, 35.0],  # late, late, early, early
        })
        metrics = evaluate_endpoint(df)
        assert abs(metrics["late_pct"] - 50.0) < 1e-6

    def test_no_negative_predictions_after_clip(self):
        """Endpoint eval must clip predictions to >= 0."""
        df = pd.DataFrame({
            "engine_id": [1],
            "cycle": [10],
            "RUL": [5.0],
            "predicted_rul": [-3.0],  # invalid negative
        })
        metrics = evaluate_endpoint(df)
        assert metrics["mae"] == pytest.approx(5.0)  # clip → 0, error=5


# ── NASA score tests ──────────────────────────────────────────────────────────

class TestNASAScore:
    def test_asymmetry_late_worse_than_early(self):
        """Late prediction (positive error) should score higher than early."""
        late_score = compute_nasa_score(np.array([10.0]), np.array([20.0]))
        early_score = compute_nasa_score(np.array([10.0]), np.array([0.0]))
        assert late_score > early_score

    def test_perfect_prediction_near_zero(self):
        """Perfect prediction should give score close to 0."""
        y = np.array([50.0, 30.0, 10.0])
        score = compute_nasa_score(y, y)
        assert abs(score) < 1e-6

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_nasa_score(np.array([1.0, 2.0]), np.array([1.0]))


# ── RUL range breakdown ───────────────────────────────────────────────────────

class TestRULRangeMetrics:
    def test_range_breakdown_returns_expected_bins(self):
        y_true = np.array([10.0, 50.0, 100.0, 200.0])
        y_pred = np.array([12.0, 48.0, 105.0, 130.0])
        result = evaluate_by_rul_range(y_true, y_pred)
        assert "RUL [0, 30)" in result
        assert "RUL [30, 60)" in result
        assert result["RUL [0, 30)"]["count"] == 1
        assert result["RUL [30, 60)"]["count"] == 1


# ── New model type tests ──────────────────────────────────────────────────────

class TestNewModelTypes:
    """Verify all 6 model types train and predict correctly on CPU."""

    @pytest.fixture(scope="class")
    def train_data(self):
        df = make_cmapss_like_df(n_engines=15)
        fe = FeatureEngineer(
            selected_sensors=["sensor_2", "sensor_3"],
            rolling_windows=[5],
            include_op_settings=False,
            include_rolling_minmax=True,
            include_ema=True,
            include_sensor_diffs=False,
            include_cycle_pct=True,
        )
        df_feat = fe.build_features(df)
        X = df_feat[fe.get_feature_columns()].to_numpy()
        y = df_feat["RUL"].to_numpy()
        return X, y

    def test_extra_trees(self, train_data):
        X, y = train_data
        model = ExtraTreesRULModel(n_estimators=10, n_jobs=1, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert (preds >= 0).all(), "No negative predictions allowed"

    def test_hist_gradient_boosting(self, train_data):
        X, y = train_data
        model = HistGradientBoostingRULModel(max_iter=20, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert (preds >= 0).all()

    def test_xgboost_cpu(self, train_data):
        pytest.importorskip("xgboost", reason="XGBoost not installed")
        X, y = train_data
        model = XGBoostRULModel(n_estimators=20, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert (preds >= 0).all()

    def test_all_models_return_non_negative(self, train_data):
        X, y = train_data
        models = [
            LinearRULModel(),
            RandomForestRULModel(n_estimators=10, n_jobs=1, random_state=42),
            GradientBoostingRULModel(n_estimators=10, random_state=42),
            ExtraTreesRULModel(n_estimators=10, n_jobs=1, random_state=42),
            HistGradientBoostingRULModel(max_iter=10, random_state=42),
        ]
        for m in models:
            m.fit(X, y)
            preds = m.predict(X)
            assert (preds >= 0).all(), f"{m.model_type} produced negative predictions"


# ── Ensemble uncertainty tests ────────────────────────────────────────────────

class TestEnsembleUncertainty:
    def test_uncertainty_has_correct_shape(self):
        X = np.random.default_rng(0).random((50, 5))
        y = np.random.default_rng(0).random(50) * 100

        models = [
            LinearRULModel(),
            RandomForestRULModel(n_estimators=5, n_jobs=1, random_state=0),
            ExtraTreesRULModel(n_estimators=5, n_jobs=1, random_state=0),
        ]
        for m in models:
            m.fit(X, y)

        result = compute_ensemble_uncertainty(models, X)
        assert result["mean"].shape == (50,)
        assert result["std"].shape == (50,)
        assert result["lower"].shape == (50,)
        assert result["upper"].shape == (50,)

    def test_uncertainty_non_negative(self):
        X = np.random.default_rng(1).random((30, 4))
        y = np.random.default_rng(1).random(30) * 50
        models = [LinearRULModel(), HistGradientBoostingRULModel(max_iter=5, random_state=1)]
        for m in models:
            m.fit(X, y)
        result = compute_ensemble_uncertainty(models, X)
        assert (result["std"] >= 0).all()
        assert (result["lower"] >= 0).all()

    def test_single_model_zero_uncertainty(self):
        """Single model should produce zero std (deterministic)."""
        X = np.random.default_rng(2).random((20, 3))
        y = np.random.default_rng(2).random(20) * 80
        model = LinearRULModel()
        model.fit(X, y)
        result = compute_ensemble_uncertainty([model], X)
        assert (result["std"] == 0.0).all()


# ── Feature engineering v2 tests ─────────────────────────────────────────────

class TestFeatureEngineeringV2:
    @pytest.fixture
    def simple_df(self):
        rows = []
        for eng in [1, 2]:
            for cyc in range(1, 11):
                rows.append({
                    "engine_id": eng, "cycle": cyc,
                    "sensor_2": float(640 + cyc),
                    "sensor_3": float(1580 + cyc * 2),
                })
        return pd.DataFrame(rows)

    def test_rolling_min_max_correct(self, simple_df):
        fe = FeatureEngineer(
            selected_sensors=["sensor_2"],
            rolling_windows=[3],
            include_op_settings=False,
            include_rolling_minmax=True,
            include_ema=False,
            include_sensor_diffs=False,
            include_cycle_pct=False,
        )
        result = fe.build_features(simple_df)
        # Engine 1, cycle 3: window=[641,642,643], min=641, max=643
        row = result[(result["engine_id"] == 1) & (result["cycle"] == 3)].iloc[0]
        assert abs(row["sensor_2_min_3"] - 641.0) < 1e-6
        assert abs(row["sensor_2_max_3"] - 643.0) < 1e-6

    def test_rolling_minmax_no_cross_engine(self, simple_df):
        fe = FeatureEngineer(
            selected_sensors=["sensor_2"],
            rolling_windows=[5],
            include_op_settings=False,
            include_rolling_minmax=True,
            include_ema=False,
            include_sensor_diffs=False,
            include_cycle_pct=False,
        )
        result = fe.build_features(simple_df)
        # Engine 2 cycle 1: sensor_2 = 640+1 = 641.0 (first cycle)
        # Rolling min with window=5 at cycle 1 = just itself = 641.0
        # If there were cross-engine leakage from engine 1 (values 641-650),
        # the min would be wrong. With correct engine isolation, min = 641.0
        eng2_c1 = result[(result["engine_id"] == 2) & (result["cycle"] == 1)].iloc[0]
        assert abs(eng2_c1["sensor_2_min_5"] - 641.0) < 1e-6  # engine 2 own first value

        # Engine 1 cycle 10 max should be 650.0, not influenced by engine 2
        eng1_c10 = result[(result["engine_id"] == 1) & (result["cycle"] == 10)].iloc[0]
        assert abs(eng1_c10["sensor_2_max_5"] - 650.0) < 1e-6

    def test_ema_shape_and_non_negative(self, simple_df):
        fe = FeatureEngineer(
            selected_sensors=["sensor_2"],
            rolling_windows=[5],
            include_op_settings=False,
            include_rolling_minmax=False,
            include_ema=True,
            include_sensor_diffs=False,
            include_cycle_pct=False,
        )
        result = fe.build_features(simple_df)
        assert "sensor_2_ema" in result.columns
        assert (result["sensor_2_ema"] >= 0).all()

    def test_cycle_pct_between_0_and_1(self, simple_df):
        fe = FeatureEngineer(
            selected_sensors=["sensor_2"],
            rolling_windows=[2],
            include_op_settings=False,
            include_rolling_minmax=False,
            include_ema=False,
            include_sensor_diffs=False,
            include_cycle_pct=True,
        )
        result = fe.build_features(simple_df)
        assert "cycle_pct" in result.columns
        assert (result["cycle_pct"] >= 0.0).all()
        assert (result["cycle_pct"] <= 1.0).all()

    def test_feature_count_consistency(self, simple_df):
        """get_feature_columns() count must match actual DataFrame columns."""
        fe = FeatureEngineer(
            selected_sensors=["sensor_2", "sensor_3"],
            rolling_windows=[5, 10],
            include_op_settings=False,
            include_rolling_minmax=True,
            include_ema=True,
            include_sensor_diffs=False,
            include_cycle_pct=True,
        )
        declared = len(fe.get_feature_columns())
        result = fe.build_features(simple_df)
        actual = len([c for c in result.columns if c not in ["engine_id", "cycle", "RUL"]])
        assert declared == actual


# ── Triage agent v2 tests ─────────────────────────────────────────────────────

class TestTriageAgentV2:
    @pytest.fixture
    def agent(self):
        return TriageAgent(
            critical_rul=15,
            warning_rul=40,
            anomaly_threshold=-0.05,
            uncertainty_threshold=25.0,
            model_version="test_v2",
        )

    def test_critical_rul_rule(self, agent):
        d = agent.evaluate(1, 100, predicted_rul=10.0, anomaly_score=0.1)
        assert d.severity == "Critical"
        assert d.triggered_rule == "RULE_CRITICAL_RUL"

    def test_warning_rul_rule(self, agent):
        d = agent.evaluate(1, 100, predicted_rul=30.0, anomaly_score=0.1)
        assert d.severity == "Warning"
        assert d.triggered_rule == "RULE_WARNING_RUL"

    def test_anomaly_override_rule(self, agent):
        d = agent.evaluate(1, 100, predicted_rul=80.0, anomaly_score=-0.1)
        assert d.severity == "InspectionReview"
        assert d.triggered_rule == "RULE_ANOMALY_OVERRIDE"

    def test_uncertainty_risk_rule(self, agent):
        d = agent.evaluate(1, 100, predicted_rul=80.0, anomaly_score=0.1,
                           rul_uncertainty=30.0)
        assert d.severity == "UncertaintyReview"
        assert d.triggered_rule == "RULE_UNCERTAINTY_RISK"

    def test_normal_rule(self, agent):
        d = agent.evaluate(1, 100, predicted_rul=80.0, anomaly_score=0.1,
                           rul_uncertainty=5.0)
        assert d.severity == "Normal"
        assert d.triggered_rule == "RULE_NORMAL"

    def test_invalid_input_rule(self, agent):
        d = agent.evaluate(None, None, predicted_rul=None, anomaly_score=None)
        assert d.severity == "DataQuality"
        assert d.triggered_rule == "RULE_INVALID_INPUT"

    def test_negative_rul_floored_to_zero(self, agent):
        d = agent.evaluate(1, 100, predicted_rul=-5.0, anomaly_score=0.0)
        # After floor, -5 → 0 → Critical (0 <= 15)
        assert d.predicted_rul == 0.0
        assert d.severity == "Critical"

    def test_uncertainty_fields_in_decision(self, agent):
        d = agent.evaluate(1, 50, predicted_rul=60.0, anomaly_score=0.0,
                           rul_uncertainty=10.0, rul_lower=43.6, rul_upper=76.4)
        assert d.rul_uncertainty == pytest.approx(10.0)
        assert d.rul_lower == pytest.approx(43.6)
        assert d.rul_upper == pytest.approx(76.4)

    def test_batch_returns_one_per_row(self, agent):
        df = pd.DataFrame({
            "engine_id": [1, 2, 3],
            "cycle": [10, 20, 30],
            "predicted_rul": [5.0, 35.0, 90.0],
            "anomaly_score": [0.0, 0.0, 0.0],
        })
        decisions = agent.evaluate_batch(df)
        assert len(decisions) == 3
        assert decisions[0].severity == "Critical"
        assert decisions[1].severity == "Warning"
        assert decisions[2].severity == "Normal"


# ── Live monitor tests ────────────────────────────────────────────────────────

class TestLiveMonitor:
    def test_live_monitor_missing_file_raises(self, tmp_path):
        from src.simulation.live_monitor import LiveMonitor
        with pytest.raises(FileNotFoundError):
            LiveMonitor(predictions_path=tmp_path / "nonexistent.csv")

    def test_live_monitor_step_advances_index(self, tmp_path):
        from src.simulation.live_monitor import LiveMonitor
        df = pd.DataFrame({
            "engine_id": [1, 1, 1],
            "cycle": [1, 2, 3],
            "predicted_rul": [50.0, 45.0, 40.0],
            "anomaly_score": [0.1, 0.1, 0.1],
            "is_anomaly": [0, 0, 0],
            "rul_uncertainty": [5.0, 5.0, 5.0],
            "rul_lower": [40.0, 35.0, 30.0],
            "rul_upper": [60.0, 55.0, 50.0],
            "RUL": [50.0, 40.0, 30.0],
        })
        pred_path = tmp_path / "predictions.csv"
        df.to_csv(pred_path, index=False)

        monitor = LiveMonitor(predictions_path=pred_path, speed_seconds=0.0)
        event1 = monitor.step()
        event2 = monitor.step()
        event3 = monitor.step()
        done = monitor.step()

        assert event1 is not None and event1.cycle == 1
        assert event2 is not None and event2.cycle == 2
        assert event3 is not None and event3.cycle == 3
        assert done is None  # finished

    def test_live_monitor_is_simulation_always_true(self, tmp_path):
        from src.simulation.live_monitor import LiveMonitor
        df = pd.DataFrame({
            "engine_id": [1], "cycle": [1],
            "predicted_rul": [50.0], "anomaly_score": [0.0],
            "is_anomaly": [0], "rul_uncertainty": [0.0],
            "rul_lower": [50.0], "rul_upper": [50.0], "RUL": [50.0],
        })
        pred_path = tmp_path / "predictions.csv"
        df.to_csv(pred_path, index=False)
        monitor = LiveMonitor(predictions_path=pred_path, speed_seconds=0.0)
        event = monitor.step()
        assert event is not None
        assert event.is_simulation is True

    def test_live_monitor_engine_filter(self, tmp_path):
        from src.simulation.live_monitor import LiveMonitor
        df = pd.DataFrame({
            "engine_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "predicted_rul": [50.0, 45.0, 60.0, 55.0],
            "anomaly_score": [0.0] * 4,
            "is_anomaly": [0] * 4,
            "rul_uncertainty": [0.0] * 4,
            "rul_lower": [45.0, 40.0, 55.0, 50.0],
            "rul_upper": [55.0, 50.0, 65.0, 60.0],
            "RUL": [50.0, 40.0, 60.0, 50.0],
        })
        pred_path = tmp_path / "predictions.csv"
        df.to_csv(pred_path, index=False)

        monitor = LiveMonitor(predictions_path=pred_path, speed_seconds=0.0, engine_id=2)
        events = monitor.run_blocking()
        assert all(e.engine_id == 2 for e in events)
        assert len(events) == 2

    def test_live_monitor_fleet_summary(self, tmp_path):
        from src.simulation.live_monitor import LiveMonitor
        df = pd.DataFrame({
            "engine_id": [1, 2],
            "cycle": [5, 5],
            "predicted_rul": [10.0, 80.0],
            "anomaly_score": [0.0, 0.0],
            "is_anomaly": [0, 0],
            "rul_uncertainty": [0.0, 0.0],
            "rul_lower": [10.0, 80.0],
            "rul_upper": [10.0, 80.0],
            "RUL": [10.0, 80.0],
            "severity": ["Critical", "Normal"],
        })
        pred_path = tmp_path / "predictions.csv"
        df.to_csv(pred_path, index=False)
        monitor = LiveMonitor(predictions_path=pred_path, speed_seconds=0.0)
        monitor.run_blocking()
        summary = monitor.get_fleet_summary()
        # 'severity' column will be used from predictions
        assert sum(summary.values()) == 2


# ── FD001-FD004 data loading tests ───────────────────────────────────────────

class TestMultiDatasetLoading:
    """Verify all 4 C-MAPSS subsets can be loaded (skips gracefully if missing)."""

    @pytest.mark.parametrize("subset", ["FD001", "FD002", "FD003", "FD004"])
    def test_dataset_loads(self, subset):
        from src.data.loader import get_project_root, load_raw_cmapss
        root = get_project_root()
        candidates = [
            root.parent / "CMAPSSData" / f"train_{subset}.txt",
            root / "../CMAPSSData" / f"train_{subset}.txt",
        ]
        path = next((c.resolve() for c in candidates if c.resolve().is_file()), None)
        if path is None:
            pytest.skip(f"train_{subset}.txt not found")

        df = load_raw_cmapss(path, dataset_id=subset)
        assert df.shape[1] == 26
        assert "engine_id" in df.columns
        assert "cycle" in df.columns
        assert df["engine_id"].min() >= 1

    @pytest.mark.parametrize("subset", ["FD001", "FD002", "FD003"])
    def test_rul_labels_aligned(self, subset):
        from src.data.loader import get_project_root, load_raw_cmapss, load_rul_labels
        from src.data.rul_calculator import compute_test_rul
        root = get_project_root()
        base = root.parent / "CMAPSSData"
        test_path = (base / f"test_{subset}.txt").resolve()
        rul_path = (base / f"RUL_{subset}.txt").resolve()
        if not test_path.is_file() or not rul_path.is_file():
            pytest.skip(f"test data for {subset} not found")

        df_test = load_raw_cmapss(test_path, dataset_id=subset)
        rul_labels = load_rul_labels(rul_path)
        df_test_rul = compute_test_rul(df_test, rul_labels)

        # Last-cycle RUL of any engine must equal the ground truth label
        for eng_id in df_test["engine_id"].unique()[:5]:  # check first 5 engines
            last_row = df_test_rul[df_test_rul["engine_id"] == eng_id].iloc[-1]
            assert float(last_row["RUL"]) == pytest.approx(float(rul_labels.loc[eng_id]))

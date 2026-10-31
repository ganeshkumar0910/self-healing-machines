# Feature Analysis Report — Self-Healing Machines v2
# Dataset: NASA C-MAPSS FD001 | Features: 218 (v2) vs 115 (v1)

---

## Feature Engineering Changes: v1 → v2

### v1 Feature Set (115 features)
| Category | Count | Description |
|----------|-------|-------------|
| Base sensors | 14 | Raw sensor readings (non-constant sensors) |
| Operating settings | 3 | op_setting_1, op_setting_2, op_setting_3 |
| Rolling mean | 42 | 14 sensors × 3 windows (5, 10, 20) |
| Rolling std | 42 | 14 sensors × 3 windows |
| Trend (slope) | 14 | Lagged difference per sensor |
| **Total** | **115** | |

### v2 Feature Set (218 features)
| Category | Count | Description |
|----------|-------|-------------|
| Base sensors | 14 | Raw sensor readings |
| Operating settings | 3 | op_setting_1-3 |
| Rolling mean | 42 | 14 sensors × 3 windows (5, 10, 20) |
| Rolling std | 42 | 14 sensors × 3 windows |
| **Rolling min** *(NEW)* | 42 | 14 sensors × 3 windows |
| **Rolling max** *(NEW)* | 42 | 14 sensors × 3 windows |
| **EMA (alpha=0.1)** *(NEW)* | 14 | Exponential moving avg per sensor |
| Trend (slope) | 14 | Lagged difference per sensor |
| **Sensor differences** *(NEW)* | 4 | Physically meaningful pair differences |
| **cycle_pct** *(NEW)* | 1 | cycle / max_cycle (degradation stage) |
| **Total** | **218** | **+103 new features (+89.6%)** |

---

## New Feature Descriptions

### Rolling Min / Max
**Why:** Rolling min and max capture the range of sensor behavior in a window.
A suddenly high rolling max indicates a spike; a dropping rolling min indicates degradation.
These features are particularly useful for detecting anomalies that rolling mean misses.

**Implementation:** Engine-partitioned (no cross-engine leakage).
`sensor_X_min_W` = min(sensor_X) over last W cycles for same engine.

---

### Exponential Moving Average (EMA, α=0.1)
**Why:** EMA gives more weight to recent sensor readings while smoothing noise.
Unlike rolling mean (equal weights), EMA adapts faster to trend changes.

**Formula:** `EMA_t = α × sensor_t + (1 − α) × EMA_{t-1}`

With α=0.1: slowly adapts to reflect persistent degradation trends.
Implemented engine-wise via `pandas.ewm(alpha=0.1, adjust=False)`.

---

### Sensor Difference Pairs
**Why:** Domain knowledge suggests certain sensor pairs measure related physical quantities.
Their difference highlights imbalance or efficiency loss between engine sections.

| Feature | Sensors | Physical Meaning |
|---------|---------|-----------------|
| `sensor_2_minus_sensor_3` | HPC outlet temp vs LPT outlet temp | Thermal gradient across turbine |
| `sensor_4_minus_sensor_11` | HPC outlet pressure vs HPC effl temp | Pressure-temperature efficiency ratio |
| `sensor_9_minus_sensor_14` | Physical pressure pair | Pressure ratio across stage |
| `sensor_7_minus_sensor_12` | Fan static pressure vs corrected fan speed | Fan efficiency indicator |

When a difference deviates from its normal range, it signals degradation in that engine section.

---

### Cycle Percentage (cycle_pct)
**Why:** Absolute cycle number is dataset-dependent (engine A might run 200 cycles,
engine B might run 350). `cycle_pct = cycle / max_observed_cycle` normalizes to [0, 1].

- cycle_pct = 0.1 → early life, low degradation expected
- cycle_pct = 0.9 → late life, high degradation zone

This feature helps the model calibrate predictions based on degradation stage rather
than absolute cycle number.

---

## Cross-Engine Leakage Verification

All v2 features are computed within engine-partitioned groups:
```python
grouped = df_sorted.groupby("engine_id", sort=False)
min_s = grouped[sensor].rolling(window=w, min_periods=1).min()
ema_s = grouped[sensor].transform(lambda x: x.ewm(alpha=0.1, adjust=False).mean())
```

**Test result:** `test_rolling_minmax_no_cross_engine` — PASSED.
Engine 2 cycle 1 rolling features unaffected by engine 1 data.

---

## Impact on Model Performance

| Feature Set | Test EP-MAE (best model) | Feature Count | Notes |
|-------------|--------------------------|---------------|-------|
| v1 (115 features) | 43.66 cycles | 115 | Row-level eval (misleading) |
| v2 (218 features) | 35.91 cycles | 218 | Endpoint eval + capped test RUL |

Honest improvement: **-17.8% test endpoint MAE**.

Note: The improvement includes both feature engineering and the corrected evaluation protocol.
Isolating feature impact alone requires a controlled ablation study (future work).

---

## Recommended Future Features

| Feature | Rationale |
|---------|-----------|
| FFT components (top-k frequencies) | Detect cyclic sensor patterns |
| Inter-engine fleet percentile rank | Compare engine health vs peer fleet |
| Rolling coefficient of variation | Sensor noise / instability indicator |
| Cross-sensor correlation anomaly | Detect correlated vs uncorrelated degradation |

*These would require more compute but should improve accuracy toward literature-level LSTM results.*

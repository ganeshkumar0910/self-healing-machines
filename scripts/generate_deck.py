"""
Generate the Round 1 Design Deck using python-pptx.
Run: py -3 scripts/generate_deck.py
Output: reports/round1/Round1_Design_Deck.pptx
"""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import datetime


# ─── Color Palette ───────────────────────────────────────────────────────────
BG_DARK     = RGBColor(0x0F, 0x17, 0x2A)   # Dark navy background
ACCENT_BLUE = RGBColor(0x3B, 0x82, 0xF6)   # Bright blue
ACCENT_CYAN = RGBColor(0x06, 0xB6, 0xD4)   # Cyan
ACCENT_GREEN= RGBColor(0x22, 0xC5, 0x5E)   # Green
ACCENT_AMBER= RGBColor(0xF5, 0x9E, 0x0B)   # Amber
ACCENT_RED  = RGBColor(0xEF, 0x44, 0x44)   # Red
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY  = RGBColor(0xCB, 0xD5, 0xE1)
MID_GRAY    = RGBColor(0x64, 0x74, 0x8B)


def set_bg(slide, color: RGBColor):
    """Set slide background color."""
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, text: str, left, top, width, height,
                 font_size: int = 18, bold: bool = False,
                 color: RGBColor = WHITE, align=PP_ALIGN.LEFT,
                 wrap: bool = True):
    """Add a styled text box to the slide."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    return txBox


def add_title_section(slide, title: str, subtitle: str = ""):
    """Add a standard section header."""
    add_text_box(slide, title,
                 Inches(0.5), Inches(0.3), Inches(9), Inches(0.7),
                 font_size=32, bold=True, color=ACCENT_BLUE)
    if subtitle:
        add_text_box(slide, subtitle,
                     Inches(0.5), Inches(1.0), Inches(9), Inches(0.4),
                     font_size=14, color=LIGHT_GRAY)


def add_divider(slide, top_inches: float):
    """Add a horizontal accent line."""
    line = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(0.5), Inches(top_inches),
        Inches(9), Inches(0.03)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT_BLUE
    line.line.fill.background()


def add_bullet_list(slide, items: list, left, top, width, height,
                    font_size: int = 14, color: RGBColor = WHITE,
                    bullet_color: RGBColor = ACCENT_CYAN):
    """Add a bulleted list."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(4)
        run = p.add_run()
        run.text = f"▸  {item}"
        run.font.size = Pt(font_size)
        run.font.color.rgb = color


def add_colored_box(slide, text: str, left, top, width, height,
                    bg_color: RGBColor = ACCENT_BLUE,
                    text_color: RGBColor = WHITE,
                    font_size: int = 13, bold: bool = False):
    """Add a colored rectangle with centered text."""
    shape = slide.shapes.add_shape(1, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = text_color


def create_presentation() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]  # Blank layout

    # ── SLIDE 1: Title ────────────────────────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide, BG_DARK)

    # Gradient accent band
    band = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(10), Inches(0.08))
    band.fill.solid()
    band.fill.fore_color.rgb = ACCENT_BLUE
    band.line.fill.background()

    add_text_box(slide, "🔧 Self-Healing Machines",
                 Inches(0.8), Inches(1.5), Inches(8.5), Inches(1.2),
                 font_size=38, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    add_text_box(slide, "ML-Powered Predictive Maintenance &\nAutomated Work Order Management",
                 Inches(0.8), Inches(2.8), Inches(8.5), Inches(0.9),
                 font_size=20, color=ACCENT_CYAN, align=PP_ALIGN.CENTER)

    add_divider(slide, 3.9)

    add_text_box(slide, "Round 1 — Design Presentation",
                 Inches(0.8), Inches(4.1), Inches(8.5), Inches(0.5),
                 font_size=15, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)

    add_text_box(slide, f"NASA CMAPSS FD001 Dataset  ·  Python · scikit-learn · Streamlit · MLflow · SQLite",
                 Inches(0.8), Inches(4.7), Inches(8.5), Inches(0.4),
                 font_size=12, color=MID_GRAY, align=PP_ALIGN.CENTER)

    add_text_box(slide, f"Prepared: {datetime.date.today().strftime('%B %Y')}",
                 Inches(0.8), Inches(5.3), Inches(8.5), Inches(0.4),
                 font_size=11, color=MID_GRAY, align=PP_ALIGN.CENTER)

    # ── SLIDE 2: Problem Statement ────────────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide, BG_DARK)
    add_divider(slide, 0.0)

    add_title_section(slide, "Problem Statement", "Why predictive maintenance matters")

    # Problem boxes
    problems = [
        ("❌ Reactive Maintenance", "Respond AFTER failure — costly downtime, safety risk"),
        ("📅 Scheduled Maintenance", "Replace healthy parts — wasteful, over-conservative"),
        ("🎯 Predictive Maintenance", "Act BEFORE failure — right time, right engine"),
    ]
    colors = [ACCENT_RED, ACCENT_AMBER, ACCENT_GREEN]
    for i, (title, desc) in enumerate(problems):
        left = Inches(0.5 + i * 3.1)
        add_colored_box(slide, title, left, Inches(1.8), Inches(2.9), Inches(0.7),
                        bg_color=colors[i], font_size=13, bold=True)
        add_text_box(slide, desc, left, Inches(2.6), Inches(2.9), Inches(0.9),
                     font_size=12, color=LIGHT_GRAY)

    add_text_box(slide, "Our Solution:",
                 Inches(0.5), Inches(3.8), Inches(9), Inches(0.4),
                 font_size=16, bold=True, color=ACCENT_CYAN)

    solution_points = [
        "Predict Remaining Useful Life (RUL) in cycles using sensor data",
        "Detect anomalies early — before RUL drops to critical levels",
        "Transparent rule-based triage: no black-box AI, every decision explained",
        "Automated work orders with priority, audit trail, and status tracking",
    ]
    add_bullet_list(slide, solution_points, Inches(0.5), Inches(4.3), Inches(9), Inches(2.5))

    # ── SLIDE 3: Objectives and Solution ──────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide, BG_DARK)
    add_divider(slide, 0.0)
    add_title_section(slide, "Objectives & Proposed Solution", "Six core capabilities")

    objectives = [
        ("1", "RUL Prediction", "Predict remaining cycles before failure using ML regression"),
        ("2", "Anomaly Detection", "Unsupervised Isolation Forest on sensor readings"),
        ("3", "Rule-Based Triage", "Transparent Python rules: Critical / Warning / Normal / Inspect"),
        ("4", "Work Order Management", "Auto-generate maintenance tasks with priority and audit"),
        ("5", "Experiment Tracking", "MLflow: log every training run, version models, promote safely"),
        ("6", "Live Dashboard", "Streamlit: 5 pages, real data, no hardcoded values"),
    ]

    for i, (num, title, desc) in enumerate(objectives):
        row = i // 2
        col = i % 2
        left = Inches(0.5 + col * 4.8)
        top  = Inches(1.8 + row * 1.6)
        add_colored_box(slide, f"{num}. {title}", left, top, Inches(4.5), Inches(0.55),
                        bg_color=ACCENT_BLUE, font_size=13, bold=True)
        add_text_box(slide, desc, left, Inches(top.inches + 0.6), Inches(4.5), Inches(0.8),
                     font_size=12, color=LIGHT_GRAY)

    # ── SLIDE 4: Dataset ──────────────────────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide, BG_DARK)
    add_divider(slide, 0.0)
    add_title_section(slide, "Dataset: NASA CMAPSS FD001", "Commercial Modular Aero-Propulsion System Simulation")

    facts = [
        ("Training Engines", "100"),
        ("Test Engines", "100"),
        ("Operating Conditions", "1 (Sea Level)"),
        ("Fault Mode", "HPC Degradation"),
        ("Input Columns", "26 (ID, cycle, 3 settings, 21 sensors)"),
        ("Official RUL Labels", "RUL_FD001.txt (100 values)"),
    ]
    for i, (label, val) in enumerate(facts):
        row = i // 2
        col = i % 2
        left = Inches(0.5 + col * 4.8)
        top = Inches(1.8 + row * 1.3)
        add_colored_box(slide, label, left, top, Inches(2.2), Inches(0.5),
                        bg_color=RGBColor(0x1E, 0x29, 0x3B), font_size=11, text_color=LIGHT_GRAY)
        add_colored_box(slide, val, Inches(left.inches + 2.3), top, Inches(2.1), Inches(0.5),
                        bg_color=ACCENT_BLUE, font_size=11, bold=True)

    add_text_box(slide, "Why FD001?",
                 Inches(0.5), Inches(5.3), Inches(9), Inches(0.35),
                 font_size=14, bold=True, color=ACCENT_CYAN)
    add_bullet_list(slide, [
        "Single operating condition — cleaner signal for baseline model",
        "Single fault mode — interpretable, reproducible degradation pattern",
        "Community benchmark — published baselines enable comparison",
        "Pre-provided — no external download required",
    ], Inches(0.5), Inches(5.7), Inches(9), Inches(1.5), font_size=12)

    # ── SLIDE 5: Data Engineering Pipeline ───────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide, BG_DARK)
    add_divider(slide, 0.0)
    add_title_section(slide, "Data Engineering Pipeline", "From raw text files to clean feature matrices")

    stages = [
        ("Raw Data\nLoader", ACCENT_BLUE),
        ("Schema\nValidator", ACCENT_CYAN),
        ("RUL\nCalculator", ACCENT_GREEN),
        ("Feature\nEngineer", ACCENT_AMBER),
        ("Preprocessor\n(fit train only)", ACCENT_BLUE),
        ("Processed\nCSV / joblib", ACCENT_CYAN),
    ]
    for i, (label, color) in enumerate(stages):
        left = Inches(0.3 + i * 1.55)
        add_colored_box(slide, label, left, Inches(1.8), Inches(1.35), Inches(1.1),
                        bg_color=color, font_size=11, bold=True)
        if i < len(stages) - 1:
            add_text_box(slide, "→", Inches(left.inches + 1.4), Inches(2.1), Inches(0.25), Inches(0.5),
                         font_size=20, color=WHITE, align=PP_ALIGN.CENTER)

    key_points = [
        "RUL labels: train = max_cycle - current_cycle  |  test = final_cycle - c + official_final_RUL",
        "Rolling features computed PER ENGINE in chronological order — zero cross-engine leakage",
        "14 of 21 sensors selected (non-constant variance in FD001): s2,s3,s4,s7,s8,s9,s11-s15,s17,s20,s21",
        "Rolling windows: 5, 10, 20 cycles → mean, std, trend per sensor",
        "Scaler and Imputer fitted on TRAINING engines only — applied identically to test/dashboard",
        "Optional RUL cap at 125 cycles — prevents early-lifecycle dominance during training",
    ]
    add_bullet_list(slide, key_points, Inches(0.5), Inches(3.2), Inches(9.2), Inches(3.8),
                    font_size=12)

    # ── SLIDE 6: ML Approach ──────────────────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide, BG_DARK)
    add_divider(slide, 0.0)
    add_title_section(slide, "ML Approach & Evaluation Metrics", "Three regression models + unsupervised anomaly detection")

    models_data = [
        ("Linear Regression", "Baseline", ACCENT_BLUE, "Interpretable minimum\nperformance benchmark"),
        ("Random Forest", "Main Model", ACCENT_GREEN, "Captures non-linear\ndegradation patterns"),
        ("Gradient Boosting", "Comparison", ACCENT_AMBER, "Often top tabular\nregressor — compared"),
        ("Isolation Forest", "Anomaly Det.", ACCENT_CYAN, "Unsupervised; no\nanomaly labels needed"),
    ]
    for i, (model, role, color, desc) in enumerate(models_data):
        left = Inches(0.4 + i * 2.35)
        add_colored_box(slide, role, left, Inches(1.8), Inches(2.1), Inches(0.4),
                        bg_color=color, font_size=10, bold=True)
        add_colored_box(slide, model, left, Inches(2.25), Inches(2.1), Inches(0.6),
                        bg_color=RGBColor(0x1E, 0x29, 0x3B), font_size=12, bold=True, text_color=WHITE)
        add_text_box(slide, desc, left, Inches(2.9), Inches(2.1), Inches(0.8),
                     font_size=11, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)

    add_text_box(slide, "Evaluation Metrics  (computed from actual predictions — never hardcoded)",
                 Inches(0.5), Inches(4.0), Inches(9.2), Inches(0.4),
                 font_size=13, bold=True, color=ACCENT_CYAN)

    metrics_points = [
        "MAE (Mean Absolute Error) — average prediction error in cycles",
        "RMSE (Root Mean Squared Error) — penalises large errors more than MAE",
        "NASA Asymmetric Score: d=pred−true; s=exp(−d/13)−1 if d<0, else exp(d/10)−1; Score=Σs",
        "Asymmetry: overestimating RUL (missing urgency) penalised MORE than underestimating",
        "Train/validation split: engine-wise (80/20 engine IDs) — zero row-level leakage",
    ]
    add_bullet_list(slide, metrics_points, Inches(0.5), Inches(4.5), Inches(9.2), Inches(2.8), font_size=12)

    # ── SLIDE 7: System Architecture ──────────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide, BG_DARK)
    add_divider(slide, 0.0)
    add_title_section(slide, "System Architecture", "Five decoupled layers — modular and testable")

    layers = [
        ("DATA LAYER", "Loader · Validator · RUL Calc · Feature Engineer · Preprocessor", ACCENT_BLUE),
        ("ML LAYER", "LinearRUL · RandomForest · GradientBoosting · IsolationForest · PredictionService", ACCENT_CYAN),
        ("AGENT LAYER", "TriageAgent (rule-based) · WorkOrderService · SQLite DB", ACCENT_GREEN),
        ("MLOPS LAYER", "ExperimentTracker (MLflow) · RetrainingPipeline · Model Registry", ACCENT_AMBER),
        ("DASHBOARD LAYER", "Streamlit App · 5 Pages · Plotly Charts · Real Data Only", ACCENT_BLUE),
    ]
    for i, (layer, components, color) in enumerate(layers):
        top = Inches(1.8 + i * 1.0)
        add_colored_box(slide, layer, Inches(0.5), top, Inches(2.2), Inches(0.6),
                        bg_color=color, font_size=11, bold=True)
        add_text_box(slide, components, Inches(2.9), top, Inches(6.9), Inches(0.6),
                     font_size=11, color=LIGHT_GRAY)

    add_text_box(slide, "▸  All layers are independently testable  ▸  No external APIs or LLMs  ▸  CPU-only",
                 Inches(0.5), Inches(7.0), Inches(9.2), Inches(0.4),
                 font_size=11, color=MID_GRAY)

    # ── SLIDE 8: Triage & Work Orders ─────────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide, BG_DARK)
    add_divider(slide, 0.0)
    add_title_section(slide, "Rule-Based Triage & Work Order Lifecycle", "Transparent decisions — no LLM — fully auditable")

    rules = [
        ("RULE_INVALID_INPUT", "Invalid / NaN input", "DataQuality", "Reject — data quality issue", ACCENT_RED),
        ("RULE_CRITICAL_RUL", "Predicted RUL ≤ 15 cycles", "Critical", "Emergency maintenance NOW", ACCENT_RED),
        ("RULE_WARNING_RUL", "15 < RUL ≤ 40 cycles", "Warning", "Schedule maintenance soon", ACCENT_AMBER),
        ("RULE_ANOMALY_OVERRIDE", "RUL > 40, anomaly detected", "InspectionReview", "Targeted sensor inspection", RGBColor(0x8B, 0x5C, 0xF6)),
        ("RULE_NORMAL", "RUL > 40, no anomaly", "Normal", "Continue standard monitoring", ACCENT_GREEN),
    ]

    headers = ["Rule ID", "Condition", "Severity", "Action"]
    col_widths = [1.8, 2.5, 1.5, 3.1]
    col_starts = [0.4, 2.3, 4.9, 6.5]

    for j, header in enumerate(headers):
        add_colored_box(slide, header,
                        Inches(col_starts[j]), Inches(1.7),
                        Inches(col_widths[j]), Inches(0.4),
                        bg_color=RGBColor(0x1E, 0x40, 0x6E), font_size=11, bold=True)

    for i, (rule, cond, sev, action, color) in enumerate(rules):
        top = Inches(2.15 + i * 0.75)
        row_bg = RGBColor(0x12, 0x1F, 0x38) if i % 2 == 0 else RGBColor(0x0F, 0x17, 0x2A)
        vals = [rule, cond, sev, action]
        for j, (val, cw, cs) in enumerate(zip(vals, col_widths, col_starts)):
            fg = color if j == 2 else WHITE
            bg = row_bg
            add_colored_box(slide, val, Inches(cs), top, Inches(cw), Inches(0.65),
                            bg_color=bg, font_size=10, bold=(j == 2), text_color=fg)

    add_text_box(slide, "Work Order: auto-created for Critical / Warning / InspectionReview  |  SQLite  |  Status: Open → InProgress → Closed  |  Duplicate prevention via DB constraint",
                 Inches(0.5), Inches(7.0), Inches(9.2), Inches(0.4),
                 font_size=10, color=LIGHT_GRAY)

    # ── SLIDE 9: MLOps & Retraining ───────────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide, BG_DARK)
    add_divider(slide, 0.0)
    add_title_section(slide, "MLOps, Retraining & Monitoring", "Safe model promotion — previous model always preserved")

    steps = [
        "1. Load approved training data (FD001)",
        "2. Validate data integrity",
        "3. Rebuild features from scratch",
        "4. Engine-wise train/val split (same seed)",
        "5. Train CANDIDATE model (RF)",
        "6. Evaluate candidate on val split → MAE_candidate",
        "7. Load ACTIVE model → evaluate on SAME val set → MAE_active",
        "8. If MAE_candidate < MAE_active × (1 − min_improvement): PROMOTE",
        "   Backup active → artifacts/models/backup_{timestamp}/",
        "   Write artifacts/models/active_model.txt",
        "   Else: REJECT — active model unchanged",
        "9. Log everything to MLflow (params, metrics, artifacts, decision)",
    ]
    add_bullet_list(slide, steps, Inches(0.5), Inches(1.7), Inches(5.5), Inches(5.2),
                    font_size=12)

    add_text_box(slide, "MLflow Logs Per Run:",
                 Inches(6.1), Inches(1.7), Inches(3.7), Inches(0.4),
                 font_size=13, bold=True, color=ACCENT_CYAN)
    mlflow_items = [
        "Experiment name", "Dataset hash", "Model type",
        "Hyperparameters", "Feature config", "RUL cap",
        "Val MAE / RMSE / NASA", "Test metrics", "Promotion status",
    ]
    add_bullet_list(slide, mlflow_items, Inches(6.1), Inches(2.2), Inches(3.7), Inches(4.0), font_size=12)

    # ── SLIDE 10: Expected Outcomes & Next Steps ───────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide, BG_DARK)
    add_divider(slide, 0.0)
    add_title_section(slide, "Expected Outcomes, Limitations & Next Steps", "Round 1 → Round 2 roadmap")

    outcomes_title = "Round 2 Deliverables"
    add_text_box(slide, outcomes_title, Inches(0.5), Inches(1.6), Inches(4.5), Inches(0.4),
                 font_size=14, bold=True, color=ACCENT_GREEN)
    outcomes = [
        "Trained RUL model (RF) with actual MAE/RMSE/NASA metrics",
        "Anomaly detector fitted on training data",
        "Triage agent with unit-tested rule branches",
        "SQLite work orders (Open/InProgress/Closed)",
        "MLflow experiment history with model versions",
        "5-page Streamlit dashboard (real data, no placeholders)",
        "All pytest tests passing (or failures documented)",
        "Round 1 design document + this presentation",
    ]
    add_bullet_list(slide, outcomes, Inches(0.5), Inches(2.1), Inches(4.5), Inches(3.5), font_size=12)

    add_text_box(slide, "Limitations", Inches(5.3), Inches(1.6), Inches(4.4), Inches(0.4),
                 font_size=14, bold=True, color=ACCENT_AMBER)
    limitations = [
        "Simulation data — not real operational sensors",
        "Triage thresholds (15/40 cycles) are demo values",
        "No ground-truth anomaly labels → no precision/recall",
        "Static dataset — no live sensor ingestion",
        "Single-user SQLite (no concurrent writes)",
    ]
    add_bullet_list(slide, limitations, Inches(5.3), Inches(2.1), Inches(4.4), Inches(2.0), font_size=12)

    add_text_box(slide, "Next Steps", Inches(5.3), Inches(4.2), Inches(4.4), Inches(0.4),
                 font_size=14, bold=True, color=ACCENT_CYAN)
    next_steps = [
        "Connect to live sensor stream (MQTT / Kafka)",
        "Expand to FD002–FD004 subsets",
        "Add LSTM model for sequence-aware RUL",
        "Deploy as FastAPI microservice + Docker",
        "Integrate with CMMS (SAP PM / IBM Maximo)",
    ]
    add_bullet_list(slide, next_steps, Inches(5.3), Inches(4.7), Inches(4.4), Inches(2.2), font_size=12)

    add_colored_box(slide, "🔧 Self-Healing Machines  —  Questions Welcome",
                    Inches(0.5), Inches(6.9), Inches(9.2), Inches(0.5),
                    bg_color=ACCENT_BLUE, font_size=14, bold=True)

    return prs


def main():
    output_dir = Path(__file__).parent.parent / "reports" / "round1"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "Round1_Design_Deck.pptx"

    print("Generating Round 1 Design Deck...")
    prs = create_presentation()
    prs.save(str(output_path))
    print(f"[OK] Saved: {output_path}")
    print(f"   Slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()

"""Self-Healing Machines: ML-Powered Predictive Maintenance Dashboard.

Main Streamlit Application Entrypoint.
"""

from datetime import datetime
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, Optional, Tuple
import yaml
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Dashboard page renderers
from dashboard.pages.data_explorer import render_data_explorer
from dashboard.pages.engine_health import render_engine_health
from dashboard.pages.fleet_overview import render_fleet_overview
from dashboard.pages.model_monitoring import render_model_monitoring
from dashboard.pages.work_orders import render_work_orders

# Backend system imports with graceful fallbacks
try:
    from src.data.loader import get_project_root, load_config
except ImportError:
    def get_project_root() -> Path:
        return PROJECT_ROOT

    def load_config(path: Optional[str] = None) -> Dict[str, Any]:
        cfg_path = Path(path) if path else PROJECT_ROOT / "config" / "settings.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

try:
    from src.database.db_manager import DatabaseManager
except ImportError:
    class DatabaseManager:  # type: ignore
        """Fallback database manager."""
        def __init__(self, db_path: Optional[str] = None):
            self.db_path = db_path or str(PROJECT_ROOT / "data" / "maintenance.db")

try:
    from src.maintenance.work_order_service import WorkOrderService
except ImportError:
    class WorkOrderService:  # type: ignore
        """Fallback work order service."""
        def __init__(self, db_manager: Any = None):
            self.db_manager = db_manager


# Streamlit Application Configuration
st.set_page_config(
    page_title="Self-Healing Machines",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_app_config() -> Dict[str, Any]:
    """Load and cache project configuration at startup."""
    cfg_file = PROJECT_ROOT / "config" / "settings.yaml"
    try:
        cfg = load_config(str(cfg_file))
        if cfg:
            return cfg
    except Exception:
        pass

    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass

    return {
        "paths": {
            "database": "data/maintenance.db",
            "artifacts_dir": "artifacts/models",
            "processed_data_dir": "data/processed",
        },
        "dataset": {"default_subset": "FD001", "subsets": ["FD001", "FD002", "FD003", "FD004"]},
        "triage": {"critical_rul_threshold": 15, "warning_rul_threshold": 40, "anomaly_score_threshold": -0.05},
    }


@st.cache_resource
def init_services(config_dict: Dict[str, Any]) -> Tuple[Any, Any]:
    """Initialize and cache DatabaseManager and WorkOrderService."""
    db_rel = config_dict.get("paths", {}).get("database", "data/maintenance.db")
    db_path = str(PROJECT_ROOT / db_rel)

    db_mgr = None
    try:
        db_mgr = DatabaseManager(db_path=db_path)
    except Exception:
        try:
            db_mgr = DatabaseManager()
        except Exception:
            db_mgr = None

    wo_svc = None
    try:
        wo_svc = WorkOrderService(db_manager=db_mgr)
    except Exception:
        try:
            wo_svc = WorkOrderService()
        except Exception:
            wo_svc = None

    return db_mgr, wo_svc


def inject_custom_css() -> None:
    """Inject premium dark-theme stylesheet."""
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Overall Dark Background */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    /* Sidebar Gradient & Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090d16 0%, #0f172a 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
        padding-top: 1rem;
    }

    /* Custom Radio Navigation Styling */
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 6px;
    }

    section[data-testid="stSidebar"] .stRadio label {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 8px 12px;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-bottom: 4px;
        color: #cbd5e1 !important;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(56, 189, 248, 0.12);
        border-color: rgba(56, 189, 248, 0.3);
        color: #f8fafc !important;
    }

    /* Button Polish */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        border: 1px solid #38bdf8;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.35);
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        border-color: #7dd3fc;
        transform: translateY(-1px);
    }

    /* Card Shadow Polish */
    div[data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }

    /* Expander Container */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #090d16;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def main() -> None:
    """Main dashboard application orchestration."""
    inject_custom_css()

    # Load configuration and initialize service dependencies
    config = get_app_config()
    db_manager, wo_service = init_services(config)

    # Sidebar Header
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0 16px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="font-size: 28px;">🔧</div>
                <div>
                    <h2 style="font-size: 19px; font-weight: 800; color: #f8fafc; margin: 0; line-height: 1.2;">
                        Self-Healing
                    </h2>
                    <span style="font-size: 11px; font-weight: 600; color: #38bdf8; letter-spacing: 0.8px; text-transform: uppercase;">
                        Predictive Maintenance
                    </span>
                </div>
            </div>
            <p style="color: #94a3b8; font-size: 12px; margin: 8px 0 0 0; line-height: 1.4;">
                NASA C-MAPSS Turbofan Health Telemetry & Autonomous Triage
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    # Navigation Menu
    pages = [
        "🏭 Fleet Overview",
        "🔍 Engine Health",
        "📋 Work Orders",
        "📊 Model Monitoring",
        "🗃 Data Explorer",
    ]

    selected_page = st.sidebar.radio(
        "Navigation",
        options=pages,
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # System Status Indicators in Sidebar
    active_txt = PROJECT_ROOT / "artifacts" / "models" / "active_model.txt"
    active_version = "v1.0.0"
    if active_txt.exists():
        try:
            active_version = active_txt.read_text().strip()
        except Exception:
            pass

    st.sidebar.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 10px 12px; margin-bottom: 16px;">
            <div style="font-size: 11px; font-weight: 600; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px;">
                System Status
            </div>
            <div style="display: flex; align-items: center; gap: 8px; font-size: 12px; color: #cbd5e1; margin-bottom: 4px;">
                <span style="height: 8px; width: 8px; background-color: #22c55e; border-radius: 50%; display: inline-block;"></span>
                <span>Champion: <b>{active_version}</b></span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px; font-size: 12px; color: #cbd5e1;">
                <span style="height: 8px; width: 8px; background-color: #38bdf8; border-radius: 50%; display: inline-block;"></span>
                <span>Database: <b>SQLite Active</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Footer
    st.sidebar.markdown(
        """
        <div style="font-size: 11px; color: #64748b; line-height: 1.5;">
            <b>Self-Healing Machines Dashboard</b><br>
            Version: <code>v1.0.0</code><br>
            Dataset: NASA C-MAPSS Turbofan<br>
            MLOps: MLflow & Automated Triage
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Page Routing with Error Boundary
    try:
        if selected_page == "🏭 Fleet Overview":
            render_fleet_overview(config, db_manager)
        elif selected_page == "🔍 Engine Health":
            render_engine_health(config)
        elif selected_page == "📋 Work Orders":
            render_work_orders(config, db_manager, wo_service)
        elif selected_page == "📊 Model Monitoring":
            render_model_monitoring(config)
        elif selected_page == "🗃 Data Explorer":
            render_data_explorer(config)
    except Exception as exc:
        st.error(f"❌ **An unexpected exception occurred while rendering `{selected_page}`:** {exc}")
        with st.expander("🔍 Technical Stack Trace for Debugging", expanded=False):
            st.code(traceback.format_exc(), language="python")


if __name__ == "__main__":
    main()

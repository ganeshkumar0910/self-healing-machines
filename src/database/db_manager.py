"""SQLite Database Manager for persisting work orders and lifecycle audit history."""

from contextlib import contextmanager
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database schema, connections, and transactional queries."""

    def __init__(self, db_path: Union[str, Path]) -> None:
        """Initialize DatabaseManager with target SQLite database file.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a configured SQLite connection.

        Returns:
            sqlite3.Connection: Connection with Row factory enabled.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @contextmanager
    def connection_scope(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding an active database connection and closing it on exit."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create database tables and constraints if they do not already exist."""
        logger.info("Initializing database schema at: %s", self.db_path)

        create_work_orders = """
        CREATE TABLE IF NOT EXISTS work_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wo_id TEXT UNIQUE NOT NULL,
            engine_id INTEGER NOT NULL,
            cycle INTEGER NOT NULL,
            predicted_rul REAL,
            anomaly_score REAL,
            severity TEXT NOT NULL,
            reason TEXT,
            recommended_action TEXT,
            model_version TEXT,
            status TEXT DEFAULT 'Open',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            triggered_rule TEXT,
            explanation TEXT,
            UNIQUE(engine_id, cycle, severity)
        );
        """

        create_wo_history = """
        CREATE TABLE IF NOT EXISTS wo_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wo_id TEXT NOT NULL,
            old_status TEXT,
            new_status TEXT NOT NULL,
            changed_at TEXT NOT NULL,
            notes TEXT,
            FOREIGN KEY (wo_id) REFERENCES work_orders (wo_id)
        );
        """

        create_indices = """
        CREATE INDEX IF NOT EXISTS idx_wo_engine ON work_orders(engine_id);
        CREATE INDEX IF NOT EXISTS idx_wo_status ON work_orders(status);
        CREATE INDEX IF NOT EXISTS idx_wo_severity ON work_orders(severity);
        CREATE INDEX IF NOT EXISTS idx_history_wo_id ON wo_history(wo_id);
        """

        with self.connection_scope() as conn:
            cursor = conn.cursor()
            cursor.execute(create_work_orders)
            cursor.execute(create_wo_history)
            cursor.executescript(create_indices)
            conn.commit()

        logger.info("Database schema successfully initialized.")

    def execute_query(
        self, sql: str, params: Union[Tuple[Any, ...], List[Any]] = ()
    ) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return rows as a list of dictionaries.

        Args:
            sql: SQL SELECT query string.
            params: Tuple or list of parameters to bind.

        Returns:
            list[dict]: List of row dictionaries.
        """
        with self.connection_scope() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def execute_write(
        self, sql: str, params: Union[Tuple[Any, ...], List[Any]] = ()
    ) -> int:
        """Execute an INSERT, UPDATE, or DELETE statement with commit.

        Args:
            sql: SQL statement string.
            params: Parameters to bind.

        Returns:
            int: Number of affected rows (cursor.rowcount).
        """
        with self.connection_scope() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            return cursor.rowcount

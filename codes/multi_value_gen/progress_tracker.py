"""
SQLite-based checkpoint system for tracking pipeline progress.
Enables resumption after interruption.
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ProgressTracker:
    def __init__(self, db_path, resume=False):
        self.db_path = Path(db_path)
        if not resume and self.db_path.exists():
            self.db_path.unlink()
            logger.info("Cleared previous progress database")

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

        if resume and self.db_path.exists():
            self._log_resume_status()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS completed_combinations (
                dataset TEXT,
                dialect TEXT,
                num_records INTEGER,
                num_errors INTEGER,
                completed_at TEXT,
                PRIMARY KEY (dataset, dialect)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS completed_rows (
                dataset TEXT,
                dialect TEXT,
                row_idx INTEGER,
                PRIMARY KEY (dataset, dialect, row_idx)
            )
        """)
        self.conn.commit()

    def _log_resume_status(self):
        cur = self.conn.execute("SELECT COUNT(*) FROM completed_combinations")
        done = cur.fetchone()[0]
        cur2 = self.conn.execute(
            "SELECT dataset, dialect, COUNT(*) FROM completed_rows GROUP BY dataset, dialect"
        )
        partial = cur2.fetchall()
        logger.info(f"Resuming: {done} combinations complete, {len(partial)} in progress")

    def is_combination_done(self, dataset, dialect):
        cur = self.conn.execute(
            "SELECT 1 FROM completed_combinations WHERE dataset=? AND dialect=?",
            (dataset, dialect),
        )
        return cur.fetchone() is not None

    def mark_combination_done(self, dataset, dialect, num_records=0, num_errors=0):
        self.conn.execute(
            "INSERT OR REPLACE INTO completed_combinations VALUES (?, ?, ?, ?, ?)",
            (dataset, dialect, num_records, num_errors,
             datetime.now(timezone.utc).isoformat()),
        )
        # Clean up row-level tracking for this combination
        self.conn.execute(
            "DELETE FROM completed_rows WHERE dataset=? AND dialect=?",
            (dataset, dialect),
        )
        self.conn.commit()

    def is_row_done(self, dataset, dialect, row_idx):
        cur = self.conn.execute(
            "SELECT 1 FROM completed_rows WHERE dataset=? AND dialect=? AND row_idx=?",
            (dataset, dialect, row_idx),
        )
        return cur.fetchone() is not None

    def mark_row_done(self, dataset, dialect, row_idx):
        self.conn.execute(
            "INSERT OR IGNORE INTO completed_rows VALUES (?, ?, ?)",
            (dataset, dialect, row_idx),
        )
        # Commit periodically for performance
        if row_idx % 100 == 0:
            self.conn.commit()

    def flush(self):
        self.conn.commit()

    def get_partial_rows(self, dataset, dialect):
        """Get set of already-completed row indices for a partially-done combination."""
        cur = self.conn.execute(
            "SELECT row_idx FROM completed_rows WHERE dataset=? AND dialect=?",
            (dataset, dialect),
        )
        return {r[0] for r in cur.fetchall()}

    def print_summary(self):
        cur = self.conn.execute("SELECT COUNT(*) FROM completed_combinations")
        done = cur.fetchone()[0]
        cur2 = self.conn.execute(
            "SELECT SUM(num_records), SUM(num_errors) FROM completed_combinations"
        )
        row = cur2.fetchone()
        total_records = row[0] or 0
        total_errors = row[1] or 0
        logger.info(f"Completed {done} dataset-dialect combinations")
        logger.info(f"Total records transformed: {total_records}")
        logger.info(f"Total transform errors: {total_errors}")

    def close(self):
        self.conn.close()

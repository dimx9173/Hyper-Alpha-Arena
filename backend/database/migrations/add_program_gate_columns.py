"""
Add performance gate columns to account_program_bindings.

- gate_status: 'pending' | 'pass' | 'fail' — result of latest performance
  gate (annual/exploration validator) check for the bound program.
- last_gate_check_at: when the gate was last (re)evaluated. The execution
  service re-runs the validator when this is older than 24h; a 'fail'
  result deactivates the binding.

Idempotent: skips columns that already exist. PostgreSQL only (local dev
runs Postgres; sqlite migration targets are legacy).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from database.connection import SessionLocal


def migrate():
    db = SessionLocal()
    try:
        cols = {
            row[0]
            for row in db.execute(
                text("SELECT column_name FROM information_schema.columns "
                     "WHERE table_name='account_program_bindings'")
            ).fetchall()
        }
        if "gate_status" not in cols:
            db.execute(text(
                "ALTER TABLE account_program_bindings "
                "ADD COLUMN gate_status VARCHAR(20) NOT NULL DEFAULT 'pending'"
            ))
            print("added gate_status")
        else:
            print("gate_status already exists")
        if "last_gate_check_at" not in cols:
            db.execute(text(
                "ALTER TABLE account_program_bindings "
                "ADD COLUMN last_gate_check_at TIMESTAMP NULL"
            ))
            print("added last_gate_check_at")
        else:
            print("last_gate_check_at already exists")
        db.commit()
        print("done")
    finally:
        db.close()


if __name__ == "__main__":
    migrate()

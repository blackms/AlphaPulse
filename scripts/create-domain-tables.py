#!/usr/bin/env python3
"""
Create domain tables using SQLAlchemy models.

This script creates all application tables (users, trades, positions, etc.)
that are defined in the SQLAlchemy models but not yet created by migrations.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import create_engine, inspect
from alpha_pulse.config.database import get_db_config
from alpha_pulse.models import Base  # This imports all models

def create_tables():
    """Create all tables defined in SQLAlchemy models."""
    # Get database engine
    db_config = get_db_config()
    engine = db_config.get_engine()

    # Get inspector to check existing tables
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    print("📊 Existing tables:")
    for table in sorted(existing_tables):
        print(f"  - {table}")
    print()

    # Get tables from metadata
    metadata_tables = list(Base.metadata.tables.keys())
    new_tables = [t for t in metadata_tables if t not in existing_tables]

    if new_tables:
        print(f"🔨 Creating {len(new_tables)} new tables:")
        for table in sorted(new_tables):
            print(f"  - {table}")
        print()

        # Create all tables
        Base.metadata.create_all(engine)
        print("✅ Tables created successfully!")
    else:
        print("✅ All tables already exist!")

    # Verify
    inspector = inspect(engine)
    final_tables = inspector.get_table_names()
    print(f"\n📊 Total tables: {len(final_tables)}")

if __name__ == "__main__":
    try:
        create_tables()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

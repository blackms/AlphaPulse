"""
Migration script to add audit logs table.

This script creates the audit_logs table with proper indexes and constraints
for efficient audit logging and querying.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from datetime import datetime


def upgrade():
    """Create audit_logs table."""
    
    # Create custom types if using PostgreSQL
    # For other databases, adjust accordingly
    
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('event_type', sa.String(length=100), nullable=False, index=True),
        sa.Column('severity', sa.String(length=20), nullable=False, index=True),
        
        # Context fields
        sa.Column('user_id', sa.String(length=100), index=True),
        sa.Column('session_id', sa.String(length=100), index=True),
        sa.Column('ip_address', sa.String(length=50)),
        sa.Column('user_agent', sa.String(length=500)),
        sa.Column('request_id', sa.String(length=100), index=True),
        sa.Column('correlation_id', sa.String(length=100), index=True),
        
        # Event data - using JSON for flexibility
        # In production with encryption enabled, this would use EncryptedJSON
        sa.Column('event_data', sa.JSON()),
        
        # For encrypted version (when encryption is enabled):
        # sa.Column('event_data_encrypted', sa.Text()),  # Stores encrypted JSON
        
        # Performance metrics
        sa.Column('duration_ms', sa.Float()),
        
        # Success/failure tracking
        sa.Column('success', sa.Boolean(), default=True),
        sa.Column('error_message', sa.Text()),
        
        # For encrypted error messages (when encryption is enabled):
        # sa.Column('error_message_encrypted', sa.Text()),
        
        # Compliance fields
        sa.Column('data_classification', sa.String(length=50)),
        sa.Column('regulatory_flags', sa.JSON()),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for common queries
    op.create_index(
        'idx_audit_timestamp_type',
        'audit_logs',
        ['timestamp', 'event_type']
    )
    
    op.create_index(
        'idx_audit_user_timestamp',
        'audit_logs',
        ['user_id', 'timestamp']
    )
    
    op.create_index(
        'idx_audit_correlation',
        'audit_logs',
        ['correlation_id']
    )
    
    # Create partial indexes for performance
    # Failed operations index
    op.create_index(
        'idx_audit_failures',
        'audit_logs',
        ['timestamp', 'event_type'],
        postgresql_where=sa.text('success = false')
    )
    
    # High severity events index
    op.create_index(
        'idx_audit_high_severity',
        'audit_logs',
        ['timestamp', 'severity'],
        postgresql_where=sa.text("severity IN ('error', 'critical')")
    )
    
    # Add comments for documentation
    op.execute("COMMENT ON TABLE audit_logs IS 'Comprehensive audit log for all system events'")
    op.execute("COMMENT ON COLUMN audit_logs.event_type IS 'Type of event (auth.login, trade.executed, etc.)'")
    op.execute("COMMENT ON COLUMN audit_logs.severity IS 'Event severity (debug, info, warning, error, critical)'")
    op.execute("COMMENT ON COLUMN audit_logs.data_classification IS 'Data classification level (public, internal, confidential, restricted)'")
    op.execute("COMMENT ON COLUMN audit_logs.regulatory_flags IS 'Compliance flags (GDPR, SOX, PCI, etc.)'")


def downgrade():
    """Drop audit_logs table."""
    
    # Drop indexes
    op.drop_index('idx_audit_high_severity', table_name='audit_logs')
    op.drop_index('idx_audit_failures', table_name='audit_logs')
    op.drop_index('idx_audit_correlation', table_name='audit_logs')
    op.drop_index('idx_audit_user_timestamp', table_name='audit_logs')
    op.drop_index('idx_audit_timestamp_type', table_name='audit_logs')
    
    # Drop table
    op.drop_table('audit_logs')


# Manual migration script for existing systems
if __name__ == "__main__":
    """
    Manual migration script to add audit logs table.
    
    Usage:
        python -m alpha_pulse.migrations.add_audit_logs
    """
    import sys
    from sqlalchemy import create_engine, text
    from alpha_pulse.config.database import get_db_url
    
    print("=== AlphaPulse Audit Logs Table Migration ===")
    print()
    
    # Get database URL
    db_url = get_db_url()
    engine = create_engine(db_url)
    
    # Check if table already exists
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'audit_logs')"
        ))
        exists = result.scalar()
        
        if exists:
            print("❌ Table 'audit_logs' already exists!")
            sys.exit(1)
            
    print("Creating audit_logs table...")
    
    try:
        # Create table
        with engine.connect() as conn:
            # Create table
            conn.execute(text("""
                CREATE TABLE audit_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    event_type VARCHAR(100) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    user_id VARCHAR(100),
                    session_id VARCHAR(100),
                    ip_address VARCHAR(50),
                    user_agent VARCHAR(500),
                    request_id VARCHAR(100),
                    correlation_id VARCHAR(100),
                    event_data JSONB,
                    duration_ms FLOAT,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    data_classification VARCHAR(50),
                    regulatory_flags JSONB
                )
            """))
            
            # Create indexes
            conn.execute(text(
                "CREATE INDEX idx_audit_timestamp ON audit_logs(timestamp)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_audit_event_type ON audit_logs(event_type)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_audit_severity ON audit_logs(severity)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_audit_user_id ON audit_logs(user_id)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_audit_timestamp_type ON audit_logs(timestamp, event_type)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_audit_user_timestamp ON audit_logs(user_id, timestamp)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_audit_correlation ON audit_logs(correlation_id)"
            ))
            
            # Partial indexes
            conn.execute(text(
                "CREATE INDEX idx_audit_failures ON audit_logs(timestamp, event_type) "
                "WHERE success = false"
            ))
            conn.execute(text(
                "CREATE INDEX idx_audit_high_severity ON audit_logs(timestamp, severity) "
                "WHERE severity IN ('error', 'critical')"
            ))
            
            conn.commit()
            
        print("✅ Successfully created audit_logs table")
        print()
        print("Next steps:")
        print("1. Update your application configuration")
        print("2. Restart the application to enable audit logging")
        print("3. Verify logging with: SELECT COUNT(*) FROM audit_logs;")
        
    except Exception as e:
        print(f"❌ Error creating table: {e}")
        sys.exit(1)
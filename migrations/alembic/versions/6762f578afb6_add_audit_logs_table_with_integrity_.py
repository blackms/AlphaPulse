"""Add audit logs table with integrity protection

Revision ID: 6762f578afb6
Revises: abd8932b8e55
Create Date: 2025-07-05 14:49:09.501000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6762f578afb6'
down_revision: Union[str, None] = 'abd8932b8e55'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create audit_logs table
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
        sa.Column('event_data', sa.JSON()),
        
        # Performance metrics
        sa.Column('duration_ms', sa.Float()),
        
        # Success/failure tracking
        sa.Column('success', sa.Boolean(), default=True),
        sa.Column('error_message', sa.Text()),
        
        # Compliance fields
        sa.Column('data_classification', sa.String(length=50)),
        sa.Column('regulatory_flags', sa.JSON()),
        
        # Tamper protection
        sa.Column('integrity_hash', sa.String(length=64)),
        
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
    
    # Create partial indexes for performance (PostgreSQL specific)
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


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_audit_high_severity', table_name='audit_logs')
    op.drop_index('idx_audit_failures', table_name='audit_logs')
    op.drop_index('idx_audit_correlation', table_name='audit_logs')
    op.drop_index('idx_audit_user_timestamp', table_name='audit_logs')
    op.drop_index('idx_audit_timestamp_type', table_name='audit_logs')
    
    # Drop table
    op.drop_table('audit_logs')

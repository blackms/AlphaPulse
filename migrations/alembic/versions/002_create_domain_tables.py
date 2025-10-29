"""Create domain tables

Revision ID: 002
Revises: 001
Create Date: 2025-10-29 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade():
    # Create users table (without tenant_id - will be added in migration 006)
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(50), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('is_verified', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
        sa.UniqueConstraint('username', name=op.f('uq_users_username')),
        sa.UniqueConstraint('email', name=op.f('uq_users_email'))
    )
    op.create_index(op.f('idx_user_username'), 'users', ['username'])
    op.create_index(op.f('idx_user_active'), 'users', ['is_active'])

    # Create roles table
    op.create_table('roles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_roles')),
        sa.UniqueConstraint('name', name=op.f('uq_roles_name'))
    )

    # Create user_roles association table
    op.create_table('user_roles',
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('role_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_user_roles_user_id_users')),
        sa.ForeignKeyConstraint(['role_id'], ['roles.id'], name=op.f('fk_user_roles_role_id_roles')),
        sa.PrimaryKeyConstraint('user_id', 'role_id', name=op.f('pk_user_roles'))
    )

    # Create trading_accounts table (without tenant_id - will be added in migration 007)
    op.create_table('trading_accounts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('account_id', sa.String(50), nullable=False),
        sa.Column('account_number', sa.String(255), nullable=False),
        sa.Column('exchange_name', sa.String(50), nullable=False),
        sa.Column('account_type', sa.String(20), nullable=False),
        sa.Column('balance', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('available_balance', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_trading_accounts')),
        sa.UniqueConstraint('account_id', name=op.f('uq_trading_accounts_account_id'))
    )
    op.create_index(op.f('idx_account_exchange'), 'trading_accounts', ['exchange_name', 'account_type'])

    # Create positions table (without tenant_id - will be added in migration 007)
    op.create_table('positions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('position_id', sa.String(50), nullable=False),
        sa.Column('account_id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),
        sa.Column('size', sa.Float(), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=False),
        sa.Column('current_price', sa.Float(), nullable=True),
        sa.Column('unrealized_pnl', sa.Float(), nullable=True),
        sa.Column('realized_pnl', sa.Float(), server_default='0.0', nullable=True),
        sa.Column('stop_loss', sa.Float(), nullable=True),
        sa.Column('take_profit', sa.Float(), nullable=True),
        sa.Column('opened_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('closed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(20), server_default='open', nullable=False),
        sa.ForeignKeyConstraint(['account_id'], ['trading_accounts.id'], name=op.f('fk_positions_account_id_trading_accounts')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_positions')),
        sa.UniqueConstraint('position_id', name=op.f('uq_positions_position_id'))
    )
    op.create_index(op.f('idx_position_account'), 'positions', ['account_id'])
    op.create_index(op.f('idx_position_symbol'), 'positions', ['symbol'])
    op.create_index(op.f('idx_position_status'), 'positions', ['status'])

    # Create trades table (without tenant_id - will be added in migration 007)
    op.create_table('trades',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('trade_id', sa.String(50), nullable=False),
        sa.Column('account_id', sa.Integer(), nullable=False),
        sa.Column('position_id', sa.Integer(), nullable=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),
        sa.Column('trade_type', sa.String(20), nullable=False),
        sa.Column('quantity', sa.Float(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('fee', sa.Float(), server_default='0.0', nullable=True),
        sa.Column('gross_value', sa.Float(), nullable=False),
        sa.Column('net_value', sa.Float(), nullable=False),
        sa.Column('executed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['account_id'], ['trading_accounts.id'], name=op.f('fk_trades_account_id_trading_accounts')),
        sa.ForeignKeyConstraint(['position_id'], ['positions.id'], name=op.f('fk_trades_position_id_positions')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_trades')),
        sa.UniqueConstraint('trade_id', name=op.f('uq_trades_trade_id'))
    )
    op.create_index(op.f('idx_trade_account'), 'trades', ['account_id'])
    op.create_index(op.f('idx_trade_symbol'), 'trades', ['symbol'])
    op.create_index(op.f('idx_trade_executed'), 'trades', ['executed_at'])

    # Create risk_metrics table (without tenant_id - will be added in migration 007)
    op.create_table('risk_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('account_id', sa.Integer(), nullable=False),
        sa.Column('portfolio_value', sa.Float(), nullable=False),
        sa.Column('total_exposure', sa.Float(), nullable=False),
        sa.Column('leverage', sa.Float(), server_default='1.0', nullable=False),
        sa.Column('var_95', sa.Float(), nullable=True),
        sa.Column('var_99', sa.Float(), nullable=True),
        sa.Column('cvar_95', sa.Float(), nullable=True),
        sa.Column('sharpe_ratio', sa.Float(), nullable=True),
        sa.Column('sortino_ratio', sa.Float(), nullable=True),
        sa.Column('max_drawdown', sa.Float(), nullable=True),
        sa.Column('calculated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['account_id'], ['trading_accounts.id'], name=op.f('fk_risk_metrics_account_id_trading_accounts')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_risk_metrics'))
    )
    op.create_index(op.f('idx_risk_account'), 'risk_metrics', ['account_id'])
    op.create_index(op.f('idx_risk_calculated'), 'risk_metrics', ['calculated_at'])

    # Create portfolio_snapshots table (without tenant_id - will be added in migration 007)
    op.create_table('portfolio_snapshots',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('account_id', sa.Integer(), nullable=False),
        sa.Column('total_value', sa.Float(), nullable=False),
        sa.Column('cash_balance', sa.Float(), nullable=False),
        sa.Column('daily_return', sa.Float(), nullable=True),
        sa.Column('cumulative_return', sa.Float(), nullable=True),
        sa.Column('snapshot_time', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['account_id'], ['trading_accounts.id'], name=op.f('fk_portfolio_snapshots_account_id_trading_accounts')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_portfolio_snapshots')),
        sa.UniqueConstraint('account_id', 'snapshot_time', name=op.f('uq_account_snapshot'))
    )
    op.create_index(op.f('idx_snapshot_account'), 'portfolio_snapshots', ['account_id'])
    op.create_index(op.f('idx_snapshot_time'), 'portfolio_snapshots', ['snapshot_time'])

    # Create agent_signals table (without tenant_id - will be added in migration 007)
    op.create_table('agent_signals',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('agent_type', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('signal', sa.String(10), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('reasoning', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_agent_signals'))
    )
    op.create_index(op.f('idx_agent_signal_type'), 'agent_signals', ['agent_type'])
    op.create_index(op.f('idx_agent_signal_symbol'), 'agent_signals', ['symbol'])
    op.create_index(op.f('idx_agent_signal_created'), 'agent_signals', ['created_at'])

    # Create audit_logs table (without tenant_id - will be added in migration 007)
    op.create_table('audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('event_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(255), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_audit_logs_user_id_users')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_audit_logs'))
    )
    op.create_index(op.f('idx_audit_user'), 'audit_logs', ['user_id'])
    op.create_index(op.f('idx_audit_event_type'), 'audit_logs', ['event_type'])
    op.create_index(op.f('idx_audit_created'), 'audit_logs', ['created_at'])


def downgrade():
    op.drop_index(op.f('idx_audit_created'), table_name='audit_logs')
    op.drop_index(op.f('idx_audit_event_type'), table_name='audit_logs')
    op.drop_index(op.f('idx_audit_user'), table_name='audit_logs')
    op.drop_table('audit_logs')

    op.drop_index(op.f('idx_agent_signal_created'), table_name='agent_signals')
    op.drop_index(op.f('idx_agent_signal_symbol'), table_name='agent_signals')
    op.drop_index(op.f('idx_agent_signal_type'), table_name='agent_signals')
    op.drop_table('agent_signals')

    op.drop_index(op.f('idx_snapshot_time'), table_name='portfolio_snapshots')
    op.drop_index(op.f('idx_snapshot_account'), table_name='portfolio_snapshots')
    op.drop_table('portfolio_snapshots')

    op.drop_index(op.f('idx_risk_calculated'), table_name='risk_metrics')
    op.drop_index(op.f('idx_risk_account'), table_name='risk_metrics')
    op.drop_table('risk_metrics')

    op.drop_index(op.f('idx_trade_executed'), table_name='trades')
    op.drop_index(op.f('idx_trade_symbol'), table_name='trades')
    op.drop_index(op.f('idx_trade_account'), table_name='trades')
    op.drop_table('trades')

    op.drop_index(op.f('idx_position_status'), table_name='positions')
    op.drop_index(op.f('idx_position_symbol'), table_name='positions')
    op.drop_index(op.f('idx_position_account'), table_name='positions')
    op.drop_table('positions')

    op.drop_index(op.f('idx_account_exchange'), table_name='trading_accounts')
    op.drop_table('trading_accounts')

    op.drop_table('user_roles')
    op.drop_table('roles')

    op.drop_index(op.f('idx_user_active'), table_name='users')
    op.drop_index(op.f('idx_user_username'), table_name='users')
    op.drop_table('users')

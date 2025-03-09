"""Create initial tables

Revision ID: 001
Revises: 
Create Date: 2025-03-09 12:26:39.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create sync_status table
    op.create_table('sync_status',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('exchange_id', sa.String(), nullable=False),
        sa.Column('data_type', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('last_sync', sa.DateTime(timezone=True), nullable=True),
        sa.Column('next_sync', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_sync_status')),
        sa.UniqueConstraint('exchange_id', 'data_type', name=op.f('uq_sync_status_exchange_data_type'))
    )
    
    # Create exchange_balances table
    op.create_table('exchange_balances',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('exchange_id', sa.String(), nullable=False),
        sa.Column('currency', sa.String(), nullable=False),
        sa.Column('available', sa.Float(), nullable=True),
        sa.Column('locked', sa.Float(), nullable=True),
        sa.Column('total', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_exchange_balances')),
        sa.UniqueConstraint('exchange_id', 'currency', name=op.f('uq_exchange_balances_exchange_currency'))
    )
    
    # Create exchange_positions table
    op.create_table('exchange_positions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('exchange_id', sa.String(), nullable=False),
        sa.Column('symbol', sa.String(), nullable=False),
        sa.Column('quantity', sa.Float(), nullable=True),
        sa.Column('entry_price', sa.Float(), nullable=True),
        sa.Column('current_price', sa.Float(), nullable=True),
        sa.Column('unrealized_pnl', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_exchange_positions')),
        sa.UniqueConstraint('exchange_id', 'symbol', name=op.f('uq_exchange_positions_exchange_symbol'))
    )
    
    # Create exchange_orders table
    op.create_table('exchange_orders',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('exchange_id', sa.String(), nullable=False),
        sa.Column('order_id', sa.String(), nullable=False),
        sa.Column('symbol', sa.String(), nullable=False),
        sa.Column('order_type', sa.String(), nullable=True),
        sa.Column('side', sa.String(), nullable=True),
        sa.Column('price', sa.Float(), nullable=True),
        sa.Column('amount', sa.Float(), nullable=True),
        sa.Column('filled', sa.Float(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_exchange_orders')),
        sa.UniqueConstraint('exchange_id', 'order_id', name=op.f('uq_exchange_orders_exchange_order_id'))
    )
    
    # Create exchange_prices table
    op.create_table('exchange_prices',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('exchange_id', sa.String(), nullable=False),
        sa.Column('base_currency', sa.String(), nullable=False),
        sa.Column('quote_currency', sa.String(), nullable=False),
        sa.Column('price', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_exchange_prices')),
        sa.UniqueConstraint('exchange_id', 'base_currency', 'quote_currency', name=op.f('uq_exchange_prices_exchange_currencies'))
    )


def downgrade():
    op.drop_table('exchange_prices')
    op.drop_table('exchange_orders')
    op.drop_table('exchange_positions')
    op.drop_table('exchange_balances')
    op.drop_table('sync_status')
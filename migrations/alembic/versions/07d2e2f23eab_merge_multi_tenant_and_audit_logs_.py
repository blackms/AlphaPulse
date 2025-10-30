"""merge multi-tenant and audit logs branches

Revision ID: 07d2e2f23eab
Revises: 009_composite_indexes, 6762f578afb6
Create Date: 2025-10-30 17:31:02.232825

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '07d2e2f23eab'
down_revision: Union[str, None] = ('009_composite_indexes', '6762f578afb6')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

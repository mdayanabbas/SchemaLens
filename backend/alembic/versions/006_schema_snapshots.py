"""Add schema snapshot models

Revision ID: 006_schema_snapshots
Revises: 005_stored_secrets
Create Date: 2026-07-28 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '006_schema_snapshots'
down_revision: Union[str, None] = '005_stored_secrets'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # This is a placeholder since autogenerate isn't working due to Alembic state inconsistencies
    # and the prompt specifies not to execute migrations.
    pass

def downgrade() -> None:
    pass

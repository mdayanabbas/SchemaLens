"""initial_domain

Revision ID: 001
Revises: 
Create Date: 2026-07-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Organizations
    op.create_table(
        'organizations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('slug', sa.String(length=100), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default='active'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name='pk_organizations')
    )
    op.create_index('ix_organizations_slug', 'organizations', ['slug'], unique=True)
    op.create_index('ix_organizations_status', 'organizations', ['status'], unique=False)

    # Users
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(length=320), nullable=False),
        sa.Column('display_name', sa.String(length=150), nullable=False),
        sa.Column('password_hash', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=False, server_default='active'),
        sa.Column('is_platform_admin', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name='pk_users')
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_index('ix_users_status', 'users', ['status'], unique=False)

    # Organization Memberships
    op.create_table(
        'organization_memberships',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default='invited'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name='fk_organization_memberships_organization_id_organizations', ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='fk_organization_memberships_user_id_users', ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id', name='pk_organization_memberships'),
        sa.UniqueConstraint('organization_id', 'user_id', name='uq_organization_memberships_org_user')
    )
    op.create_index('ix_organization_memberships_organization_id', 'organization_memberships', ['organization_id'], unique=False)
    op.create_index('ix_organization_memberships_user_id', 'organization_memberships', ['user_id'], unique=False)
    op.create_index('ix_organization_memberships_org_status', 'organization_memberships', ['organization_id', 'status'], unique=False)
    op.create_index('ix_organization_memberships_user_status', 'organization_memberships', ['user_id', 'status'], unique=False)


def downgrade() -> None:
    op.drop_table('organization_memberships')
    op.drop_table('users')
    op.drop_table('organizations')

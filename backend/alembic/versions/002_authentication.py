"""authentication

Revision ID: 002
Revises: 001
Create Date: 2026-07-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Refresh Tokens
    op.create_table(
        'refresh_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('family_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('token_hash', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default='active'),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rotated_to_token_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_ip_hash', sa.String(), nullable=True),
        sa.Column('created_user_agent_hash', sa.String(), nullable=True),
        sa.Column('last_used_ip_hash', sa.String(), nullable=True),
        sa.Column('last_used_user_agent_hash', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['rotated_to_token_id'], ['refresh_tokens.id'], name='fk_refresh_tokens_rotated_to_token_id_refresh_tokens', ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='fk_refresh_tokens_user_id_users', ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id', name='pk_refresh_tokens')
    )
    op.create_index('ix_refresh_tokens_family_id', 'refresh_tokens', ['family_id'], unique=False)
    op.create_index('ix_refresh_tokens_status', 'refresh_tokens', ['status'], unique=False)
    op.create_index('ix_refresh_tokens_token_hash', 'refresh_tokens', ['token_hash'], unique=True)
    op.create_index('ix_refresh_tokens_user_id', 'refresh_tokens', ['user_id'], unique=False)
    op.create_index('ix_refresh_tokens_expires_at', 'refresh_tokens', ['expires_at'], unique=False)

    # Authentication Events
    op.create_table(
        'authentication_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('event_type', sa.String(), nullable=False),
        sa.Column('outcome', sa.String(), nullable=False),
        sa.Column('email_hash', sa.String(), nullable=True),
        sa.Column('request_id', sa.String(), nullable=True),
        sa.Column('ip_hash', sa.String(), nullable=True),
        sa.Column('user_agent_hash', sa.String(), nullable=True),
        sa.Column('safe_metadata_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='fk_authentication_events_user_id_users', ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name='pk_authentication_events')
    )
    op.create_index('ix_authentication_events_event_type', 'authentication_events', ['event_type'], unique=False)
    op.create_index('ix_authentication_events_user_id', 'authentication_events', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_table('authentication_events')
    op.drop_table('refresh_tokens')

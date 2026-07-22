"""Audit trail

Revision ID: 003_audit_trail
Revises: 002_authentication
Create Date: 2026-07-22 07:15:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003_audit_trail'
down_revision = '002_authentication'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # 1. Create the new audit_events table
    op.create_table(
        'audit_events',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('organization_id', sa.UUID(), nullable=True),
        sa.Column('actor_user_id', sa.UUID(), nullable=True),
        sa.Column('actor_type', sa.String(length=50), nullable=False),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('outcome', sa.String(length=50), nullable=False),
        sa.Column('resource_type', sa.String(length=50), nullable=False),
        sa.Column('resource_id', sa.UUID(), nullable=True),
        sa.Column('request_id', sa.String(length=255), nullable=True),
        sa.Column('workflow_id', sa.UUID(), nullable=True),
        sa.Column('event_version', sa.Integer(), nullable=False),
        sa.Column('source', sa.String(length=50), nullable=False),
        sa.Column('ip_hash', sa.String(length=64), nullable=True),
        sa.Column('user_agent_hash', sa.String(length=64), nullable=True),
        sa.Column('metadata_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('occurred_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        
        sa.ForeignKeyConstraint(['actor_user_id'], ['users.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # 2. Create compound indexes required by the query patterns
    op.create_index('ix_audit_events_org_occurred', 'audit_events', ['organization_id', sa.text('occurred_at DESC')], unique=False)
    op.create_index('ix_audit_events_actor_occurred', 'audit_events', ['actor_user_id', sa.text('occurred_at DESC')], unique=False)
    op.create_index('ix_audit_events_action_occurred', 'audit_events', ['action', sa.text('occurred_at DESC')], unique=False)
    op.create_index('ix_audit_events_resource_occurred', 'audit_events', ['resource_type', 'resource_id', sa.text('occurred_at DESC')], unique=False)
    op.create_index('ix_audit_events_workflow_occurred', 'audit_events', ['workflow_id', sa.text('occurred_at DESC')], unique=False)
    op.create_index('ix_audit_events_occurred_at', 'audit_events', [sa.text('occurred_at DESC')], unique=False)

def downgrade() -> None:
    op.drop_index('ix_audit_events_occurred_at', table_name='audit_events')
    op.drop_index('ix_audit_events_workflow_occurred', table_name='audit_events')
    op.drop_index('ix_audit_events_resource_occurred', table_name='audit_events')
    op.drop_index('ix_audit_events_action_occurred', table_name='audit_events')
    op.drop_index('ix_audit_events_actor_occurred', table_name='audit_events')
    op.drop_index('ix_audit_events_org_occurred', table_name='audit_events')
    op.drop_table('audit_events')

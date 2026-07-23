"""database connections and policies

Revision ID: 004_connections
Revises: 003_audit_trail
Create Date: 2026-07-23 23:06:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004_connections'
down_revision = '003_audit_trail'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # 1. database_connections
    op.create_table(
        'database_connections',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('organization_id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=150), nullable=False),
        sa.Column('description', sa.String(length=1000), nullable=True),
        sa.Column('environment', sa.String(length=50), nullable=False),
        sa.Column('dialect', sa.String(length=50), nullable=False),
        sa.Column('host', sa.String(length=255), nullable=False),
        sa.Column('port', sa.Integer(), nullable=False),
        sa.Column('database_name', sa.String(length=100), nullable=False),
        sa.Column('default_catalog', sa.String(length=100), nullable=True),
        sa.Column('ssl_mode', sa.String(length=50), nullable=False),
        sa.Column('secret_provider', sa.String(length=50), nullable=False),
        sa.Column('secret_reference', sa.String(length=500), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('last_tested_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_test_status', sa.String(length=50), nullable=False),
        sa.Column('last_test_error_code', sa.String(length=100), nullable=True),
        sa.Column('created_by_user_id', sa.UUID(), nullable=False),
        sa.Column('updated_by_user_id', sa.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        
        sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['updated_by_user_id'], ['users.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('organization_id', 'name', name='uq_database_connections_organization_id_name')
    )
    op.create_index(op.f('ix_database_connections_created_at'), 'database_connections', ['created_at'], unique=False)
    op.create_index(op.f('ix_database_connections_dialect'), 'database_connections', ['dialect'], unique=False)
    op.create_index(op.f('ix_database_connections_environment'), 'database_connections', ['environment'], unique=False)
    op.create_index(op.f('ix_database_connections_last_test_status'), 'database_connections', ['last_test_status'], unique=False)
    op.create_index(op.f('ix_database_connections_organization_id'), 'database_connections', ['organization_id'], unique=False)
    op.create_index(op.f('ix_database_connections_status'), 'database_connections', ['status'], unique=False)

    # 2. connection_policies
    op.create_table(
        'connection_policies',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('organization_id', sa.UUID(), nullable=False),
        sa.Column('connection_id', sa.UUID(), nullable=False),
        sa.Column('approved_schemas_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('blocked_schemas_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('allow_schema_scanning', sa.Boolean(), nullable=False),
        sa.Column('allow_query_generation', sa.Boolean(), nullable=False),
        sa.Column('allow_query_execution', sa.Boolean(), nullable=False),
        sa.Column('approval_mode', sa.String(length=50), nullable=False),
        sa.Column('max_statement_timeout_ms', sa.Integer(), nullable=False),
        sa.Column('max_lock_timeout_ms', sa.Integer(), nullable=False),
        sa.Column('max_rows', sa.Integer(), nullable=False),
        sa.Column('max_response_bytes', sa.Integer(), nullable=False),
        sa.Column('max_estimated_rows', sa.Integer(), nullable=False),
        sa.Column('max_estimated_cost', sa.Float(), nullable=False),
        sa.Column('max_joined_tables', sa.Integer(), nullable=False),
        sa.Column('max_subquery_depth', sa.Integer(), nullable=False),
        sa.Column('allow_system_catalogs', sa.Boolean(), nullable=False),
        sa.Column('allow_cross_joins', sa.Boolean(), nullable=False),
        sa.Column('require_fully_qualified_tables', sa.Boolean(), nullable=False),
        sa.Column('created_by_user_id', sa.UUID(), nullable=False),
        sa.Column('updated_by_user_id', sa.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        
        sa.CheckConstraint('max_estimated_cost > 0', name='chk_max_estimated_cost'),
        sa.CheckConstraint('max_estimated_rows > 0', name='chk_max_estimated_rows'),
        sa.CheckConstraint('max_joined_tables > 0', name='chk_max_joined_tables'),
        sa.CheckConstraint('max_lock_timeout_ms > 0', name='chk_max_lock_timeout'),
        sa.CheckConstraint('max_response_bytes > 0', name='chk_max_response_bytes'),
        sa.CheckConstraint('max_rows > 0', name='chk_max_rows'),
        sa.CheckConstraint('max_statement_timeout_ms > 0', name='chk_max_statement_timeout'),
        sa.CheckConstraint('max_subquery_depth > 0', name='chk_max_subquery_depth'),
        
        sa.ForeignKeyConstraint(['connection_id'], ['database_connections.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['updated_by_user_id'], ['users.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_connection_policies_connection_id'), 'connection_policies', ['connection_id'], unique=True)
    op.create_index(op.f('ix_connection_policies_organization_id'), 'connection_policies', ['organization_id'], unique=False)

def downgrade() -> None:
    # Drop connection_policies first due to FK
    op.drop_index(op.f('ix_connection_policies_organization_id'), table_name='connection_policies')
    op.drop_index(op.f('ix_connection_policies_connection_id'), table_name='connection_policies')
    op.drop_table('connection_policies')
    
    # Drop database_connections
    op.drop_index(op.f('ix_database_connections_status'), table_name='database_connections')
    op.drop_index(op.f('ix_database_connections_organization_id'), table_name='database_connections')
    op.drop_index(op.f('ix_database_connections_last_test_status'), table_name='database_connections')
    op.drop_index(op.f('ix_database_connections_environment'), table_name='database_connections')
    op.drop_index(op.f('ix_database_connections_dialect'), table_name='database_connections')
    op.drop_index(op.f('ix_database_connections_created_at'), table_name='database_connections')
    op.drop_table('database_connections')

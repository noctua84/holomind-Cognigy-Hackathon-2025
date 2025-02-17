"""Add run_id to metrics table

Revision ID: add_run_id_to_metrics
"""
from alembic import op
import sqlalchemy as sa

def upgrade():
    # Add run_id column
    op.add_column('metrics', sa.Column('run_id', sa.Integer(), nullable=True))
    
    # Add foreign key constraint
    op.create_foreign_key(
        'fk_metrics_run_id', 'metrics', 'runs',
        ['run_id'], ['id']
    )

def downgrade():
    # Remove foreign key constraint
    op.drop_constraint('fk_metrics_run_id', 'metrics', type_='foreignkey')
    
    # Remove run_id column
    op.drop_column('metrics', 'run_id') 
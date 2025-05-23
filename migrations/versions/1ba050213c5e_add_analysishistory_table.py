"""Add AnalysisHistory table

Revision ID: 1ba050213c5e
Revises: 5a48e9663e64
Create Date: 2025-04-30 10:16:37.171326

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1ba050213c5e'
down_revision = '5a48e9663e64'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('analysis_history',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.Column('original_filename', sa.String(length=200), nullable=False),
    sa.Column('stored_filename', sa.String(length=100), nullable=False),
    sa.Column('predicted_class', sa.String(length=50), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=False),
    sa.Column('severity_level', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('stored_filename')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('analysis_history')
    # ### end Alembic commands ###

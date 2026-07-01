"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-06-30
"""
from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pipeline_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=True),
        sa.Column("dsc_name", sa.String(length=100), nullable=True),
        sa.Column("ice_volume_m3", sa.Float(), nullable=True),
        sa.Column("n_candidates", sa.Integer(), nullable=True),
        sa.Column("n_dsc_craters", sa.Integer(), nullable=True),
        sa.Column("dash_feasible", sa.Boolean(), nullable=True),
        sa.Column("slam_mae", sa.Float(), nullable=True),
        sa.Column("efpi_ice_pct", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("duration_s", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "stage_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("pipeline_runs.id"), nullable=False),
        sa.Column("stage_index", sa.Integer(), nullable=False),
        sa.Column("stage_name", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=True),
        sa.Column("log_output", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "figures",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("pipeline_runs.id"), nullable=False),
        sa.Column("filename", sa.String(length=100), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("figures")
    op.drop_table("stage_results")
    op.drop_table("pipeline_runs")

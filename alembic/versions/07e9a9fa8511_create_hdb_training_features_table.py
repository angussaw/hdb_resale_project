"""create_hdb_training_features_table

Revision ID: 07e9a9fa8511
Revises: 
Create Date: 2025-03-14 15:19:00.577379

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = '07e9a9fa8511'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'hdb_training_features',
        sa.Column('month', sa.Integer),
        sa.Column('town', sa.String(50)),
        sa.Column('flat_type', sa.String(50)),
        sa.Column('block', sa.String(50)),
        sa.Column('street_name', sa.String(100)),
        sa.Column('storey_range', sa.String(50)),
        sa.Column('floor_area_sqm', sa.Float),
        sa.Column('flat_model', sa.String(50)),
        sa.Column('lease_commence_date', sa.Integer),
        sa.Column('remaining_lease', sa.String(50)),
        sa.Column('resale_price', sa.Float),
        sa.Column('cpi', sa.Float),
        sa.Column('region', sa.String(50)),
        sa.Column('year_month', sa.String(50)),
        sa.Column('year', sa.Integer),
        sa.Column('lease_age', sa.Integer),
        sa.Column('latitude', sa.Float),
        sa.Column('longitude', sa.Float),
        sa.Column('no_of_malls_within_2_km', sa.Integer),
        sa.Column('distance_to_nearest_malls', sa.Float),
        sa.Column('no_of_schools_within_2_km', sa.Integer),
        sa.Column('distance_to_nearest_schools', sa.Float),
        sa.Column('no_of_parks_within_2_km', sa.Integer),
        sa.Column('distance_to_nearest_parks', sa.Float),
        sa.Column('no_of_MRT_stations_within_2_km', sa.Integer),
        sa.Column('distance_to_nearest_MRT_stations', sa.Float),
        sa.Column('date_context', sa.String(50))
    )

def downgrade() -> None:
    op.drop_table('hdb_training_features')
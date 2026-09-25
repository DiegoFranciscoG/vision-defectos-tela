"""Reference catalogs: carpet defect classes (with textrack codes) and the MVTec AD dataset.

The values are written here instead of imported from the application so the migration keeps
producing the same rows even if the code changes later.

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

defect_classes = sa.table(
    "defect_classes",
    sa.column("code", sa.String),
    sa.column("name_es", sa.String),
    sa.column("is_defect", sa.Boolean),
    sa.column("is_hole", sa.Boolean),
    sa.column("textrack_code", sa.String),
)
datasets = sa.table(
    "datasets",
    sa.column("code", sa.String),
    sa.column("name", sa.String),
    sa.column("version", sa.String),
    sa.column("source_url", sa.String),
    sa.column("license_spdx", sa.String),
    sa.column("license_url", sa.String),
    sa.column("commercial_use_allowed", sa.Boolean),
    sa.column("redistribution_allowed", sa.Boolean),
    sa.column("citation", sa.Text),
    sa.column("created_at", sa.DateTime(timezone=True)),
)

MVTEC_CITATION = (
    "Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., Steger, C. (2021). The MVTec Anomaly "
    "Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. "
    "International Journal of Computer Vision 129, 1038-1059. DOI 10.1007/s11263-020-01400-4. "
    "Bergmann, P., Fauser, M., Sattlegger, D., Steger, C. (2019). MVTec AD - A Comprehensive "
    "Real-World Dataset for Unsupervised Anomaly Detection. IEEE CVPR 2019."
)


def upgrade() -> None:
    op.bulk_insert(
        defect_classes,
        [
            {"code": "good", "name_es": "Sin defecto", "is_defect": False, "is_hole": False,
             "textrack_code": None},
            {"code": "color", "name_es": "Mancha o variación de color", "is_defect": True,
             "is_hole": False, "textrack_code": "SHADE_VARIATION"},
            {"code": "cut", "name_es": "Corte en la superficie", "is_defect": True,
             "is_hole": False, "textrack_code": None},
            {"code": "hole", "name_es": "Agujero", "is_defect": True, "is_hole": True,
             "textrack_code": "FABRIC_HOLE"},
            {"code": "metal_contamination", "name_es": "Fragmento metálico", "is_defect": True,
             "is_hole": False, "textrack_code": "BROKEN_NEEDLE"},
            {"code": "thread", "name_es": "Hilo suelto", "is_defect": True, "is_hole": False,
             "textrack_code": "LOOSE_THREAD"},
        ],
    )  # fmt: skip
    op.bulk_insert(
        datasets,
        [
            {
                "code": "mvtec_ad_carpet",
                "name": "MVTec Anomaly Detection Dataset - carpet",
                "version": "2019",
                "source_url": "https://www.mvtec.com/company/research/datasets/mvtec-ad",
                "license_spdx": "CC-BY-NC-SA-4.0",
                "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                "commercial_use_allowed": False,
                "redistribution_allowed": True,
                "citation": MVTEC_CITATION,
                "created_at": datetime(2026, 9, 25, tzinfo=UTC),
            }
        ],
    )


def downgrade() -> None:
    op.execute(datasets.delete().where(datasets.c.code == "mvtec_ad_carpet"))
    op.execute(defect_classes.delete())

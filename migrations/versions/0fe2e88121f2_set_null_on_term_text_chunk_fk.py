"""set null on term text chunk fk

Revision ID: 0fe2e88121f2
Revises: 17d95e97e0b5
Create Date: 2026-05-30 12:16:42.001630

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0fe2e88121f2"
down_revision: str | Sequence[str] | None = "17d95e97e0b5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint("terms_text_chunk_id_fkey", "terms", type_="foreignkey")
    op.create_foreign_key(
        "terms_text_chunk_id_fkey",
        "terms",
        "text_chunks",
        ["text_chunk_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("terms_text_chunk_id_fkey", "terms", type_="foreignkey")
    op.create_foreign_key(
        "terms_text_chunk_id_fkey",
        "terms",
        "text_chunks",
        ["text_chunk_id"],
        ["id"],
    )

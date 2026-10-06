"""add updated_by and updated_at to face_embeddings

Adds record-audit columns:
  - updated_by  UUID         : who last changed the row (set by the app on UPDATE)
  - updated_at  TIMESTAMPTZ  : when it was last changed (auto-maintained by a trigger)

Revision ID: 18b60258ad16
Revises: 212a74e721cb
Create Date: 2026-10-05
"""
from alembic import op

revision = "18b60258ad16"
down_revision = "212a74e721cb"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE face_embeddings ADD COLUMN IF NOT EXISTS updated_by UUID;")
    op.execute("ALTER TABLE face_embeddings ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();")

    # Keep updated_at current automatically on every UPDATE.
    op.execute("""
        CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("DROP TRIGGER IF EXISTS trg_face_embeddings_updated_at ON face_embeddings;")
    op.execute("""
        CREATE TRIGGER trg_face_embeddings_updated_at
        BEFORE UPDATE ON face_embeddings
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_face_embeddings_updated_at ON face_embeddings;")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at();")
    op.execute("ALTER TABLE face_embeddings DROP COLUMN IF EXISTS updated_at;")
    op.execute("ALTER TABLE face_embeddings DROP COLUMN IF EXISTS updated_by;")

"""initial face schema (face_embeddings + face_verification_log + pgvector)

Revision ID: 212a74e721cb
Revises:
Create Date: 2026-10-05

Idempotent (CREATE ... IF NOT EXISTS) so it also records the version on a
database whose tables were already created from app/db/schema.sql.
"""
from alembic import op

revision = "212a74e721cb"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    op.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id       UUID        NOT NULL,
            organization_id UUID        NOT NULL,
            employee_id     UUID        NOT NULL,
            employee_name   TEXT,
            embedding       vector(128) NOT NULL,
            created_by      UUID,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_face_tenant_employee ON face_embeddings (tenant_id, employee_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_face_tenant_org ON face_embeddings (tenant_id, organization_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_face_embedding_hnsw ON face_embeddings USING hnsw (embedding vector_cosine_ops);")

    op.execute("""
        CREATE TABLE IF NOT EXISTS face_verification_log (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id         UUID        NOT NULL,
            organization_id   UUID,
            matched_employee  UUID,
            employee_name     TEXT,
            cosine            REAL,
            passive_score     REAL,
            active_passed     BOOLEAN,
            verified          BOOLEAN,
            reason            TEXT,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_faceverif_tenant_time ON face_verification_log (tenant_id, created_at DESC);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS face_verification_log;")
    op.execute("DROP TABLE IF EXISTS face_embeddings;")
    # extension is left in place (other objects may rely on it)

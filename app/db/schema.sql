-- Schema for the hrms_face database (PostgreSQL + pgvector).
-- Mirrors the HRMS convention: all ids are UUID, multi-tenant via tenant_id,
-- organization-scoped, with created_by audit columns.
--     psql "$PG_DSN" -f app/db/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per enrolled face image. An employee may have several rows
-- (multiple enrollment photos) which we average into a centroid at match time.
CREATE TABLE IF NOT EXISTS face_embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID        NOT NULL,              -- company (hrms_sharedservices.tenants)
    organization_id UUID        NOT NULL,              -- org unit (masterdata.organizations)
    employee_id     UUID        NOT NULL,              -- masterdata.employees(id)
    employee_name   TEXT,
    embedding       vector(128) NOT NULL,              -- SFace output, L2-normalized
    created_by      UUID,                              -- who enrolled this face (acting user)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Tenant / org / employee scoped lookups and deletes.
CREATE INDEX IF NOT EXISTS idx_face_tenant_employee
    ON face_embeddings (tenant_id, employee_id);
CREATE INDEX IF NOT EXISTS idx_face_tenant_org
    ON face_embeddings (tenant_id, organization_id);

-- Approximate nearest-neighbour index for cosine distance (<=>).
CREATE INDEX IF NOT EXISTS idx_face_embedding_hnsw
    ON face_embeddings USING hnsw (embedding vector_cosine_ops);

-- Audit trail of verification attempts.
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

CREATE INDEX IF NOT EXISTS idx_faceverif_tenant_time
    ON face_verification_log (tenant_id, created_at DESC);

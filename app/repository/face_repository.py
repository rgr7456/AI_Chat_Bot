"""pgvector-backed storage for employee face embeddings.

All ids are UUIDs (matching the HRMS model). Every operation is tenant-scoped,
and optionally organization-scoped.
"""
from typing import List, Optional, Tuple
from uuid import UUID

import numpy as np

from app.db.pool import get_pool


def insert_embedding(tenant_id: UUID, organization_id: UUID, employee_id: UUID,
                     employee_name: str, embedding: np.ndarray,
                     created_by: Optional[UUID] = None) -> None:
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    with get_pool().connection() as conn:
        conn.execute(
            """INSERT INTO face_embeddings
               (tenant_id, organization_id, employee_id, employee_name,
                embedding, created_by)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (tenant_id, organization_id, employee_id, employee_name, vec, created_by),
        )


def search(tenant_id: UUID, embedding: np.ndarray, top_k: int = 5,
           organization_id: Optional[UUID] = None) -> List[Tuple[UUID, str, UUID, float]]:
    """Return [(employee_id, employee_name, organization_id, cosine_similarity)]
    nearest within the tenant (and org, if given)."""
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    clause = "WHERE tenant_id = %s"
    params: list = [vec, tenant_id]
    if organization_id is not None:
        clause += " AND organization_id = %s"
        params.append(organization_id)
    params.extend([vec, top_k])
    with get_pool().connection() as conn:
        rows = conn.execute(
            f"""SELECT employee_id, employee_name, organization_id,
                       1 - (embedding <=> %s) AS similarity
                FROM face_embeddings
                {clause}
                ORDER BY embedding <=> %s
                LIMIT %s""",
            tuple(params),
        ).fetchall()
    return [(r[0], r[1], r[2], float(r[3])) for r in rows]


def get_employee(tenant_id: UUID, employee_id: UUID) -> Optional[dict]:
    """Return {employee_name, organization_id} for an enrolled employee, else None."""
    with get_pool().connection() as conn:
        row = conn.execute(
            """SELECT employee_name, organization_id
               FROM face_embeddings
               WHERE tenant_id = %s AND employee_id = %s
               LIMIT 1""",
            (tenant_id, employee_id),
        ).fetchone()
    if not row:
        return None
    return {"employee_name": row[0], "organization_id": str(row[1])}


def fetch_embeddings_for_employee(tenant_id: UUID, employee_id: UUID) -> List[np.ndarray]:
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT embedding FROM face_embeddings WHERE tenant_id = %s AND employee_id = %s",
            (tenant_id, employee_id),
        ).fetchall()
    out = []
    for r in rows:
        e = np.asarray(r[0], dtype=np.float32).reshape(-1)
        out.append(e / (np.linalg.norm(e) + 1e-10))
    return out


def list_employees(tenant_id: UUID, organization_id: Optional[UUID] = None) -> List[dict]:
    clause = "WHERE tenant_id = %s"
    params: list = [tenant_id]
    if organization_id is not None:
        clause += " AND organization_id = %s"
        params.append(organization_id)
    with get_pool().connection() as conn:
        rows = conn.execute(
            f"""SELECT employee_id, MAX(employee_name) AS employee_name,
                       organization_id, COUNT(*) AS enrollments
                FROM face_embeddings {clause}
                GROUP BY employee_id, organization_id
                ORDER BY employee_id""",
            tuple(params),
        ).fetchall()
    return [{"employee_id": str(r[0]), "employee_name": r[1],
             "organization_id": str(r[2]), "enrollments": int(r[3])} for r in rows]


def delete_employee(tenant_id: UUID, employee_id: UUID) -> int:
    with get_pool().connection() as conn:
        cur = conn.execute(
            "DELETE FROM face_embeddings WHERE tenant_id = %s AND employee_id = %s",
            (tenant_id, employee_id),
        )
        return cur.rowcount


def log_verification(tenant_id: UUID, organization_id: Optional[UUID],
                     matched_employee: Optional[UUID], employee_name: Optional[str],
                     cosine: Optional[float], passive_score: Optional[float],
                     active_passed: Optional[bool], verified: bool, reason: str) -> None:
    with get_pool().connection() as conn:
        conn.execute(
            """INSERT INTO face_verification_log
               (tenant_id, organization_id, matched_employee, employee_name,
                cosine, passive_score, active_passed, verified, reason)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (tenant_id, organization_id, matched_employee, employee_name,
             cosine, passive_score, active_passed, verified, reason),
        )


def recent_verifications(tenant_id: UUID, limit: int = 10) -> List[dict]:
    with get_pool().connection() as conn:
        rows = conn.execute(
            """SELECT employee_name, matched_employee, cosine, verified, reason, created_at
               FROM face_verification_log
               WHERE tenant_id = %s
               ORDER BY created_at DESC
               LIMIT %s""",
            (tenant_id, limit),
        ).fetchall()
    return [{
        "employee_name": r[0],
        "matched_employee": str(r[1]) if r[1] else None,
        "cosine": float(r[2]) if r[2] is not None else None,
        "verified": bool(r[3]) if r[3] is not None else None,
        "reason": r[4],
        "created_at": r[5].isoformat() if r[5] else None,
    } for r in rows]


def centroid(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
    if not embeddings:
        return None
    c = np.mean(np.stack(embeddings, axis=0), axis=0)
    return c / (np.linalg.norm(c) + 1e-10)

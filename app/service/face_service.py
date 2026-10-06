"""Business logic: enrollment, 1:N verification, and punch-in liveness flow.

All ids are UUIDs (HRMS model): tenant_id, organization_id, employee_id, created_by.
Punch-in is *verify-only*: it confirms identity + liveness and returns the matched
employee. Recording attendance is left to the HRMS time-management service.
"""
from typing import List, Optional
from uuid import UUID

import numpy as np

from app.core.config import settings
from app.ml.engine import engine, decode_image
from app.ml.recognizer import cosine_similarity
from app.repository import face_repository as repo


class FaceError(Exception):
    """Raised for expected, client-facing problems (no face, spoof, etc.)."""


def to_uuid(value, field: str) -> UUID:
    """Validate and convert an incoming id to UUID (bad format -> FaceError -> 422)."""
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        raise FaceError(f"invalid_uuid:{field}")


def _opt_uuid(value, field: str) -> Optional[UUID]:
    if value is None or value == "":
        return None
    return to_uuid(value, field)


# ---------------------------------------------------------------- enrollment
def enroll(tenant_id, organization_id, employee_id, employee_name,
           images: List[bytes], created_by=None,
           require_liveness: bool = True) -> dict:
    t = to_uuid(tenant_id, "tenant_id")
    org = to_uuid(organization_id, "organization_id")
    emp = to_uuid(employee_id, "employee_id")
    cby = _opt_uuid(created_by, "created_by")

    stored = 0
    for raw in images:
        img = decode_image(raw)
        face = engine.detector.detect_one(img)
        if face is None:
            raise FaceError("no_face_detected")
        # Enrollment does not hard-block on passive liveness (admin action,
        # photos are allowed). Liveness is enforced at verify/punch time.
        emb = engine.recognizer.embed(img, face)
        repo.insert_embedding(t, org, emp, employee_name, emb, created_by=cby)
        stored += 1
    return {"employee_id": str(emp), "employee_name": employee_name,
            "organization_id": str(org), "enrolled_images": stored}


# ------------------------------------------------------------ matching core
def _match(tenant_id: UUID, embedding: np.ndarray,
           organization_id: Optional[UUID]) -> Optional[dict]:
    candidates = repo.search(tenant_id, embedding, top_k=5, organization_id=organization_id)
    if not candidates:
        return None
    for employee_id, employee_name, org_id, ann_sim in candidates:
        stored = repo.fetch_embeddings_for_employee(tenant_id, employee_id)
        c = repo.centroid(stored)
        sim = cosine_similarity(c, embedding) if c is not None else ann_sim
        if sim >= settings.FACE_MATCH_COSINE:
            return {"employee_id": str(employee_id), "employee_name": employee_name,
                    "organization_id": str(org_id), "cosine": round(float(sim), 4)}
    return None


# --------------------------------------------------------- simple verify (1 image)
def verify(tenant_id, image_bytes: bytes, organization_id=None,
           require_liveness: bool = True) -> dict:
    t = to_uuid(tenant_id, "tenant_id")
    org = _opt_uuid(organization_id, "organization_id")

    img = decode_image(image_bytes)
    face = engine.detector.detect_one(img)
    if face is None:
        raise FaceError("no_face_detected")

    passive_score = None
    if require_liveness and engine.liveness is not None:
        live = engine.liveness.score(img, face.box)
        passive_score = round(live.score, 4)
        if settings.LIVENESS_PASSIVE_ENFORCE and not live.is_real:
            repo.log_verification(t, org, None, None, None, passive_score, None, False, "passive_liveness_failed")
            raise FaceError(f"liveness_failed(score={live.score:.2f})")

    emb = engine.recognizer.embed(img, face)
    match = _match(t, emb, org)
    verified = match is not None
    repo.log_verification(
        t, org,
        to_uuid(match["employee_id"], "employee_id") if match else None,
        match["employee_name"] if match else None,
        match["cosine"] if match else None, passive_score, None, verified,
        "ok" if verified else "no_match",
    )
    if not verified:
        raise FaceError("no_match")
    return {"verified": True, "passive_score": passive_score, **match}


# ----------------------------------------------------------- punch-in flow
def start_punch_challenge(tenant_id) -> dict:
    to_uuid(tenant_id, "tenant_id")  # validate
    cid, action, prompt = engine.challenges.start(str(tenant_id))
    return {"challenge_id": cid, "action": action, "prompt": prompt,
            "ttl_seconds": settings.CHALLENGE_TTL_SECONDS}


def verify_punch(tenant_id, challenge_id: str, frames: List[bytes],
                 organization_id=None, require_liveness: bool = True) -> dict:
    t = to_uuid(tenant_id, "tenant_id")
    org = _opt_uuid(organization_id, "organization_id")
    if not frames:
        raise FaceError("no_frames_submitted")

    images = [decode_image(f) for f in frames]
    faces = [engine.detector.detect_one(im) for im in images]

    # 1) Active challenge: requested head-turn + a frontal frame must be present.
    active_passed, reason = engine.challenges.verify(challenge_id, str(t), faces)
    if not active_passed:
        repo.log_verification(t, org, None, None, None, None, False, False, reason)
        raise FaceError(f"active_liveness_failed({reason})")

    # Pick the most frontal frame for passive liveness + recognition.
    from app.ml.challenge import yaw_metric
    best_idx = min(
        (i for i, f in enumerate(faces) if f is not None),
        key=lambda i: abs(yaw_metric(faces[i])),
        default=None,
    )
    if best_idx is None:
        raise FaceError("no_face_detected")
    img, face = images[best_idx], faces[best_idx]

    # 2) Passive liveness on the frontal frame.
    passive_score = None
    if require_liveness and engine.liveness is not None:
        live = engine.liveness.score(img, face.box)
        passive_score = round(live.score, 4)
        if settings.LIVENESS_PASSIVE_ENFORCE and not live.is_real:
            repo.log_verification(t, org, None, None, None, passive_score, True, False, "passive_liveness_failed")
            raise FaceError(f"passive_liveness_failed(score={live.score:.2f})")

    # 3) Recognition.
    emb = engine.recognizer.embed(img, face)
    match = _match(t, emb, org)
    verified = match is not None
    repo.log_verification(
        t, org,
        to_uuid(match["employee_id"], "employee_id") if match else None,
        match["employee_name"] if match else None,
        match["cosine"] if match else None, passive_score, True, verified,
        "ok" if verified else "no_match",
    )
    if not verified:
        raise FaceError("no_match")
    return {"verified": True, "active_passed": True, "passive_score": passive_score, **match}


# --------------------------------------------------------------- admin
def list_employees(tenant_id, organization_id=None) -> dict:
    t = to_uuid(tenant_id, "tenant_id")
    org = _opt_uuid(organization_id, "organization_id")
    employees = repo.list_employees(t, org)
    return {"count": len(employees), "employees": employees}


def verify_employee(tenant_id, employee_id, challenge_id: str, frames: List[bytes],
                    organization_id=None, require_liveness: bool = True) -> dict:
    """1:1 verification: liveness + match ONLY against the given employee_id.

    Flow: validate employee id -> employee must be enrolled -> active liveness ->
    passive liveness -> compare to THIS employee's face only.
    Raises FaceError with a specific code the UI maps to a message:
      invalid_uuid:employee_id | employee_not_found |
      active_liveness_failed(...) | passive_liveness_failed(...) | face_not_matched
    """
    t = to_uuid(tenant_id, "tenant_id")
    emp = to_uuid(employee_id, "employee_id")       # -> invalid_uuid:employee_id
    org = _opt_uuid(organization_id, "organization_id")

    # 1) Employee must exist / be enrolled.
    info = repo.get_employee(t, emp)
    stored = repo.fetch_embeddings_for_employee(t, emp)
    if info is None or not stored:
        repo.log_verification(t, org, emp, None, None, None, None, False, "employee_not_found")
        raise FaceError("employee_not_found")

    if not frames:
        raise FaceError("no_frames_submitted")
    images = [decode_image(f) for f in frames]
    faces = [engine.detector.detect_one(im) for im in images]

    # 2) Active liveness (head-turn challenge).
    active_passed, reason = engine.challenges.verify(challenge_id, str(t), faces)
    if not active_passed:
        repo.log_verification(t, org, emp, info["employee_name"], None, None, False, False, reason)
        raise FaceError(f"active_liveness_failed({reason})")

    # Pick the most frontal frame.
    from app.ml.challenge import yaw_metric
    best_idx = min((i for i, f in enumerate(faces) if f is not None),
                   key=lambda i: abs(yaw_metric(faces[i])), default=None)
    if best_idx is None:
        raise FaceError("no_face_detected")
    img, face = images[best_idx], faces[best_idx]

    # 3) Passive liveness (advisory unless LIVENESS_PASSIVE_ENFORCE).
    passive_score = None
    if require_liveness and engine.liveness is not None:
        live = engine.liveness.score(img, face.box)
        passive_score = round(live.score, 4)
        if settings.LIVENESS_PASSIVE_ENFORCE and not live.is_real:
            repo.log_verification(t, org, emp, info["employee_name"], None, passive_score, True, False, "passive_liveness_failed")
            raise FaceError(f"passive_liveness_failed(score={live.score:.2f})")

    # 4) Face match — compare ONLY to this employee's enrolled faces.
    emb = engine.recognizer.embed(img, face)
    c = repo.centroid(stored)
    sim = cosine_similarity(c, emb) if c is not None else 0.0
    verified = sim >= settings.FACE_MATCH_COSINE
    repo.log_verification(t, org, emp, info["employee_name"], round(float(sim), 4),
                          passive_score, True, verified, "ok" if verified else "face_not_matched")
    if not verified:
        raise FaceError("face_not_matched")

    return {
        "verified": True,
        "employee_id": str(emp),
        "employee_name": info["employee_name"],
        "organization_id": info["organization_id"],
        "cosine": round(float(sim), 4),
        "passive_score": passive_score,
        "active_passed": True,
    }


def recent(tenant_id, limit: int = 10) -> dict:
    t = to_uuid(tenant_id, "tenant_id")
    items = repo.recent_verifications(t, max(1, min(limit, 50)))
    return {"count": len(items), "items": items}


def delete_employee(tenant_id, employee_id) -> dict:
    t = to_uuid(tenant_id, "tenant_id")
    emp = to_uuid(employee_id, "employee_id")
    deleted = repo.delete_employee(t, emp)
    return {"employee_id": str(emp), "deleted_rows": deleted}

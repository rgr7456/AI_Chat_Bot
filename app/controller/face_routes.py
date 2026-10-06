"""HTTP API for the face service. All routes are tenant-scoped; ids are UUIDs.

Endpoints
  POST   /face/enroll                 enroll one employee (1..n images)
  POST   /face/verify                 1:N verify a single image (+ passive liveness)
  POST   /face/punch/challenge        start a punch-in active-liveness challenge
  POST   /face/punch/verify           submit frames for the challenge; returns identity
  GET    /face/employees              list enrolled employees for the tenant
  DELETE /face/employees/{employee_id} delete an employee's embeddings
  GET    /face/health                 readiness probe

Tenant/actor come from the HRMS JWT when AUTH_ENABLED=true, otherwise from the
X-Tenant-Id / X-Actor-Id headers (dev).
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.auth import require_tenant, actor_id
from app.service import face_service as svc
from app.service.face_service import FaceError

log = logging.getLogger("face.routes")
router = APIRouter()


def _handle(fn, *args):
    try:
        return fn(*args)
    except (FaceError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        log.exception("unexpected error in %s", getattr(fn, "__name__", fn))
        raise HTTPException(status_code=500, detail="internal_error")


@router.get("/health")
async def health():
    ready = svc.engine.detector is not None
    return {"status": "ok" if ready else "loading", "models_loaded": ready}


@router.post("/enroll")
async def enroll(
    employee_id: str = Form(...),
    employee_name: str = Form(...),
    organization_id: str = Form(...),
    created_by: Optional[str] = Form(None),
    images: List[UploadFile] = File(...),
    tenant_id: str = Depends(require_tenant),
    actor: Optional[str] = Depends(actor_id),
):
    raw = [await f.read() for f in images]
    created = actor or created_by     # JWT actor wins in production
    return await run_in_threadpool(
        _handle, svc.enroll, tenant_id, organization_id, employee_id,
        employee_name, raw, created,
    )


@router.post("/verify")
async def verify(
    image: UploadFile = File(...),
    organization_id: Optional[str] = Query(None),
    tenant_id: str = Depends(require_tenant),
):
    raw = await image.read()
    return await run_in_threadpool(_handle, svc.verify, tenant_id, raw, organization_id)


@router.post("/punch/challenge")
async def punch_challenge(tenant_id: str = Depends(require_tenant)):
    return await run_in_threadpool(_handle, svc.start_punch_challenge, tenant_id)


@router.post("/punch/verify")
async def punch_verify(
    challenge_id: str = Form(...),
    frames: List[UploadFile] = File(...),
    organization_id: Optional[str] = Form(None),
    tenant_id: str = Depends(require_tenant),
):
    raw = [await f.read() for f in frames]
    return await run_in_threadpool(
        _handle, svc.verify_punch, tenant_id, challenge_id, raw, organization_id,
    )


@router.post("/verify-employee")
async def verify_employee(
    employee_id: str = Form(...),
    challenge_id: str = Form(...),
    frames: List[UploadFile] = File(...),
    organization_id: Optional[str] = Form(None),
    tenant_id: str = Depends(require_tenant),
):
    """1:1 liveness + face verification against a specific employee_id."""
    raw = [await f.read() for f in frames]
    return await run_in_threadpool(
        _handle, svc.verify_employee, tenant_id, employee_id, challenge_id, raw, organization_id,
    )


@router.get("/recent")
async def recent(
    limit: int = Query(10),
    tenant_id: str = Depends(require_tenant),
):
    return await run_in_threadpool(_handle, svc.recent, tenant_id, limit)


@router.get("/employees")
async def employees_list(
    organization_id: Optional[str] = Query(None),
    tenant_id: str = Depends(require_tenant),
):
    return await run_in_threadpool(_handle, svc.list_employees, tenant_id, organization_id)


@router.delete("/employees/{employee_id}")
async def employee_delete(employee_id: str, tenant_id: str = Depends(require_tenant)):
    return await run_in_threadpool(_handle, svc.delete_employee, tenant_id, employee_id)

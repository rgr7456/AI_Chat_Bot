"""Tenant resolution + optional JWT validation.

Every endpoint is tenant-scoped. During local dev (AUTH_ENABLED=false) the
tenant is taken from the `X-Tenant-Id` header. In production (AUTH_ENABLED=true)
the tenant is read from a validated HRMS JWT (Authorization: Bearer ...).
"""
from fastapi import Header, HTTPException
from typing import Optional
from jose import jwt, JWTError

from app.core.config import settings

# Common claim names HRMS tokens might use for the tenant.
_TENANT_CLAIMS = ("tenant_id", "tenantId", "tid", "tenant")


def require_tenant(
    authorization: Optional[str] = Header(default=None),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-Id"),
) -> str:
    if not settings.AUTH_ENABLED:
        if not x_tenant_id:
            raise HTTPException(status_code=400, detail="X-Tenant-Id header required (auth disabled)")
        return x_tenant_id

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        claims = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    for c in _TENANT_CLAIMS:
        if claims.get(c):
            return str(claims[c])
    raise HTTPException(status_code=403, detail="No tenant claim in token")


# Claim names that may carry the acting user's id.
_ACTOR_CLAIMS = ("user_id", "userId", "uid", "sub")


def actor_id(
    authorization: Optional[str] = Header(default=None),
    x_actor_id: Optional[str] = Header(default=None, alias="X-Actor-Id"),
) -> Optional[str]:
    """The acting user (used as created_by). From JWT when auth is on, else header."""
    if not settings.AUTH_ENABLED:
        return x_actor_id
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    try:
        claims = jwt.decode(authorization.split(" ", 1)[1].strip(),
                            settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        return None
    for c in _ACTOR_CLAIMS:
        if claims.get(c):
            return str(claims[c])
    return None

# HRMS Face Service

Commercial-free face recognition + liveness for HRMS punch-in, built as a
tenant-aware FastAPI microservice.

- **Detection:** OpenCV YuNet (MIT)
- **Recognition:** OpenCV SFace → 128-dim embeddings (Apache-2.0)
- **Passive liveness:** Silent-Face / MiniFASNet via ONNX (Apache-2.0)
- **Active liveness:** random head-turn challenge from landmarks
- **Vector store:** PostgreSQL + pgvector (`vector(128)`, cosine / HNSW)

All models are free for commercial use.

## Setup

```bash
# 1. Dependencies
pip install -r requirements.txt

# 2. Model files (see models/README.md)
#    downloads YuNet + SFace, and exports the Silent-Face ONNX models

# 3. Config
cp .env.example .env            # then fill PG_PASSWORD etc.

# 4. Database (pgvector must be installed in Postgres)
psql "$(python -c 'from app.core.config import settings; print(settings.dsn)')" -f app/db/schema.sql

# 5. Run
uvicorn app.main:app --reload --port 8000
# Swagger: http://127.0.0.1:8000/docs
```

> Postgres needs the `vector` extension. Locally, the `pgvector/pgvector:pg16`
> Docker image is the easiest route.

## Verify the models first (no DB needed)

```bash
python scripts/verify_models.py staff_images/**/*.jpg
```
Shows embedding dimension (should be 128), per-image liveness scores, and
same-person vs different-person cosine similarities so you can tune
`FACE_MATCH_COSINE`.

## API (all tenant-scoped)

All ids are **UUIDs** (matching the HRMS model): `tenant_id`, `organization_id`,
`employee_id`, `created_by`. Send `X-Tenant-Id` (and `X-Actor-Id` for `created_by`)
while `AUTH_ENABLED=false`; in production send a HRMS JWT as
`Authorization: Bearer <token>` and set `AUTH_ENABLED=true` (tenant + acting user
are then read from the token).

| Method | Path | Purpose |
|---|---|---|
| POST | `/face/enroll` | Enroll an employee (form: `employee_id`, `employee_name`, `organization_id`, `created_by?`, `images[]`) |
| POST | `/face/verify` | 1:N verify one image (passive liveness); optional `?organization_id=` |
| POST | `/face/punch/challenge` | Start a punch-in active-liveness challenge → `challenge_id` + action |
| POST | `/face/punch/verify` | Submit `challenge_id` + `frames[]` (+ optional `organization_id`); returns matched employee |
| GET | `/face/employees` | List enrolled employees for the tenant (optional `?organization_id=`) |
| DELETE | `/face/employees/{employee_id}` | Delete an employee's embeddings (right-to-erasure) |
| GET | `/face/health` | Readiness probe |

### Punch-in flow
1. `POST /face/punch/challenge` → `{challenge_id, action: "turn_left", prompt}`.
2. Client records a short burst of frames performing the action.
3. `POST /face/punch/verify` with `challenge_id` + the frames.
4. Service checks **active** (requested turn + a frontal frame) then **passive**
   liveness, then recognizes the face and returns
   `{verified, employee_id, employee_name, organization_id, cosine, passive_score}`.
   The HRMS time-management service records the actual attendance row.

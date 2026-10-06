# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A FastAPI microservice for HRMS staff face authentication and punch-in liveness. It is **commercial-license-clean**:

- **Detection:** OpenCV YuNet (`cv2.FaceDetectorYN`, MIT)
- **Recognition:** OpenCV SFace (`cv2.FaceRecognizerSF`, Apache-2.0) → **128-dim** L2-normalized embeddings
- **Passive liveness:** Silent-Face / MiniFASNet, two models run via ONNX Runtime (Apache-2.0)
- **Active liveness:** random head-turn challenge computed from YuNet's 5 landmarks (our code)
- **Vector store:** PostgreSQL + **pgvector** (`vector(128)`, cosine distance `<=>`, HNSW index)

There is intentionally **no PyTorch/TensorFlow/MediaPipe** at runtime — only OpenCV + ONNX Runtime. (The earlier Milvus + MTCNN + keras-facenet stack was replaced; see "Legacy" below.)

## Commands

```bash
pip install -r requirements.txt

# Download/export models into models/ (YuNet, SFace, Silent-Face ONNX)
#   see models/README.md

cp .env.example .env            # fill PG_PASSWORD etc.

# Create / migrate schema (Alembic; reads the DB URL from .env via app settings)
alembic upgrade head
#   alembic revision -m "msg"   # new migration    | alembic current / history
#   (app/db/schema.sql is kept only as a plain reference; Alembic is the source of truth)

# Run the API (from repo root so the `app` package resolves)
uvicorn app.main:app --reload --port 8000     # Swagger at /docs

# Phase-1 model proof, no DB needed — embedding dim + pairwise similarity + liveness
python scripts/verify_models.py staff_images/**/*.jpg

# Export Silent-Face .pth -> .onnx (needs torch in a scratch venv)
python scripts/export_silent_face_onnx.py --repo /tmp/silent-face --out models
```

No automated test suite or linter is configured yet.

## Architecture

Request flow:

`app/main.py` → `app/controller/face_routes.py` → `app/service/face_service.py` → `app/repository/face_repository.py` → pgvector
with ML in `app/ml/*` behind a single loaded `engine`.

- **`app/main.py`** — FastAPI app. A `lifespan` handler loads all models **once** at startup (`engine.load()` + `warmup()`) and opens/closes the pgvector pool. Models are never loaded at import time.
- **`app/ml/engine.py`** — holds the loaded singletons (`detector`, `recognizer`, `liveness`, `challenges`) and `decode_image()`. Import `engine` and use `engine.detector`, etc.; it is populated at startup.
- **`app/ml/detector.py`** — YuNet wrapper. Returns `Face` objects carrying the **raw 15-value YuNet row** (needed by `SFace.alignCrop`) plus parsed box/landmarks. Largest face first.
- **`app/ml/recognizer.py`** — SFace embedding (align → feature → L2-normalize) + `cosine_similarity`.
- **`app/ml/liveness.py`** — passive anti-spoofing. Crops per each model's scale (parsed from the filename, e.g. `2.7_80x80_...`), runs the ONNX models, averages softmax; class index **1 = real**.
- **`app/ml/challenge.py`** — active liveness. `yaw_metric()` estimates head yaw from nose-vs-eyes; `ChallengeStore` issues TTL'd one-time challenges and verifies that both a frontal frame and a frame turned in the requested direction are present. In-memory — back with Redis if scaled horizontally.
- **`app/service/face_service.py`** — `enroll`, `verify`, `start_punch_challenge`, `verify_punch`, `list_employees`, `delete_employee`. Matching = pgvector ANN top-k, then a **centroid** re-check of the best candidate's stored embeddings against `FACE_MATCH_COSINE`. Raises `FaceError` for expected client-facing conditions. All ids are validated/converted to `uuid.UUID` via `to_uuid()`.
- **`app/repository/face_repository.py`** — all pgvector SQL. **Every query is tenant-scoped (`WHERE tenant_id = ...`)**. Also writes `face_verification_log` audit rows.
- **`app/core/config.py`** — `settings` (pydantic-settings, reads `.env`). `settings.dsn` builds the Postgres DSN; `EMBEDDING_DIM = 128` is fixed by SFace.
- **`app/core/auth.py`** — `require_tenant` dependency. With `AUTH_ENABLED=false` it reads `X-Tenant-Id`; with `true` it validates a HRMS JWT and extracts a tenant claim.

### Data model

Employee-centric, **all ids are UUID** to match the HRMS (`hrms_masterdata`): `face_embeddings(id, tenant_id, organization_id, employee_id, employee_name, embedding vector(128), created_by, created_at)`. Lookups are scoped by `tenant_id` (and optionally `organization_id`). `created_by` = the acting user (from JWT `sub`/`user_id` in prod, `X-Actor-Id` header in dev). Endpoints: `/face/enroll`, `/face/verify`, `/face/punch/challenge`, `/face/punch/verify`, `GET /face/employees`, `DELETE /face/employees/{employee_id}`.

### Key conventions

- **Punch-in is verify-only**: it confirms identity + liveness and returns the match; recording attendance is the Java `hrms-time-management-service`'s job. This face service is the Python side of the larger `hrms_core` (Spring Boot) platform and is intended to be exposed to the agent service (`Kai`) as tools.
- Embeddings are **always L2-normalized**; distances are cosine. Keep the pgvector column at `vector(128)` unless the recognition model changes.
- Model file **names must not be renamed** — `liveness.py` parses scale/size from them.
- `FaceError` → HTTP 422 with a safe message; unexpected errors are logged server-side and returned as a generic 500 (never leak tracebacks to clients).
- Thresholds (`FACE_MATCH_COSINE`, `LIVENESS_*`) live in `.env`; calibrate `FACE_MATCH_COSINE` against real staff photos with `scripts/verify_models.py`.

### Licensing

YuNet (MIT), SFace (Apache-2.0), Silent-Face (Apache-2.0) are all commercial-safe. Do **not** reintroduce InsightFace `buffalo_l` or keras-facenet weights for commercial use — their pretrained weights are non-commercial/research-only unless separately licensed.

The legacy Milvus + dormant SQLAlchemy/Alembic code has been removed; the repo now contains only the active pgvector-based face + OCR service.

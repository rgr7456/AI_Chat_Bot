"""Central configuration for the face service (pgvector + SFace + liveness)."""
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- PostgreSQL / pgvector ---
    PG_HOST: str = "localhost"
    PG_PORT: int = 5432
    PG_DB: str = "hrms_face"
    PG_USER: str = "hrms_user"
    PG_PASSWORD: str = ""
    PG_DSN: Optional[str] = None  # if set, overrides the parts above

    # --- Model files ---
    YUNET_MODEL: str = "models/face_detection_yunet_2023mar.onnx"
    SFACE_MODEL: str = "models/face_recognition_sface_2021dec.onnx"
    ANTISPOOF_MODEL_1: str = "models/2.7_80x80_MiniFASNetV2.onnx"
    ANTISPOOF_MODEL_2: str = "models/4_0_0_80x80_MiniFASNetV1SE.onnx"

    # --- Recognition thresholds ---
    FACE_MATCH_COSINE: float = 0.38      # SFace cosine similarity for a match
    DETECT_SCORE_THRESHOLD: float = 0.8  # YuNet min detection score

    # --- Liveness ---
    LIVENESS_PASSIVE_THRESHOLD: float = 0.70
    # When False, the passive (Silent-Face) score is computed and returned but
    # does NOT reject — tune the threshold on your real camera first, then enable.
    # Active (head-turn) liveness is always enforced on punch-in regardless.
    LIVENESS_PASSIVE_ENFORCE: bool = False
    LIVENESS_TURN_THRESHOLD: float = 0.22
    LIVENESS_MIRROR: bool = False
    CHALLENGE_TTL_SECONDS: int = 30

    # --- OCR (Indian KYC documents: PAN / Aadhaar / bank) ---
    OCR_ENABLED: bool = True
    OCR_USE_ANGLE_CLS: bool = True   # handle rotated/skewed ID photos
    OCR_TEXT_SCORE: float = 0.5      # drop recognitions below this confidence
    # UIDAI rules restrict storing the full Aadhaar number. When True the
    # parser returns only 'aadhaar_number_masked' (last 4 digits).
    OCR_AADHAAR_MASK: bool = True

    # --- Auth ---
    AUTH_ENABLED: bool = False
    JWT_SECRET: str = ""
    JWT_ALGORITHM: str = "HS256"

    # --- CORS ---
    CORS_ORIGINS: str = "*"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def dsn(self) -> str:
        if self.PG_DSN:
            return self.PG_DSN
        return (
            f"postgresql://{self.PG_USER}:{self.PG_PASSWORD}"
            f"@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"
        )

    @property
    def cors_origins(self) -> List[str]:
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()

# Embedding dimension is fixed by the SFace model.
EMBEDDING_DIM = 128

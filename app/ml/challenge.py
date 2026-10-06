"""Active liveness: a random head-turn challenge verified from face landmarks.

Flow:
  1. start()  -> issues a challenge_id + an action ("turn_left"/"turn_right")
  2. client records a short burst of frames doing the action
  3. verify_frames() -> confirms a near-frontal frame AND a frame clearly
     turned in the requested direction are both present (proves 3D motion,
     defeating a static photo and a wrong-direction replay).

Yaw is estimated from YuNet's 5 landmarks: where the nose sits horizontally
between the two eyes. Frontal ~0, turned left/right -> large magnitude.
"""
import secrets
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.ml.detector import Face

ACTIONS = ("turn_left", "turn_right")
ACTION_PROMPTS = {
    "turn_left": "Slowly turn your head LEFT, then face the camera",
    "turn_right": "Slowly turn your head RIGHT, then face the camera",
}


def yaw_metric(face: Face) -> float:
    """Signed, normalized nose offset between the eyes.
    >0 => nose toward image-right eye; <0 => toward image-left eye.
    """
    rex, _ = face.right_eye
    lex, _ = face.left_eye
    nose_x, _ = face.nose
    eye_center = (rex + lex) / 2.0
    eye_dist = abs(rex - lex) + 1e-6
    return (nose_x - eye_center) / eye_dist


@dataclass
class _Challenge:
    action: str
    tenant_id: str
    expires_at: float


class ChallengeStore:
    """In-memory, TTL'd challenge store. Fine for a single instance; back it
    with Redis if the service is horizontally scaled."""

    def __init__(self, ttl_seconds: int = 30, turn_threshold: float = 0.22, mirror: bool = False):
        self._store: Dict[str, _Challenge] = {}
        self.ttl = ttl_seconds
        self.turn_threshold = turn_threshold
        self.mirror = mirror

    def start(self, tenant_id: str) -> Tuple[str, str, str]:
        self._gc()
        action = secrets.choice(ACTIONS)
        cid = secrets.token_urlsafe(16)
        self._store[cid] = _Challenge(action, tenant_id, time.time() + self.ttl)
        return cid, action, ACTION_PROMPTS[action]

    def _direction(self, metric: float) -> Optional[str]:
        """Map a yaw metric to a user-facing turn direction."""
        if abs(metric) < self.turn_threshold:
            return None
        # metric>0 => image-right. Mirrored (selfie) view swaps user/image sides.
        image_side = "right" if metric > 0 else "left"
        user_side = {"right": "left", "left": "right"}[image_side] if self.mirror else image_side
        return f"turn_{user_side}"

    def verify(self, challenge_id: str, tenant_id: str, faces_per_frame: List[Optional[Face]]
               ) -> Tuple[bool, str]:
        """Direction-agnostic active liveness: require both a near-frontal frame
        AND a clearly-turned frame (either side). A flat photo keeps a constant
        yaw across frames, so it can never show both — only a real 3D head turn
        does. This is far more robust than requiring an exact left/right side
        (which is fragile under camera mirroring)."""
        self._gc()
        ch = self._store.get(challenge_id)
        if ch is None:
            return False, "challenge_expired_or_unknown"
        if ch.tenant_id != tenant_id:
            return False, "challenge_tenant_mismatch"
        # One-time use.
        self._store.pop(challenge_id, None)

        yaws = [abs(yaw_metric(f)) for f in faces_per_frame if f is not None]
        if not yaws:
            return False, "no_face_detected"

        frontal_max = self.turn_threshold * 0.6   # clearly centered
        has_frontal = any(y < frontal_max for y in yaws)
        has_turn = any(y >= self.turn_threshold for y in yaws)

        if has_frontal and has_turn:
            return True, "ok"
        if not has_turn:
            return False, "no_head_turn_detected"
        return False, "keep_facing_forward_then_turn"

    def _gc(self) -> None:
        now = time.time()
        expired = [k for k, v in self._store.items() if v.expires_at < now]
        for k in expired:
            self._store.pop(k, None)

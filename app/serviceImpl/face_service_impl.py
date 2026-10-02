# from app.utils.face_utils import extract_face_embedding, read_image_bytes
# from app.repository.face_repository import (
#     insert_staff_embedding,
#     search_embedding,
#     fetch_embeddings_for_staff,
#     compute_centroid,
# )
# from typing import Optional
# import numpy as np


# def register_staff(staff_id: str, staff_name: str, image_bytes: bytes):
#     """Register single image for staff. Keeps API simple; consider adding multi-image enroll endpoint."""
#     image = read_image_bytes(image_bytes)
#     embedding = extract_face_embedding(image)
#     insert_staff_embedding(staff_id, staff_name, embedding)


# def verify_staff(image_bytes: bytes, threshold: float = 0.5, top_k: int = 5) -> Optional[dict]:
#     """Verify an incoming image. Tries centroid first (if multiple enrollments exist), then falls back to top_k search.

#     Returns a dict with staff_id, staff_name, distance if matched, else None.
#     """
#     image = read_image_bytes(image_bytes)
#     embedding = extract_face_embedding(image)

#     # Strategy: try centroid per top candidate if available
#     results = search_embedding(embedding, top_k=top_k)
#     if not results:
#         return None

#     # Iterate candidates - for each candidate try centroid-based comparison when possible
#     for hit in results:
#         candidate_id = hit.entity.get("staff_id")
#         # Fetch stored embeddings and compute centroid (if multiple)
#         stored_embs = fetch_embeddings_for_staff(candidate_id)
#         if stored_embs:
#             centroid = compute_centroid(stored_embs)
#             if centroid is not None:
#                 # compute cosine similarity
#                 sim = float(np.dot(centroid, embedding) / ((np.linalg.norm(centroid) * np.linalg.norm(embedding)) + 1e-10))
#                 dist = 1.0 - sim
#                 if dist <= threshold:
#                     return {"staff_id": candidate_id, "staff_name": hit.entity.get("staff_name"), "distance": dist}

#         # fallback to milvus-provided distance
#         if hit.distance <= threshold:
#             return {"staff_id": candidate_id, "staff_name": hit.entity.get("staff_name"), "distance": float(hit.distance)}

#     return None


from app.utils.face_utils import extract_face_embedding, read_image_bytes
from app.repository.face_repository import (
    insert_staff_embedding,
    search_embedding,
    fetch_embeddings_for_staff,
    compute_centroid,
)
from typing import Optional
import numpy as np


def register_staff(staff_id: str, staff_name: str, image_bytes):
    """Register single image or multiple images for a staff member.

    image_bytes may be:
    - bytes: single image
    - list[bytes]: multiple images

    The function will insert one embedding per image into the vector DB.
    """
    if isinstance(image_bytes, (list, tuple)):
        # multiple images
        for ib in image_bytes:
            image = read_image_bytes(ib)
            embedding = extract_face_embedding(image)
            insert_staff_embedding(staff_id, staff_name, embedding)
        return

    # single image
    image = read_image_bytes(image_bytes)
    embedding = extract_face_embedding(image)
    insert_staff_embedding(staff_id, staff_name, embedding)


def verify_staff(image_bytes: bytes, threshold: float = 0.5, top_k: int = 5) -> Optional[dict]:
    """Verify an incoming image. Tries centroid first (if multiple enrollments exist), then falls back to top_k search.

    Returns a dict with staff_id, staff_name, distance if matched, else None.
    """
    image = read_image_bytes(image_bytes)
    embedding = extract_face_embedding(image)

    # Strategy: try centroid per top candidate if available
    results = search_embedding(embedding, top_k=top_k)
    if not results:
        return None

    # Iterate candidates - for each candidate try centroid-based comparison when possible
    for hit in results:
        candidate_id = hit.entity.get("staff_id")
        # Fetch stored embeddings and compute centroid (if multiple)
        stored_embs = fetch_embeddings_for_staff(candidate_id)
        if stored_embs:
            centroid = compute_centroid(stored_embs)
            if centroid is not None:
                # compute cosine similarity
                sim = float(np.dot(centroid, embedding) / ((np.linalg.norm(centroid) * np.linalg.norm(embedding)) + 1e-10))
                dist = 1.0 - sim
                if dist <= threshold:
                    return {"staff_id": candidate_id, "staff_name": hit.entity.get("staff_name"), "distance": dist}

        # fallback to milvus-provided distance
        if hit.distance <= threshold:
            return {"staff_id": candidate_id, "staff_name": hit.entity.get("staff_name"), "distance": float(hit.distance)}

    return None


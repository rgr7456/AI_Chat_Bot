"""
Debug script: register a given image into Milvus and immediately search using the same embedding.
Helps diagnose why verification returns "no staff found" (checks embedding shapes, norms, stored vector and distances).

Usage:
  python tools/debug_register_and_search.py --image path/to/image.jpg --staff_id debug_1 --staff_name Debug

It will:
- compute embedding using app.utils.face_utils
- insert embedding via app.repository.face_repository.insert_staff_embedding
- fetch stored embeddings for the staff_id and show norm/first values
- run a search using the same embedding and print top_k distances and entity fields

Run inside your venv from repo root.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

# ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.utils.face_utils import read_image_bytes, extract_face_embedding
from app.repository.face_repository import insert_staff_embedding, fetch_embeddings_for_staff, search_embedding
from app.constants.constants import EMBEDDING_DIM, SEARCH_PARAMS

def cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    sim = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10))
    return 1.0 - sim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--staff_id', default='debug_test_1')
    parser.add_argument('--staff_name', default='Debug Test')
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print('Image not found:', img_path)
        return

    with open(img_path, 'rb') as f:
        b = f.read()

    print('Computing embedding...')
    img = read_image_bytes(b)
    emb = extract_face_embedding(img, debug=True)
    print('Embedding shape:', emb.shape, 'dtype:', emb.dtype, 'norm:', np.linalg.norm(emb))

    # Check dimension
    if emb.shape[0] != EMBEDDING_DIM:
        print(f'Warning: embedding dim {emb.shape[0]} != EMBEDDING_DIM {EMBEDDING_DIM}')

    print('Inserting embedding into Milvus...')
    insert_staff_embedding(args.staff_id, args.staff_name, emb)
    print('Inserted and flushed.')

    print('\nFetching stored embeddings for staff_id...')
    stored = fetch_embeddings_for_staff(args.staff_id)
    print(f'Found {len(stored)} stored embeddings for {args.staff_id}')
    for i, s in enumerate(stored):
        print(f' stored[{i}] norm={np.linalg.norm(s):.6f} first10={s[:10].tolist()}')

    print('\nSearching using same embedding...')
    results = search_embedding(emb, top_k=args.top_k, search_params=SEARCH_PARAMS)
    if not results:
        print('No results returned from Milvus search (empty).')
        return

    print(f'Got {len(results)} hits:')
    for i, hit in enumerate(results):
        try:
            eid = hit.id
        except Exception:
            eid = None
        ent = hit.entity
        staff_id = ent.get('staff_id') if ent else None
        staff_name = ent.get('staff_name') if ent else None
        dist = getattr(hit, 'distance', None)
        print(f' hit[{i}] id={eid} staff_id={staff_id} staff_name={staff_name} distance={dist}')

    # compute local distance to first stored embedding (if exists)
    if stored:
        d = cosine_distance(stored[0], emb)
        print(f'Local cosine distance to stored[0]: {d:.6f}')

if __name__ == '__main__':
    main()

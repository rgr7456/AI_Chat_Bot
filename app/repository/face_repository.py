import numpy as np
from pymilvus import connections, Collection, utility
from typing import List, Optional
from app.constants.constants import MILLIUS_COLLECTION_NAME, EMBEDDING_DIM, INDEX_PARAMS, SEARCH_PARAMS

def get_milvus_connection():
    """Ensure Milvus connection is established"""
    try:
        # Check if already connected
        if connections.has_connection("default"):
            return True
        # Connect to Milvus
        connections.connect("default", host="localhost", port="19530")
        return True
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise ConnectionError(f"Cannot connect to Milvus server at localhost:19530. Is Milvus running? Error: {e}")

def get_collection():
    """Get collection with proper error handling"""
    try:
        get_milvus_connection()
        
        # Check if collection exists
        if not utility.has_collection(MILLIUS_COLLECTION_NAME):
            raise RuntimeError(f"Collection '{MILLIUS_COLLECTION_NAME}' does not exist. Run: python app/utils/create_staff_collection.py")
        
        collection = Collection(MILLIUS_COLLECTION_NAME)
        return collection
    except Exception as e:
        print(f"Failed to get collection: {e}")
        raise

# Get collection instance (will be created when first accessed)
collection = None


def insert_staff_embedding(staff_id: str, staff_name: str, embedding: np.ndarray):
    """Insert one embedding (1,D) into Milvus with provided metadata."""
    try:
        collection = get_collection()
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1).tolist()
        entities = [
            [staff_id],
            [staff_name],
            emb,
        ]
        collection.insert(entities)
        collection.flush()
    except Exception as e:
        raise RuntimeError(f"Failed to insert staff embedding: {e}")


def prepare_collection(index_params: Optional[dict] = None):
    """Create index if missing and load the collection. Optionally override index params."""
    try:
        collection = get_collection()
        if index_params is None:
            index_params = INDEX_PARAMS
        
        try:
            if not collection.has_index():
                print(f"Creating index for collection '{MILLIUS_COLLECTION_NAME}'...")
                collection.create_index(field_name="embedding", index_params=index_params)
                print("✅ Index created successfully")
        except Exception as e:
            # If index creation fails, continue (collection may already exist)
            print(f"Index creation info: {e}")
            pass
        
        # Load collection into memory
        collection.load()
        print(f"✅ Collection '{MILLIUS_COLLECTION_NAME}' loaded successfully")
        
    except Exception as e:
        raise RuntimeError(f"Failed to prepare collection: {e}")


def search_embedding(embedding: np.ndarray, top_k: int = 5, search_params: Optional[dict] = None) -> List:
    """Search the collection and return list of hits.

    search_params: dict like {"metric_type":"COSINE","params":{"nprobe":20}}
    """
    try:
        prepare_collection()
        collection = get_collection()
        
        if search_params is None:
            search_params = SEARCH_PARAMS

        results = collection.search(
            data=[np.asarray(embedding, dtype=np.float32).tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["staff_id", "staff_name"],
        )
        return results[0]
    except Exception as e:
        raise RuntimeError(f"Failed to search embedding: {e}")


def fetch_embeddings_for_staff(staff_id: str) -> List[np.ndarray]:
    """Query Milvus for embeddings by staff_id and return list of numpy arrays."""
    try:
        collection = get_collection()
        expr = f'staff_id == "{staff_id}"'
        
        try:
            rows = collection.query(expr=expr, output_fields=["embedding"])
        except Exception as e:
            # If query fails (older client versions), return empty
            print(f"Query failed for staff_id {staff_id}: {e}")
            return []
            
        embs: List[np.ndarray] = []
        for r in rows:
            e = np.array(r.get("embedding"), dtype=np.float32)
            # normalize just in case
            n = np.linalg.norm(e) + 1e-10
            embs.append(e / n)
        return embs
        
    except Exception as e:
        print(f"Failed to fetch embeddings for staff {staff_id}: {e}")
        return []


def compute_centroid(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
    if not embeddings:
        return None
    stack = np.stack(embeddings, axis=0)
    centroid = np.mean(stack, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
    return centroid

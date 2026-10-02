MILLIUS_COLLECTION_NAME = "staff_faces"

EMBEDDING_DIM = 512
INDEX_PARAMS = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
# Default search params; increase nprobe for better recall (tune for your dataset)
SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"nprobe": 32}}

# Recommended face crop size for the embedder (FaceNet typical input)
MODEL_INPUT_SIZE = (160, 160)

# Default distance threshold suggestion (should be calibrated with tools/calibrate_threshold.py)
DEFAULT_DISTANCE_THRESHOLD = 0.5

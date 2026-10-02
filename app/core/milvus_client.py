# from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
# from app.core.config import settings

# def init_milvus():
#     connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)

#     fields = [
#         FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#         FieldSchema(name="staff_id", dtype=DataType.VARCHAR, max_length=50),
#         FieldSchema(name="staff_name", dtype=DataType.VARCHAR, max_length=100),
#         FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
#     ]

#     schema = CollectionSchema(fields, description="Staff face embeddings")
#     collection = Collection(name=settings.MILVUS_COLLECTION, schema=schema, consistency_level="Strong")

#     # Create index for search
#     index_params = {
#         "metric_type": "COSINE",
#         "index_type": "IVF_FLAT",
#         "params": {"nlist": 128},
#     }

#     collection.create_index(field_name="embedding", index_params=index_params)
#     return collection
    

# collection = init_milvus()

import os
import traceback

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
    exceptions,
)

from app.constants.constants import MILLIUS_COLLECTION_NAME, EMBEDDING_DIM, INDEX_PARAMS
from app.core.config import settings


def init_milvus(recreate_if_mismatch: bool = False):
    """Initialize connection and ensure the collection exists with the expected schema.

    If a collection exists with a mismatched vector dimension, the function will raise an
    error unless `recreate_if_mismatch` is True (or env RECREATE_COLLECTION=1).
    Returns the Collection object (loaded).
    """
    host = getattr(settings, "MILVUS_HOST", "localhost")
    port = getattr(settings, "MILVUS_PORT", "19530")
    collection_name = getattr(settings, "MILVUS_COLLECTION", MILLIUS_COLLECTION_NAME)

    connections.connect("default", host=host, port=port)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="staff_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="staff_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    schema = CollectionSchema(fields, description="Staff face embeddings")

    # If collection exists, inspect schema
    if utility.has_collection(collection_name):
        print(f"ℹ️ Collection '{collection_name}' already exists. Inspecting schema...")
        col = Collection(name=collection_name)
        try:
            vec_field = None
            for f in col.schema.fields:
                if f.name == "embedding":
                    vec_field = f
                    break
            if vec_field is None:
                raise RuntimeError("Existing collection does not have 'embedding' field")
            existing_dim = getattr(vec_field, "dim", None)
            print(f"Existing embedding dim: {existing_dim}, expected: {EMBEDDING_DIM}")
            if existing_dim != EMBEDDING_DIM:
                # mismatch detected
                recreate_flag = recreate_if_mismatch or os.environ.get("RECREATE_COLLECTION", "0") == "1"
                msg = (
                    f"Embedding dimension mismatch for collection '{collection_name}': "
                    f"existing={existing_dim} expected={EMBEDDING_DIM}."
                )
                if recreate_flag:
                    print(msg + " Recreating collection as RECREATE_COLLECTION=1.")
                    utility.drop_collection(collection_name)
                    collection = Collection(name=collection_name, schema=schema)
                    # create index after creation
                    try:
                        collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
                    except Exception as e:
                        print("Index creation warning:", e)
                else:
                    raise RuntimeError(msg + " If you want to recreate, set env RECREATE_COLLECTION=1 or pass recreate_if_mismatch=True")
            else:
                collection = col
                # ensure index exists
                try:
                    if not collection.has_index():
                        collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
                except Exception as e:
                    print("Index creation warning:", e)
        except Exception as e:
            print("Error inspecting existing collection:", e)
            traceback.print_exc()
            # write full traceback to a log file for easier debugging
            try:
                with open("milvus_client_error.log", "a", encoding="utf-8") as fh:
                    fh.write("\n--- Error inspecting existing collection ---\n")
                    fh.write(traceback.format_exc())
            except Exception:
                pass
            raise
    else:
        # Create collection
        try:
            collection = Collection(name=collection_name, schema=schema)
            print(f"✅ Collection '{collection_name}' created.")
            try:
                collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
            except Exception as e:
                print("Index creation warning:", e)
        except exceptions.MilvusException as e:
            print("Failed to create collection:", e)
            traceback.print_exc()
            try:
                with open("milvus_client_error.log", "a", encoding="utf-8") as fh:
                    fh.write("\n--- Failed to create collection ---\n")
                    fh.write(traceback.format_exc())
            except Exception:
                pass
            raise

    # Load collection for use
    try:
        collection.load()
        print(f"✅ Collection '{collection_name}' loaded.")
    except Exception as e:
        print("Warning: failed to load collection:", e)
        traceback.print_exc()
        try:
            with open("milvus_client_error.log", "a", encoding="utf-8") as fh:
                fh.write("\n--- Failed to load collection ---\n")
                fh.write(traceback.format_exc())
        except Exception:
            pass

    return collection


# Initialize at import time (callers can call init_milvus again if needed)
try:
    collection = init_milvus()
except Exception as ex:
    print("Milvus init error:", ex)
    collection = None

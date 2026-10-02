# from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, exceptions

# connections.connect("default", host="localhost", port="19530")

# collection_name = "staff_faces"
# fields = [
#     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#     FieldSchema(name="staff_id", dtype=DataType.VARCHAR, max_length=50),
#     FieldSchema(name="staff_name", dtype=DataType.VARCHAR, max_length=100),
#     FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
# ]
# schema = CollectionSchema(fields, description="Staff face embeddings")

# try:
#     # Try to instantiate collection with schema (will throw if collection exists or not)
#     collection = Collection(name=collection_name, schema=schema)
#     print(f"Collection '{collection_name}' created or loaded.")
# except exceptions.MilvusException as e:
#     if e.code == 4:
#         # Collection not found - try creating it by other means or fail gracefully
#         print(f"Collection '{collection_name}' does not exist, and creation via constructor failed.")
#         # Possibly retry by creating collection via Python client or prompt restart
#     else:
#         raise

# # Optionally load collection if created successfully
# try:
#     collection.load()
#     print(f"Collection '{collection_name}' loaded.")
# except Exception as load_err:
#     print(f"Could not load collection: {load_err}")


from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType, Collection, utility
)

# 1️⃣ Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# 2️⃣ Define collection name
collection_name = "staff_faces"

# 3️⃣ Define fields (schema)
fields = [
    FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True  # Auto-generate IDs
    ),
    FieldSchema(
        name="staff_id",
        dtype=DataType.VARCHAR,
        max_length=50
    ),
    FieldSchema(
        name="staff_name",
        dtype=DataType.VARCHAR,
        max_length=100
    ),
    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=512  # 512 dimensions for face embeddings
    )
]

# 4️⃣ Create schema
schema = CollectionSchema(
    fields=fields,
    description="Collection for storing staff face embeddings"
)

# 5️⃣ Check if collection exists
if utility.has_collection(collection_name):
    print(f"ℹ️ Collection '{collection_name}' already exists.")
    collection = Collection(name=collection_name)
else:
    # 6️⃣ Create new collection
    collection = Collection(name=collection_name, schema=schema)
    print(f"✅ Collection '{collection_name}' created successfully.")

# 7️⃣ Create index for fast vector search
index_params = {
    "metric_type": "COSINE",   # or "L2"
    "index_type": "IVF_FLAT",  # or "HNSW", "IVF_SQ8", etc.
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
print("✅ Index created for 'embedding' field.")

# 8️⃣ Load collection into memory
collection.load()
print("✅ Collection loaded into memory and ready for use.")

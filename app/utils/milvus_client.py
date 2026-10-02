# from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# # Connect to Milvus
# connections.connect(alias="default", host="127.0.0.1", port="19530")
# print("✅ Connected to Milvus")

# collection_name = "staff_collection"

# # Only create if it does not exist
# if not utility.has_collection(collection_name):
#     # Define schema
#     fields = [
#         FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#         FieldSchema(name="staff_id", dtype=DataType.INT64),
#         FieldSchema(name="staff_name", dtype=DataType.VARCHAR, max_length=100),
#         FieldSchema(name="face_vector", dtype=DataType.FLOAT_VECTOR, dim=128)
#     ]
#     schema = CollectionSchema(fields, description="Staff face collection")
    
#     # Create collection
#     staff_collection = Collection(name=collection_name, schema=schema)
#     print(f"✅ Collection '{collection_name}' created")

#     # Create index for vector search
#     index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
#     staff_collection.create_index(field_name="face_vector", index_params=index_params)
#     print("✅ Index created")
# else:
#     staff_collection = Collection(collection_name)
#     print(f"⚠️ Collection '{collection_name}' already exists")



# from pymilvus import connections, Collection

# # Connect to Milvus
# connections.connect("default", host="localhost", port="19530")

# collection_name = "staff_faces"
# collection = Collection(collection_name)

# # Make sure the collection is loaded
# collection.load()

# # Query all data; use output_fields to get needed fields (e.g., staff_id, staff_name)
# expr = ""  # empty expression fetches all vectors

# results = collection.query(expr=expr, output_fields=["staff_id", "staff_name"])

# print(f"Data in '{collection_name}' collection:")
# for row in results:
#     print(row)


# from pymilvus import connections, Collection, utility

# # 1️⃣ Connect to Milvus
# connections.connect("default", host="localhost", port="19530")

# # 2️⃣ Specify your collection name
# collection_name = "staff_faces"

# # 3️⃣ Check if collection exists
# if not utility.has_collection(collection_name):
#     print(f"❌ Collection '{collection_name}' does not exist.")
#     exit()

# # 4️⃣ Load collection
# collection = Collection(name=collection_name)

# # 5️⃣ Show schema details
# print(f"\n📘 Collection Name: {collection.name}")
# print(f"📄 Description: {collection.description}")
# print(f"🧱 Fields:")

# for field in collection.schema.fields:
#     print(f" - {field.name}: {field.dtype.name}, "
#           f"Primary={field.is_primary}, "
#           f"AutoID={field.auto_id}, "
#           f"Description={field.description or 'N/A'}")

# # 6️⃣ Show index info
# print("\n🔍 Indexes:")
# try:
#     index_info = collection.indexes
#     if not index_info:
#         print(" - No index created yet.")
#     else:
#         for idx in index_info:
#             print(f" - Field: {idx.field_name}")
#             print(f"   Index Type: {idx.params.get('index_type')}")
#             print(f"   Metric Type: {idx.params.get('metric_type')}")
#             print(f"   Params: {idx.params}")
# except Exception as e:
#     print("⚠️ Failed to retrieve index info:", e)


from pymilvus import connections, Collection, utility

connections.connect("default", host="localhost", port="19530")

collection_name = "staff_faces"

if not utility.has_collection(collection_name):
    print(f"❌ Collection '{collection_name}' does not exist.")
    exit()

collection = Collection(name=collection_name)

print(f"\n📘 Collection Name: {collection.name}")
print(f"📄 Description: {collection.description}")
print(f"🧱 Fields:")

for field in collection.schema.fields:
    if field.dtype.name == "FLOAT_VECTOR":
        print(f" - {field.name}: {field.dtype.name}, "
              f"Dimension={field.params.get('dim')}")
    else:
        print(f" - {field.name}: {field.dtype.name}, "
              f"Primary={field.is_primary}, AutoID={field.auto_id}")

# from pymilvus import connections, Collection
# from app.core.config import settings

# def delete_all_faces():
#     connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
    
#     collection = Collection(settings.MILVUS_COLLECTION)
    
#     # Delete all records
#     collection.delete("id >= 0")
#     collection.flush()
    
#     # Compact to remove deleted entities physically
#     collection.compact()
    
#     # Reload collection to refresh stats
#     collection.load()
    
#     print("🗑️ All data deleted successfully.")
#     print("✅ Total entities now:", collection.num_entities)

# if __name__ == "__main__":
#     delete_all_faces()

# from pymilvus import connections, Collection, utility
# from app.core.config import settings

# def delete_all_faces():
#     # Connect to Milvus
#     connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
#     print("✅ Connected to Milvus")

#     # Check if collection exists
#     if settings.MILVUS_COLLECTION in utility.list_collections():
#         collection = Collection(settings.MILVUS_COLLECTION)
        
#         # Delete all records
#         expr = "id >= 0"  # delete all rows
#         collection.delete(expr)
#         collection.flush()
#         collection.compact()
#         collection.load()

#         print(f"🗑️ All data in collection '{settings.MILVUS_COLLECTION}' deleted successfully.")
#         print("✅ Total entities now:", collection.num_entities)
#     else:
#         print(f"Collection '{settings.MILVUS_COLLECTION}' does not exist.")

# if __name__ == "__main__":
#     delete_all_faces()

# from pymilvus import connections, Collection

# connections.connect("default", host="localhost", port="19530")
# collection = Collection("staff_faces")

# # Delete all entities logically
# collection.delete(expr="id >= 0")

# # Flush to persist deletions
# collection.flush()

# # Compact to physically remove deleted entities
# collection.compact()

# # Reload collection
# collection.load()

# # Check live entities
# live_count = len(collection.query(expr="id >= 0"))

# print("🗑️ All data deleted successfully.")
# print("✅ Live entities now:", live_count)

# from pymilvus import connections, utility
# from app.core.config import settings

# # Connect to Milvus
# connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)

# collection_name = settings.MILVUS_COLLECTION

# # Check if collection exists
# if utility.has_collection(collection_name):
#     print(f"🗑️ Deleting collection '{collection_name}'...")
#     utility.drop_collection(collection_name)
#     print(f"✅ Collection '{collection_name}' deleted successfully!")
# else:
#     print(f"⚠️ Collection '{collection_name}' does not exist.")

from pymilvus import connections, Collection, utility, exceptions

connections.connect("default", host="localhost", port="19530")

collection_name = "staff_faces"

try:
    exists = utility.has_collection(collection_name)
except exceptions.MilvusException as e:
    if e.code == 4:  # Collection not found
        exists = False
    else:
        raise  # re-raise unexpected errors

if exists:
    collection = Collection(collection_name)
    collection.drop()
    print(f"Collection '{collection_name}' dropped successfully.")
else:
    print(f"Collection '{collection_name}' does not exist.")

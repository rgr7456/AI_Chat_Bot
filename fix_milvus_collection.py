#!/usr/bin/env python3
"""
Fix Milvus Collection Schema - Recreate collection with correct schema
"""

from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, 
    Collection, utility
)

def recreate_staff_collection():
    """Recreate the staff_faces collection with correct schema"""
    
    print("=== Recreating Milvus Collection ===")
    
    try:
        # Connect to Milvus
        connections.connect("default", host="localhost", port="19530")
        print("✅ Connected to Milvus")
        
        collection_name = "staff_faces"
        
        # Drop existing collection if it exists
        if utility.has_collection(collection_name):
            print(f"⚠️  Collection '{collection_name}' exists. Dropping...")
            utility.drop_collection(collection_name)
            print(f"✅ Collection '{collection_name}' dropped")
        
        # Define correct schema
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True  # Auto-generate primary keys
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
                dim=512
            )
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Staff face embeddings collection"
        )
        
        # Create new collection
        collection = Collection(name=collection_name, schema=schema)
        print(f"✅ Collection '{collection_name}' created with correct schema")
        
        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT", 
            "params": {"nlist": 128}
        }
        
        collection.create_index(field_name="embedding", index_params=index_params)
        print("✅ Index created")
        
        # Load collection
        collection.load()
        print("✅ Collection loaded and ready")
        
        # Display schema info
        print("\nCollection Schema:")
        for field in schema.fields:
            print(f"  - {field.name}: {field.dtype} (primary: {field.is_primary})")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to recreate collection: {e}")
        return False

if __name__ == "__main__":
    success = recreate_staff_collection()
    
    if success:
        print("\n🎉 Collection recreation successful!")
        print("You can now register staff members.")
    else:
        print("\n❌ Collection recreation failed!")
        print("Check Milvus server status and try again.")
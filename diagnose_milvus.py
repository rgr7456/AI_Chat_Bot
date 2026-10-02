#!/usr/bin/env python3
"""
Milvus Diagnostic Script - Check Milvus connection and collection status
"""

import traceback
from pymilvus import connections, Collection, utility, exceptions

def test_milvus_connection():
    """Test basic Milvus connection"""
    print("=== Testing Milvus Connection ===")
    
    try:
        # Test connection
        connections.connect("default", host="localhost", port="19530")
        print("✅ Connected to Milvus server")
        
        # Test server status
        server_version = utility.get_server_version()
        print(f"✅ Milvus server version: {server_version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return False

def test_collection_status():
    """Test collection existence and status"""
    print("\n=== Testing Collection Status ===")
    
    collection_name = "staff_faces"
    
    try:
        # Check if collection exists
        exists = utility.has_collection(collection_name)
        print(f"Collection '{collection_name}' exists: {exists}")
        
        if not exists:
            print("❌ Collection does not exist!")
            print("Solutions:")
            print("1. Run: python app/utils/create_staff_collection.py")
            print("2. Or create collection manually")
            return False
            
        # Try to get collection
        collection = Collection(collection_name)
        print("✅ Collection object created")
        
        # Check collection schema
        schema = collection.schema
        print(f"✅ Collection schema loaded, fields: {len(schema.fields)}")
        for field in schema.fields:
            print(f"   - {field.name}: {field.dtype} (primary: {field.is_primary})")
        
        # Check collection statistics
        num_entities = collection.num_entities
        print(f"✅ Number of entities in collection: {num_entities}")
        
        # Check if collection is loaded
        load_state = utility.load_state(collection_name)
        print(f"Collection load state: {load_state}")
        
        return True
        
    except Exception as e:
        print(f"❌ Collection test failed: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return False

def test_collection_operations():
    """Test basic collection operations"""
    print("\n=== Testing Collection Operations ===")
    
    collection_name = "staff_faces"
    
    try:
        collection = Collection(collection_name)
        
        # Test index status
        has_index = collection.has_index()
        print(f"Collection has index: {has_index}")
        
        if not has_index:
            print("⚠️  No index found - this may cause search issues")
            print("Creating index...")
            
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE", 
                "params": {"nlist": 128}
            }
            
            collection.create_index(field_name="embedding", index_params=index_params)
            print("✅ Index created")
        
        # Test loading collection
        collection.load()
        print("✅ Collection loaded successfully")
        
        # Test simple query (if data exists)
        try:
            rows = collection.query(expr="", output_fields=["staff_id"], limit=5)
            print(f"✅ Query test successful, found {len(rows)} records")
            if rows:
                print(f"Sample staff_id: {rows[0].get('staff_id', 'N/A')}")
        except Exception as e:
            print(f"⚠️  Query test failed: {str(e)}")
            
        return True
        
    except Exception as e:
        print(f"❌ Collection operations failed: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return False

def test_embedding_operations():
    """Test embedding search operations"""
    print("\n=== Testing Embedding Operations ===")
    
    collection_name = "staff_faces"
    
    try:
        collection = Collection(collection_name)
        
        # Create dummy embedding for search test
        import numpy as np
        dummy_embedding = np.random.rand(512).astype(np.float32).tolist()
        
        # Test search
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}
        
        results = collection.search(
            data=[dummy_embedding],
            anns_field="embedding",
            param=search_params,
            limit=5,
            output_fields=["staff_id", "staff_name"],
        )
        
        print(f"✅ Search test successful, found {len(results[0])} results")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding operations failed: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return False

def show_solutions():
    """Show common solutions for Milvus issues"""
    print("\n=== Common Milvus Issues & Solutions ===")
    
    print("\n1. Connection Issues:")
    print("   - Check if Milvus server is running: docker ps")
    print("   - Restart Milvus: docker-compose restart")
    print("   - Check port 19530 is accessible")
    
    print("\n2. Collection Not Found:")
    print("   - Run: python app/utils/create_staff_collection.py")
    print("   - Or create manually via Milvus admin")
    
    print("\n3. Collection Not Loaded:")
    print("   - Collection must be loaded before search")
    print("   - Call collection.load() before operations")
    
    print("\n4. Index Issues:")
    print("   - Collection needs index for efficient search")
    print("   - Create index with collection.create_index()")
    
    print("\n5. Schema Mismatch:")
    print("   - Check field names and dimensions match")
    print("   - Embedding dimension should be 512")

if __name__ == "__main__":
    print("Milvus Diagnostic Tool")
    print("=====================")
    
    # Run all tests
    connection_ok = test_milvus_connection()
    
    if connection_ok:
        collection_ok = test_collection_status()
        
        if collection_ok:
            ops_ok = test_collection_operations()
            
            if ops_ok:
                embedding_ok = test_embedding_operations()
                
                if embedding_ok:
                    print("\n🎉 All Milvus tests passed!")
                else:
                    print("\n❌ Embedding operations failed")
            else:
                print("\n❌ Collection operations failed")
        else:
            print("\n❌ Collection status check failed")
    else:
        print("\n❌ Connection failed")
    
    # Show solutions regardless
    show_solutions()
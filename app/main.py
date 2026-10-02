# from contextlib import asynccontextmanager
# from fastapi import FastAPI
# from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, exceptions

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     connections.connect("default", host="localhost", port="19530")
#     collection_name = "staff_faces"
#     fields = [
#         FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#         FieldSchema(name="staff_id", dtype=DataType.VARCHAR, max_length=50),
#         FieldSchema(name="staff_name", dtype=DataType.VARCHAR, max_length=100),
#         FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
#     ]
#     schema = CollectionSchema(fields, description="Staff face embeddings")

#     try:
#         exists = utility.has_collection(collection_name)
#     except exceptions.MilvusException as err:
#         if err.code == 4:
#             exists = False
#         else:
#             raise

#     if not exists:
#         try:
#             # Create collection by instantiation with schema
#             collection = Collection(name=collection_name, schema=schema)
#             print(f"✅ Collection '{collection_name}' created.")
#             # Create index now
#             collection.create_index(
#                 field_name="embedding",
#                 index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
#             )
#             print(f"✅ Index created for collection '{collection_name}'.")
#         except Exception as e:
#             print(f"Error creating collection or index: {e}")
#             raise
#     else:
#         collection = Collection(name=collection_name)
#         print(f"ℹ️ Collection '{collection_name}' already exists.")

#     try:
#         collection.load()
#         print(f"✅ Collection '{collection_name}' loaded.")
#     except Exception as e:
#         print(f"Error loading collection: {e}")
#         raise

#     yield  # shutdown logic here, if needed

# app = FastAPI(lifespan=lifespan)




from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.controller.face_routes import router as face_router

app = FastAPI(title="Face Authentication API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - use this for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# For production, use specific origins instead of "*":
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:3000",  # React default
#         "http://localhost:5173",  # Vite default
#         "http://127.0.0.1:3000",
#         "http://127.0.0.1:5173",
#         "https://yourdomain.com",  # Your production domain
#     ],
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "PUT", "DELETE"],
#     allow_headers=["*"],
# )

# Include the face auth routes
app.include_router(face_router, prefix="/face", tags=["Face Authentication"])

# Optional root route
@app.get("/")
async def root():
    return {"message": "Welcome to the Face Authentication API"}

@app.get("/cors-test")
async def cors_test():
    """Simple endpoint to test CORS is working"""
    return {
        "message": "CORS is working!",
        "cors_enabled": True,
        "timestamp": "2025-10-24"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

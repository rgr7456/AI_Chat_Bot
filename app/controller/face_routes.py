from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from typing import List
from app.dto.face_dto import StaffRegisterDTO
from app.serviceImpl.face_service_impl import register_staff, verify_staff
from starlette.concurrency import run_in_threadpool
from app.constants.constants import DEFAULT_DISTANCE_THRESHOLD
from app.repository.face_repository import collection
from app.repository.face_repository import (
    prepare_collection,
    search_embedding,
    fetch_embeddings_for_staff,
    insert_staff_embedding,
    get_collection,
    get_milvus_connection,
)
from fastapi import Query as FastAPIQuery
import traceback
from typing import Any, Dict
from pymilvus import utility, DataType

from pymilvus import (
    CollectionSchema,
    FieldSchema,
    DataType,
    Collection,
    utility
)


router = APIRouter()

@router.get("/debug/test")
async def debug_test():
    """Simple test endpoint to verify API is working"""
    return {"status": "ok", "message": "API is working"}

@router.post("/debug/echo")
async def debug_echo(
    test_param: str = Form(...),
    test_file: UploadFile = File(...)
):
    """Debug endpoint to test form data and file uploads"""
    try:
        file_info = {
            "filename": test_file.filename,
            "content_type": test_file.content_type,
            "size": len(await test_file.read())
        }
        return {
            "test_param": test_param,
            "file_info": file_info,
            "message": "Form data received successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")

@router.post("/register")
async def register(
    staff_id: str = Form(...),
    staff_name: str = Form(...),
    image: List[UploadFile] = File(...),
):
    # Support sending one or multiple files. FastAPI will provide a list
    try:
        images_bytes = []
        for f in image:
            images_bytes.append(await f.read())

        # run CPU/GPU heavy embedding extraction in threadpool
        # register_staff accepts either bytes or list[bytes]
        await run_in_threadpool(register_staff, staff_id, staff_name, images_bytes)
        return {"message": "Staff registered successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/verify")
async def verify(image: UploadFile = File(...), threshold: float = Query(0.2)):
    image_bytes = await image.read()
    # run verify in threadpool to avoid blocking event loop
    result = await run_in_threadpool(verify_staff, image_bytes, threshold)
    if not result:
        raise HTTPException(status_code=404, detail="Staff not found")
    return result

# python .\tools\verify_webcam.py --url http://127.0.0.1:8000/face/verify          


@router.get("/milvus/view_all")
async def view_all_milvus(
    full: bool = FastAPIQuery(False, description="If true include full embeddings"),
    limit: int = FastAPIQuery(100, description="Maximum rows to return (use 0 for no limit, careful)"),
    offset: int = FastAPIQuery(0, description="Row offset for pagination"),
):
    """Return all rows from the Milvus collection as JSON.

    By default embeddings are truncated to the first 16 floats to keep payload small;
    set `full=true` to include the full vector (may be large).
    """
    try:
        # use Collection.query to fetch all rows; output_fields controls what we get
        expr = ""  # empty expr returns all rows in pymilvus query
        try:
            rows = collection.query(expr, output_fields=["_id", "staff_id", "staff_name", "embedding"], limit=limit, offset=offset)  # _id available in some Milvus versions
        except TypeError:
            # older pymilvus versions may have different param order - try positional expr then kwargs
            try:
                rows = collection.query(expr, ["id", "staff_id", "staff_name", "embedding"], limit, offset)
            except Exception:
                # fallback: try without _id
                rows = collection.query(expr, output_fields=["staff_id", "staff_name", "embedding"], limit=limit, offset=offset)
        except Exception:
            # fallback: try without _id
            rows = collection.query(expr, output_fields=["staff_id", "staff_name", "embedding"], limit=limit, offset=offset)

        out = []
        for r in rows:
            item: Dict[str, Any] = {
                "staff_id": r.get("staff_id"),
                "staff_name": r.get("staff_name"),
            }
            emb = r.get("embedding")
            if emb is None:
                item["embedding"] = None
            else:
                if full:
                    item["embedding"] = [float(x) for x in emb]
                else:
                    # return first 16 dimensions as a quick preview
                    item["embedding_preview"] = [float(x) for x in emb[:16]]
                    item["embedding_len"] = len(emb)
            out.append(item)

        return {"count": len(out), "rows": out}
    except Exception:
        tb = traceback.format_exc()
        # Return a helpful HTTP error with the traceback so it's visible in Swagger/clients for debugging
        raise HTTPException(status_code=500, detail={"error": "milvus_query_failed", "trace": tb})


@router.get("/milvus/collections")
async def list_milvus_collections():
    """List available collections on the Milvus server."""
    try:
        cols = utility.list_collections()
        return {"collections": cols}
    except Exception:
        raise HTTPException(status_code=500, detail={"error": "list_collections_failed", "trace": traceback.format_exc()})


@router.post("/milvus/prepare")
async def milvus_prepare():
    """Create index if missing and load the collection into memory."""
    try:
        await run_in_threadpool(prepare_collection)
        return {"message": "collection prepared"}
    except Exception:
        raise HTTPException(status_code=500, detail={"error": "prepare_failed", "trace": traceback.format_exc()})


@router.post("/milvus/search_image")
async def milvus_search_image(image: UploadFile = File(...), top_k: int = Query(5)):
    """Search the collection with an uploaded image and return hits."""
    image_bytes = await image.read()
    try:
        # extract embedding in threadpool then run search
        def work():
            from app.utils.face_utils import read_image_bytes, extract_face_embedding
            img = read_image_bytes(image_bytes)
            emb = extract_face_embedding(img)
            results = search_embedding(emb, top_k=top_k)
            out = []
            for h in results:
                out.append({
                    "staff_id": h.entity.get("staff_id"),
                    "staff_name": h.entity.get("staff_name"),
                    "distance": float(h.distance),
                })
            return out

        hits = await run_in_threadpool(work)
        return {"count": len(hits), "hits": hits}
    except Exception:
        raise HTTPException(status_code=500, detail={"error": "search_failed", "trace": traceback.format_exc()})


@router.get("/milvus/fetch_embeddings/{staff_id}")
async def milvus_fetch_embeddings(staff_id: str):
    """Fetch stored embeddings for a staff_id (returns preview and count)."""
    try:
        embs = await run_in_threadpool(fetch_embeddings_for_staff, staff_id)
        out = []
        for e in embs:
            # e is a numpy array
            out.append({"embedding_preview": e.tolist()[:16], "len": e.size})
        return {"staff_id": staff_id, "count": len(out), "embeddings": out}
    except Exception:
        raise HTTPException(status_code=500, detail={"error": "fetch_failed", "trace": traceback.format_exc()})


@router.delete("/milvus/delete_staff/{staff_id}")
async def milvus_delete_staff(staff_id: str):
    """Delete all vectors for a staff_id. This is destructive — use with caution."""
    try:
        # Get collection and inspect schema to determine the field name and dtype for staff id
        collection = get_collection()
        field_name = None
        field_dtype = None
        try:
            schema_fields = collection.schema.fields
            for f in schema_fields:
                if f.name in ("staff_id", "id", "_id"):
                    field_name = f.name
                    field_dtype = getattr(f, "dtype", None)
                    break
        except Exception:
            field_name = None
            field_dtype = None

        expr = None
        if field_name is not None:
            # Build expression according to dtype
            try:
                if field_dtype in (DataType.INT64, DataType.INT32):
                    int_id = int(staff_id)
                    expr = f"{field_name} == {int_id}"
                else:
                    safe = staff_id.replace('"', '\\"')
                    expr = f'{field_name} == "{safe}"'
            except Exception:
                # fallback to string match
                safe = staff_id.replace('"', '\\"')
                expr = f'{field_name} == "{safe}"'

        # If schema inspection failed or didn't find a candidate, try previous heuristics
        if expr is None:
            tried = []
            matching_expr = None
            for candidate in ("numeric", "string"):
                if candidate == "numeric":
                    try:
                        int_id = int(staff_id)
                        expr_try = f"staff_id == {int_id}"
                    except Exception:
                        continue
                else:
                    safe = staff_id.replace('"', '\\"')
                    expr_try = f'staff_id == "{safe}"'

                tried.append(expr_try)
                try:
                    rows = collection.query(expr_try, output_fields=["staff_id", "staff_name"], limit=1)
                except Exception:
                    rows = []

                if rows and len(rows) > 0:
                    matching_expr = expr_try
                    break

            if matching_expr is None:
                return HTTPException(status_code=404, detail={"error": "not_found", "tried": tried})

            expr = matching_expr

        # perform delete using the expr  
        res = collection.delete(expr)
        collection.flush()

        # Diagnostic: return type and repr so we can see what pymilvus returned
        res_type = type(res).__name__
        try:
            res_repr = repr(res)
        except Exception:
            res_repr = str(res)

        # Try common attributes/keys for deleted count
        deleted_count = None
        try:
            if hasattr(res, "deleted_count"):
                deleted_count = getattr(res, "deleted_count")
            elif hasattr(res, "count"):
                deleted_count = getattr(res, "count")
            else:
                try:
                    deleted_count = res.get("deleted_count") if isinstance(res, dict) else None
                except Exception:
                    deleted_count = None
        except Exception:
            deleted_count = None

        return {"expr": expr, "deleted_count": deleted_count, "result_type": res_type, "result_repr": res_repr}
    except Exception:
        raise HTTPException(status_code=500, detail={"error": "delete_failed", "trace": traceback.format_exc()})


@router.get("/milvus/staff_list")
async def milvus_staff_list(limit: int = FastAPIQuery(1000, description="Maximum rows to scan for staff list (use higher to capture more unique staff)"), offset: int = FastAPIQuery(0, description="Offset for scan")):
    """Return a deduplicated list of (staff_id, staff_name) from the Milvus collection.

    This scans up to `limit` rows starting at `offset` and returns unique pairs.
    """
    try:
        collection = get_collection()
        def work(lim, off):
            expr = ""
            try:
                rows = collection.query(expr, output_fields=["staff_id", "staff_name"], limit=lim, offset=off)
            except TypeError:
                # try older signature ordering
                rows = collection.query(expr, ["staff_id", "staff_name"], lim, off)
            out = []
            seen = set()
            for r in rows:
                sid = r.get("staff_id")
                sname = r.get("staff_name")
                if sid is None:
                    continue
                if sid in seen:
                    continue
                seen.add(sid)
                out.append({"staff_id": sid, "staff_name": sname})
            return out

        staff = await run_in_threadpool(work, limit, offset)
        return {"count": len(staff), "staff": staff}
    except Exception:
        raise HTTPException(status_code=500, detail={"error": "staff_list_failed", "trace": traceback.format_exc()})
    


    # -------------------------------------------------------------------
# ✅ 1. CREATE SCHEMA / COLLECTION
@router.post("/milvus/create_schema")
async def create_milvus_schema(collection_name: str = Form("staff_faces")):
    """
    Create a new Milvus collection schema for storing staff face embeddings.
    Includes an auto-increment primary key required by Milvus.
    """

    try:  
        # Drop existing collection (optional safety)
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        # Define fields for schema
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="Auto-generated primary key"
            ),
            FieldSchema(
                name="staff_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="Unique staff ID"
            ),
            FieldSchema(
                name="staff_name",
                dtype=DataType.VARCHAR,
                max_length=255,
                description="Staff name"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=512,  # 512-dimensional vector (FaceNet output)
                description="Face embedding vector"
            ),
        ]

        schema = CollectionSchema(fields=fields, description="Face embeddings of staff for verification")

        # Create the collection
        collection = Collection(name=collection_name, schema=schema)

        # Create index on the vector field
        index_params = {
            "metric_type": "L2",  # or "COSINE"
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }

        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()

        return {
            "message": f"Collection '{collection_name}' created successfully",
            "fields": [f.name for f in fields],
            "index": index_params,
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail={"error": "create_schema_failed", "trace": traceback.format_exc()},
        )

# -------------------------------------------------------------------
# ✅ 2. DELETE COLLECTION
# -------------------------------------------------------------------
@router.delete("/milvus/delete_collection/{collection_name}")
async def delete_milvus_collection(collection_name: str):
    """
    Delete the entire Milvus collection and its data.
    """
    try:
        if not utility.has_collection(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

        utility.drop_collection(collection_name)
        return {"message": f"Collection '{collection_name}' deleted successfully"}

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail={"error": "delete_collection_failed", "trace": traceback.format_exc()},
        )

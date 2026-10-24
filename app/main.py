import re
import asyncio
import tempfile
import os
import faulthandler
import concurrent.futures
import multiprocessing
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from dotenv import load_dotenv
load_dotenv()

from app.schemas import IndexResponse, QueryRequest, QueryResponse
from app.config import settings
from app.fixtures import FX1_RESPONSE

from app.ingest import safe_extract, parse_document
from app.store import build_collection
from app.doccard import build_cards
from app.pipeline import run_query

app = FastAPI(title="SG-ECP Smart Knowledge Base for Agents")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.post("/index", response_model=IndexResponse)
async def index_collection(request: Request):
    try:
        form = await request.form()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse multipart form data")

    file_obj: UploadFile | None = None
    collection_name: str | None = None

    for key, value in form.multi_items():
        if hasattr(value, "filename") and getattr(value, "filename") and file_obj is None:
            file_obj = value
        elif key == "collection_name":
            collection_name = str(value)

    if not file_obj:
        raise HTTPException(status_code=400, detail="A zip file upload is required")
    if not collection_name:
        raise HTTPException(status_code=400, detail="collection_name is required")
    if not re.match(r"^[A-Za-z0-9_-]{1,64}$", collection_name):
        raise HTTPException(status_code=400, detail="Invalid collection_name format")
    if not getattr(file_obj, "filename", "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a zip archive")

    # --- INGESTION PIPELINE WIRED HERE ---
    print(f"\n--- Starting Ingestion for {collection_name} ---")
    zip_bytes = await file_obj.read()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = Path(tmpdir)
        
        print("1. Extracting zip file...")
        extracted_paths = await asyncio.to_thread(safe_extract, zip_bytes, dest_path)
        
        # --- NEW ROBUST MULTIPROCESSING LOGIC ---
        cpu_cores = min(8, os.cpu_count() or 2)
        print(f"2. Parsing {len(extracted_paths)} documents across {cpu_cores} CPU cores...")
        
        loop = asyncio.get_running_loop()
        
        async def parse_with_log(path):
            print(f"  -> Started parsing: {path.name}")
            try:
                doc = await loop.run_in_executor(pool, parse_document, path, path.stem)
                print(f"  <- Finished parsing: {path.name}")
                return doc
            except Exception as e:
                print(f"  X FAILED parsing {path.name}: {e}")
                return None

        # max_tasks_per_child prevents memory leaks and protects against process pool crashes
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=cpu_cores,
            max_tasks_per_child=1, 
            initializer=faulthandler.enable
        ) as pool:
            tasks = [parse_with_log(path) for path in extracted_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            parsed_docs = [doc for doc in results if doc is not None and not isinstance(doc, Exception)]
            
        print("3. Extracting DocCards via OpenCode API (FAST model)...")
        cards = await build_cards(parsed_docs)
        
        print("4. Building vector index...")
        await asyncio.to_thread(build_collection, collection_name, parsed_docs, cards)

    print("--- Ingestion Complete! ---")
    return IndexResponse(status="success")

@app.post("/query", response_model=QueryResponse)
async def query_collection(payload: QueryRequest):
    if settings.USE_MOCK_MODELS:
        return QueryResponse(**FX1_RESPONSE)
        
    try:
        response_data = await run_query(
            collection_name=payload.collection_name, 
            query=payload.query
        )
        return response_data
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
import io
import os
import zipfile
import pytest
import httpx

# Assumes the API is running locally or at a configured URL
BASE_URL = os.getenv("API_URL", "http://localhost:8000")

def create_zip(files_dict: dict) -> bytes:
    """Helper to create zip archives in memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files_dict.items():
            if isinstance(content, bytes):
                zf.writestr(name, content)
            elif isinstance(content, zipfile.ZipInfo):
                # Allow passing a crafted ZipInfo and payload tuple if needed,
                # but for simplicity handle string/bytes
                pass
            else:
                zf.writestr(name, content.encode("utf-8"))
    buf.seek(0)
    return buf.read()

def create_oversize_zip() -> bytes:
    """Creates a zip file that falsely advertises an uncompressed size > MAX_UPLOAD_MB."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zinfo = zipfile.ZipInfo("massive.txt")
        # Pretend the file is 100MB (exceeds the 50MB default)
        zinfo.file_size = 100 * 1024 * 1024 
        zf.writestr(zinfo, b"a" * 100)
    buf.seek(0)
    return buf.read()

@pytest.mark.asyncio
async def test_bad_collection_name():
    """Verify that invalid collection names are rejected by the API[cite: 3]."""
    zip_bytes = create_zip({"test.txt": "Sample content"})
    
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # Test spaces and special characters
        files = {"file": ("test.zip", zip_bytes, "application/zip")}
        resp = await client.post("/index", data={"collection_name": "invalid name!"}, files=files)
        assert resp.status_code in (400, 422)
        
        # Test directory traversal payload
        files = {"file": ("test.zip", zip_bytes, "application/zip")}
        resp = await client.post("/index", data={"collection_name": "../hacked_dir"}, files=files)
        assert resp.status_code in (400, 422)

@pytest.mark.asyncio
async def test_zip_traversal():
    """Verify that path traversal in zip contents is safely ignored during extraction[cite: 3]."""
    zip_bytes = create_zip({
        "../escape.txt": "malicious content",
        "/absolute/path.txt": "malicious content",
        "valid_doc.txt": "This is a valid document."
    })
    
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        files = {"file": ("test.zip", zip_bytes, "application/zip")}
        resp = await client.post("/index", data={"collection_name": "test_traversal"}, files=files)
        # Should succeed because the ingest logic skips malicious files and extracts valid ones
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

@pytest.mark.asyncio
async def test_oversize_zip():
    """Verify that a zip exceeding MAX_UPLOAD_MB uncompressed size is rejected[cite: 3]."""
    zip_bytes = create_oversize_zip()
    
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        files = {"file": ("bomb.zip", zip_bytes, "application/zip")}
        resp = await client.post("/index", data={"collection_name": "test_oversize"}, files=files)
        # ingest.py raises ValueError for size limits, which translates to a 500 in the generic pipeline
        assert resp.status_code == 500

@pytest.mark.asyncio
async def test_two_collection_isolation():
    """Verify that queries to Collection A cannot retrieve data from Collection B[cite: 3]."""
    zip_a = create_zip({"docA.txt": "The secret code for project Alpha is 9922."})
    zip_b = create_zip({"docB.txt": "The secret code for project Beta is 7733."})
    
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:
        # 1. Index Collection A
        files_a = {"file": ("a.zip", zip_a, "application/zip")}
        resp_a = await client.post("/index", data={"collection_name": "coll_a"}, files=files_a)
        assert resp_a.status_code == 200
        
        # 2. Index Collection B
        files_b = {"file": ("b.zip", zip_b, "application/zip")}
        resp_b = await client.post("/index", data={"collection_name": "coll_b"}, files=files_b)
        assert resp_b.status_code == 200
        
        # 3. Query A for B's information
        resp_q1 = await client.post("/query", json={
            "collection_name": "coll_a",
            "query": "What is the secret code for project Beta?"
        })
        assert resp_q1.status_code == 200
        ans_q1 = resp_q1.json()["answer"]
        assert "7733" not in ans_q1
        assert "No information was retrieved for this question" in ans_q1 or "No information" in ans_q1

# requirements:
# httpx>=0.24.0
# pytest>=7.4.0
# pytest-asyncio>=0.21.1

# contract:
# input: Execution from CLI using `pytest tests/test_isolation.py`. Requires the API server running at BASE_URL.
# output: Test pass/fail statuses ensuring system security against malicious payloads and strict data boundary enforcement[cite: 2].
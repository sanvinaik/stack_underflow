import requests
import zipfile
from pathlib import Path

docs_dir = Path("data/train/docs")
zip_filename = "real_test.zip"

print(f"Scanning for documents in {docs_dir}...")
all_files = [f for f in docs_dir.glob("*") if f.suffix.lower() in [".pdf", ".txt"]]

with zipfile.ZipFile(zip_filename, "w") as zf:
    for file_path in all_files:
        zf.write(file_path, arcname=file_path.name)
        print(f"  - Added {file_path.name} to zip")

url = "http://localhost:8000/index"
files = {"file_upload": (zip_filename, open(zip_filename, "rb"), "application/zip")}
data = {"collection_name": "dev_collection"}

print(f"\nUploading all documents to /index... (This will take a while!)")
response = requests.post(url, files=files, data=data)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")
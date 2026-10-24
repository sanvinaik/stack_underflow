import requests
import json

url = "http://localhost:8000/query"

# We are querying the exact collection we just created
payload = {
    "collection_name": "dev_collection",
    "query": "What are the key highlights or topics discussed in the Implementation Handbook?"
}

print(f"Sending query to {url}...\n")
response = requests.post(url, json=payload)

if response.status_code == 200:
    data = response.json()
    
    print("=== FINAL ANSWER ===")
    print(data.get("answer", "No answer returned."))
    
    print("\n=== CITED CHUNKS ===")
    for chunk in data.get("chunks", []):
        doc_name = chunk.get("metadata", {}).get("document", "Unknown Doc")
        print(f" - {chunk.get('chunk_id')} (from {doc_name})")
        
    print("\n=== PIPELINE TRACE (8 Steps) ===")
    for step in data.get("trace", []):
        duration = step.get('output', {}).get('duration_ms', 0)
        print(f"[{step.get('step_name')}] ({duration:.0f}ms) -> {step.get('description')}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
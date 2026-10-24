import fitz  # PyMuPDF
import json
import re
import statistics
from pathlib import Path
from collections import defaultdict

def analyze_benchmark_chunks(benchmark_path: Path) -> dict[str, list[dict]]:
    """Reads the benchmark JSONL and computes gold chunk-size statistics."""
    if not benchmark_path.exists():
        print(f"Benchmark file not found at {benchmark_path}")
        return {}

    chunk_lengths = []
    chunks_by_doc = defaultdict(list)
    
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Assuming schema based on benchmark_100.jsonl notes: data has "expected_chunks" or similar, 
            # or data itself is a query with a "chunks" array.
            # Using Addendum E8 as proxy for the chunk shape in the benchmark:
            for chunk in data.get("chunks", []):
                content = chunk.get("content", "")
                length = len(content)
                chunk_lengths.append(length)
                
                doc_name = chunk.get("metadata", {}).get("document", "unknown")
                page_nums = chunk.get("metadata", {}).get("page_numbers", [])
                
                chunks_by_doc[doc_name].append({
                    "content": content,
                    "length": length,
                    "pages": page_nums
                })
                
    if chunk_lengths:
        chunk_lengths.sort()
        print("=== Gold Chunk Stats ===")
        print(f"Count : {len(chunk_lengths)}")
        print(f"Min   : {min(chunk_lengths)} chars")
        print(f"Max   : {max(chunk_lengths)} chars")
        print(f"Median: {statistics.median(chunk_lengths):.1f} chars")
        p90_idx = int(len(chunk_lengths) * 0.90)
        print(f"P90   : {chunk_lengths[p90_idx]} chars")
        print("========================\n")
        
    return chunks_by_doc

def check_word_gluing(text: str) -> list[str]:
    """Finds samples of potentially glued words (e.g. lowercase followed immediately by uppercase)."""
    # Look for patterns like "wordAnother" or "theCompany" which indicates missing spaces
    glued_pattern = re.compile(r'[a-z]{3,}[A-Z][a-z]{2,}')
    return glued_pattern.findall(text)

def probe_pdfs(pdf_dir: Path, chunks_by_doc: dict[str, list[dict]]):
    """Evaluates real PDFs for OCR needs, word gluing, and page number conventions."""
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        print(f"PDF directory not found at {pdf_dir}")
        return

    for pdf_path in pdf_dir.glob("*.pdf"):
        print(f"--- Probing {pdf_path.name} ---")
        doc_chunks = chunks_by_doc.get(pdf_path.name, [])
        if not doc_chunks:
            print(f"No gold chunks found in benchmark for {pdf_path.name}")
        
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Failed to open {pdf_path}: {e}")
            continue
            
        low_char_pages = []
        glued_samples = []
        found_chunks = 0
        
        # Test up to 30 chunks for page convention verification
        chunks_to_test = doc_chunks[:30]
        
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text("text")
            char_count = len(text.strip())
            
            if char_count < 50:
                low_char_pages.append(page_index + 1)
                
            glued = check_word_gluing(text)
            if glued and len(glued_samples) < 5:
                glued_samples.extend(glued[:5 - len(glued_samples)])
                
            # Verify page convention for the subset of chunks
            for chunk in chunks_to_test:
                # normalize strings for loose matching
                norm_chunk = re.sub(r'\s+', '', chunk["content"].lower())
                norm_page = re.sub(r'\s+', '', text.lower())
                
                if norm_chunk in norm_page:
                    expected_pages = chunk["pages"]
                    actual_page = page_index + 1
                    if actual_page in expected_pages:
                        found_chunks += 1
                        # Remove to avoid double counting
                        chunks_to_test.remove(chunk) 

        print(f"Pages < 50 chars: {len(low_char_pages)} {low_char_pages[:10]}{'...' if len(low_char_pages)>10 else ''}")
        print(f"Glued word samples: {glued_samples}")
        
        if doc_chunks:
            tested_count = min(30, len(doc_chunks))
            print(f"Page convention match (page.number+1): {found_chunks}/{tested_count} gold chunks located precisely.")
        print()
        doc.close()

if __name__ == "__main__":
    # Assuming script is run from project root, e.g., `python -m eval.probe_data`
    data_dir = Path("data")
    benchmark_file = data_dir / "benchmark_100.jsonl"
    pdf_directory = data_dir / "pdfs" # Expecting real PDFs to be placed here for testing
    
    docs_dict = analyze_benchmark_chunks(benchmark_file)
    if docs_dict:
        probe_pdfs(pdf_directory, docs_dict)

# requirements:
# pymupdf>=1.23.0

# contract:
# input: Execution from CLI. Expects benchmark JSONL and raw PDFs in the target data directory.
# output: Console prints reporting gold chunk length stats, page extraction char counts, word-gluing anomalies, and page integer mapping correctness.
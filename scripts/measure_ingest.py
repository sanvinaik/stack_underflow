import asyncio
import os
import sys
import time
import math
from pathlib import Path

# Ensure the app module can be found if script is run from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ingest import parse_document
from app.doccard import build_cards
from app.store import _get_embed_model
from app.config import settings

async def main():
    pdf_dir = Path(settings.DATA_DIR) / "pdfs"
    
    # Check if we have documents to test
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        print(f"Directory {pdf_dir} not found. Please create it and add some PDF/TXT files.")
        return

    doc_paths = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.txt"))
    if not doc_paths:
        print(f"No documents found in {pdf_dir}. Please add at least 1 document.")
        return

    # Use up to 5 documents for the sample
    sample_paths = doc_paths[:5]
    num_sampled = len(sample_paths)
    print(f"--- Measuring ingestion on {num_sampled} sample document(s) ---")

    # 1. Measure Parse Stage
    print("\n1. Parsing documents...")
    parsed_docs = []
    total_pages = 0
    total_chunks = 0
    
    t0_parse = time.perf_counter()
    for i, path in enumerate(sample_paths):
        doc_id = f"doc_{i}"
        pdoc = parse_document(path, doc_id)
        parsed_docs.append(pdoc)
        total_pages += len(pdoc.pages)
        total_chunks += len(pdoc.chunks)
        
    parse_time = time.perf_counter() - t0_parse
    parse_rate = parse_time / total_pages if total_pages > 0 else 0
    print(f"   Parsed {total_pages} pages into {total_chunks} chunks.")
    print(f"   Parse time: {parse_time:.2f}s (Rate: {parse_rate:.4f}s per page)")

    # 2. Measure Embed Stage
    print("\n2. Embedding chunks...")
    model = _get_embed_model()
    chunk_texts = [c.content for d in parsed_docs for c in d.chunks]
    
    t0_embed = time.perf_counter()
    if chunk_texts:
        _ = model.encode(chunk_texts, convert_to_numpy=True)
    embed_time = time.perf_counter() - t0_embed
    embed_rate = embed_time / total_chunks if total_chunks > 0 else 0
    print(f"   Embedded {total_chunks} chunks.")
    print(f"   Embed time: {embed_time:.2f}s (Rate: {embed_rate:.4f}s per chunk)")

    # 3. Measure DocCard Extraction Stage
    print("\n3. Extracting DocCards (Parallel Batch)...")
    # This runs as a single batch since semaphore is 8 and we have at most 5 docs
    t0_card = time.perf_counter()
    _ = await build_cards(parsed_docs)
    card_latency = time.perf_counter() - t0_card
    print(f"   Extracted cards for {num_sampled} docs.")
    print(f"   Card batch latency (up to 8 docs): {card_latency:.2f}s")

    # 4. Extrapolate to Corpus Size
    # Defaulting to 100 docs for the estimate, but it can be configured.
    TARGET_CORPUS_DOCS = 100
    avg_pages_per_doc = total_pages / num_sampled if num_sampled > 0 else 0
    avg_chunks_per_doc = total_chunks / num_sampled if num_sampled > 0 else 0
    
    target_pages = TARGET_CORPUS_DOCS * avg_pages_per_doc
    target_chunks = TARGET_CORPUS_DOCS * avg_chunks_per_doc
    batches = math.ceil(TARGET_CORPUS_DOCS / 8)
    
    est_parse_time = parse_rate * target_pages
    est_embed_time = embed_rate * target_chunks
    est_card_time = batches * card_latency
    
    total_est_time = est_parse_time + est_embed_time + est_card_time
    buffered_est_time = total_est_time * 1.5  # 50% buffer
    
    print(f"\n--- Extrapolation for Corpus of {TARGET_CORPUS_DOCS} Documents ---")
    print(f"   Assumed Averages: {avg_pages_per_doc:.1f} pages/doc, {avg_chunks_per_doc:.1f} chunks/doc")
    print(f"   Est. Parse Time : {est_parse_time:.2f}s")
    print(f"   Est. Embed Time : {est_embed_time:.2f}s")
    print(f"   Est. Card Time  : {est_card_time:.2f}s ({batches} batches)")
    print(f"   ----------------------------------")
    print(f"   Base Estimate   : {total_est_time:.2f}s ({total_est_time/60:.2f} minutes)")
    print(f"   Stated Time (50% buffer): {buffered_est_time:.2f}s ({buffered_est_time/60:.2f} minutes)")

if __name__ == "__main__":
    asyncio.run(main())

# requirements:
# sentence-transformers>=2.2.2
# pymupdf>=1.23.0

# contract:
# input: Execution from CLI. Expects up to 5 documents in DATA_DIR/pdfs.
# output: Console prints detailing timing for parse, embed, and card stages, and an extrapolated ingestion time with a 50% buffer[cite: 3].
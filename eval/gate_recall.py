import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure the app module can be found if script is run from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.schemas import EvidenceContract
from app.config import settings
from app.store import load_collection
from app.compile import compile_query
from app.gate import apply_gate
from app.siblings import resolve_siblings
from app.retrieve import retrieve_candidates
from app.pipeline import _build_facets

CACHE_FILE = Path(".compile_cache.json")

def load_cache() -> dict[str, dict]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(cache: dict[str, dict]):
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")

async def main():
    data_dir = Path(settings.DATA_DIR)
    benchmark_file = data_dir / "benchmark_100.jsonl"
    
    if not benchmark_file.exists():
        print(f"Benchmark file not found at {benchmark_file}")
        return

    # Assume dev collection is named "dev_corpus" - change as needed
    collection_name = os.getenv("TEST_COLLECTION_NAME", "dev_corpus")
    try:
        coll = load_collection(collection_name)
        facets = _build_facets(coll.cards)
    except Exception as e:
        print(f"Could not load collection '{collection_name}': {e}")
        return

    cache = load_cache()
    
    total_gold_docs = 0
    surviving_gold_docs = 0
    
    total_gold_chunks = 0
    surviving_gold_chunks = 0
    
    results = []
    
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Evaluating {len(lines)} benchmark questions...")
    
    for idx, line in enumerate(lines):
        data = json.loads(line)
        query = data.get("query", data.get("question", ""))
        gold_chunks = data.get("chunks", [])
        
        # Extract gold docs and chunks
        q_gold_docs = set()
        q_gold_chunk_texts = set()
        for gc in gold_chunks:
            q_gold_chunk_texts.add(gc.get("content", "").strip())
            doc = gc.get("metadata", {}).get("document")
            if doc:
                # Map back to doc_id (assumes doc_id is filename minus extension or similar)
                # We will just check against the card's filename
                q_gold_docs.add(doc)
                
        # 1. Compile (with caching)
        if query in cache:
            contract_dict = cache[query]
            contract = EvidenceContract.model_validate(contract_dict)
        else:
            contract = await compile_query(query, facets)
            cache[query] = contract.model_dump()
            save_cache(cache)
            
        # 2. Gate
        gate = apply_gate(coll.cards, contract.scope, settings.GATE_MODE)
        
        # Map kept doc_ids to filenames to compare with gold_docs
        kept_filenames = {c.filename for c in coll.cards if c.doc_id in gate.kept}
        
        q_docs_survived = q_gold_docs.intersection(kept_filenames)
        total_gold_docs += len(q_gold_docs)
        surviving_gold_docs += len(q_docs_survived)
        
        # 3. Siblings
        sib = resolve_siblings(gate, coll.siblings, contract.scope)
        
        # 4. Candidate Retrieval
        candidates = retrieve_candidates(coll, contract, gate, sib)
        
        # Evaluate candidate recall (by text matching since benchmark chunk_ids might differ)
        candidate_chunk_objs = coll.chunks_for(gate.kept)
        candidate_texts = {c.content.strip() for c in candidate_chunk_objs if c.chunk_id in candidates.candidates}
        
        q_chunks_survived = q_gold_chunk_texts.intersection(candidate_texts)
        total_gold_chunks += len(q_gold_chunk_texts)
        surviving_gold_chunks += len(q_chunks_survived)
        
        results.append({
            "query": query,
            "gate_recall": len(q_docs_survived) / len(q_gold_docs) if q_gold_docs else 1.0,
            "cand_recall": len(q_chunks_survived) / len(q_gold_chunk_texts) if q_gold_chunk_texts else 1.0
        })
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(lines)}...")

    gate_recall = surviving_gold_docs / total_gold_docs if total_gold_docs else 0.0
    cand_recall = surviving_gold_chunks / total_gold_chunks if total_gold_chunks else 0.0
    
    print("\n=== Evaluation Results ===")
    print(f"Gate Recall      : {gate_recall:.4f} ({surviving_gold_docs}/{total_gold_docs} gold documents survived)")
    print(f"Candidate Recall : {cand_recall:.4f} ({surviving_gold_chunks}/{total_gold_chunks} gold chunks retrieved)")
    
    if gate_recall < 0.97:
        print("\nWARNING: Gate recall is below the 0.97 guardrail. Consider switching GATE_MODE to 'soft'.")
    else:
        print("\nSUCCESS: Gate recall meets the 0.97 guardrail.")

if __name__ == "__main__":
    asyncio.run(main())

# requirements:
# pydantic>=2.0.0

# contract:
# input: Execution from CLI. Expects benchmark_100.jsonl in DATA_DIR and a built collection.
# output: Console prints reporting gate recall and candidate recall metrics across the dataset, caching LLM compile steps.
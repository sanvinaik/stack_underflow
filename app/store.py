import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.schemas import Chunk, DocCard, SiblingSet
from app.config import settings
from app.ingest import ParsedDoc
# Imported locally in build_collection to avoid circular dependencies during initialization
# from app.siblings import build_sibling_sets

# Singleton model to avoid reloading on every request
_embed_model: SentenceTransformer | None = None

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(settings.EMBED_MODEL)
    return _embed_model

class Collection:
    def __init__(
        self,
        name: str,
        cards: list[DocCard],
        siblings: list[SiblingSet],
        chunks: list[Chunk],
        faiss_index: faiss.Index,
        bm25: BM25Okapi,
        chunk_ids: list[str]
    ):
        self.name = name
        self.cards = cards
        self.siblings = siblings
        self.chunks = chunks
        self.faiss_index = faiss_index
        self.bm25 = bm25
        self.chunk_ids = chunk_ids
        self._cid_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
        
        self._chunk_map = {c.chunk_id: c for c in chunks}
        # In ingest.py, chunk_id is f"{doc_id}_..." so we can filter by doc_id prefix
        self._doc_to_chunks: dict[str, list[Chunk]] = {}
        for c in chunks:
            # Reconstruct doc_id from chunk_id (format: doc_id_c0 or doc_id_p1_c0)
            # Find the first underscore to isolate doc_id, or match against card doc_ids
            for card in cards:
                if c.chunk_id.startswith(card.doc_id + "_"):
                    self._doc_to_chunks.setdefault(card.doc_id, []).append(c)
                    break

    def get_embedding(self, chunk_id: str) -> np.ndarray | None:
        idx = self._cid_to_idx.get(chunk_id)
        if idx is not None and idx < self.faiss_index.ntotal:
            return self.faiss_index.reconstruct(idx)
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        model = _get_embed_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        # Normalize for inner product (cosine similarity)
        faiss.normalize_L2(embeddings)
        return embeddings

    def chunks_for(self, doc_ids: list[str]) -> list[Chunk]:
        """Returns all chunks belonging to the specified doc_ids."""
        result = []
        for doc_id in doc_ids:
            result.extend(self._doc_to_chunks.get(doc_id, []))
        return result

    def dense_scores(self, query: str, chunk_ids: list[str]) -> dict[str, float]:
        """Scores specific chunks against the query using the FAISS index."""
        if not chunk_ids or self.faiss_index.ntotal == 0:
            return {}
            
        q_emb = self.embed([query])
        # Retrieve all scores (efficient enough for <100k chunks), filter for requested chunk_ids
        D, I = self.faiss_index.search(q_emb, self.faiss_index.ntotal)
        
        chunk_id_set = set(chunk_ids)
        scores = {}
        for score, idx in zip(D[0], I[0]):
            cid = self.chunk_ids[idx]
            if cid in chunk_id_set:
                scores[cid] = float(score)
        return scores

    def bm25_scores(self, query: str, chunk_ids: list[str]) -> dict[str, float]:
        """Scores specific chunks against the query using BM25."""
        if not chunk_ids or not self.chunk_ids:
            return {}
            
        tokenized_q = query.lower().split()
        all_scores = self.bm25.get_scores(tokenized_q)
        
        chunk_id_set = set(chunk_ids)
        scores = {}
        for idx, cid in enumerate(self.chunk_ids):
            if cid in chunk_id_set:
                scores[cid] = float(all_scores[idx])
        return scores

# In-memory cache for loaded collections
_collection_cache: dict[str, Collection] = {}

def build_collection(name: str, docs: list[ParsedDoc], cards: list[DocCard]) -> None:
    """
    Builds search indices and cards, saving them atomically to DATA_DIR/name.
    """
    from app.siblings import build_sibling_sets
    
    base_dir = Path(settings.DATA_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)
    target_dir = base_dir / name
    
    chunks = []
    for doc in docs:
        chunks.extend(doc.chunks)
        
    chunk_ids = [c.chunk_id for c in chunks]
    
    with tempfile.TemporaryDirectory(dir=base_dir) as tmp:
        tmp_dir = Path(tmp)
        
        # 1. Save Cards
        cards_path = tmp_dir / "cards.json"
        cards_path.write_text(json.dumps([c.model_dump() for c in cards]), encoding="utf-8")
        
        # 2. Save Chunks
        chunks_path = tmp_dir / "chunks.json"
        chunks_path.write_text(json.dumps([c.model_dump() for c in chunks]), encoding="utf-8")
        
        # 3. Build & Save FAISS Index
        model = _get_embed_model()
        if chunks:
            chunk_texts = [c.content for c in chunks]
            embeddings = model.encode(chunk_texts, convert_to_numpy=True)
            faiss.normalize_L2(embeddings)
            
            d = embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
        else:
            # Fallback for empty collection
            index = faiss.IndexFlatIP(384)
            
        faiss.write_index(index, str(tmp_dir / "faiss.index"))
        
        # 4. Build & Save BM25
        if chunks:
            tokenized_corpus = [c.content.lower().split() for c in chunks]
            bm25 = BM25Okapi(tokenized_corpus)
        else:
            bm25 = BM25Okapi([[""]])
            
        with open(tmp_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)
            
        # 5. Save chunk_ids alignment
        (tmp_dir / "chunk_ids.json").write_text(json.dumps(chunk_ids), encoding="utf-8")
        
        # 6. Sibling Sets (needs embedding of card summaries)
        card_summaries = [c.summary for c in cards]
        if card_summaries:
            summary_embeddings = model.encode(card_summaries, convert_to_numpy=True)
            faiss.normalize_L2(summary_embeddings)
        else:
            summary_embeddings = np.array([])
            
        siblings = build_sibling_sets(cards, summary_embeddings)
        siblings_path = tmp_dir / "siblings.json"
        siblings_path.write_text(json.dumps([s.model_dump() for s in siblings]), encoding="utf-8")
        
        # Atomic swap
        backup_dir = base_dir / f"{name}.old"
        if target_dir.exists():
            target_dir.rename(backup_dir)
            
        # Temp dir rename
        os.rename(tmp_dir, target_dir)
        
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
            
        # Evict from cache to force reload
        if name in _collection_cache:
            del _collection_cache[name]

def load_collection(name: str) -> Collection:
    """Loads a collection from DATA_DIR/name into memory."""
    if name in _collection_cache:
        return _collection_cache[name]
        
    target_dir = Path(settings.DATA_DIR) / name
    if not target_dir.exists() or not target_dir.is_dir():
        raise FileNotFoundError(f"Collection {name} not found.")
        
    cards_data = json.loads((target_dir / "cards.json").read_text(encoding="utf-8"))
    cards = [DocCard.model_validate(c) for c in cards_data]
    
    siblings_data = json.loads((target_dir / "siblings.json").read_text(encoding="utf-8"))
    siblings = [SiblingSet.model_validate(s) for s in siblings_data]
    
    chunks_data = json.loads((target_dir / "chunks.json").read_text(encoding="utf-8"))
    chunks = [Chunk.model_validate(c) for c in chunks_data]
    
    chunk_ids = json.loads((target_dir / "chunk_ids.json").read_text(encoding="utf-8"))
    
    index = faiss.read_index(str(target_dir / "faiss.index"))
    
    with open(target_dir / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
        
    collection = Collection(
        name=name,
        cards=cards,
        siblings=siblings,
        chunks=chunks,
        faiss_index=index,
        bm25=bm25,
        chunk_ids=chunk_ids
    )
    
    _collection_cache[name] = collection
    return collection

# requirements:
# faiss-cpu>=1.7.4
# numpy>=1.24.0
# rank-bm25>=0.2.2
# sentence-transformers>=2.2.2
# pydantic>=2.0.0

# contract:
# input: build_collection receives collection name, list of ParsedDoc, list of DocCard. load_collection receives name.
# output: load_collection returns a Collection instance with chunks_for, dense_scores, bm25_scores, and embed methods initialized and ready.
# file: app/retrieve.py

import math
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from app.schemas import EvidenceContract
from app.gate import GateResult
from app.siblings import SiblingResolution
from app.config import settings

@dataclass
class CandidateSet:
    candidates: set[str] = field(default_factory=set)
    per_requirement: dict[str, list[str]] = field(default_factory=dict)
    chunks: dict[str, Any] = field(default_factory=dict) 
    labels: dict[str, Any] = field(default_factory=dict)
    ledger: dict[str, dict[str, Any]] = field(default_factory=dict)
    tokens_used: int = 0

def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    n = len(vals)
    if n < 30:
        v_min, v_max = min(vals), max(vals)
        if v_min == v_max:
            return {k: 0.0 for k in scores}
        return {k: (v - v_min) / (v_max - v_min) for k, v in scores.items()}
    else:
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if std == 0:
            return {k: 0.0 for k in scores}
        return {k: float((v - mean) / std) for k, v in scores.items()}

def retrieve_and_select(
    coll: Any, 
    contract: EvidenceContract, 
    gate: GateResult, 
    sib: SiblingResolution
) -> CandidateSet:
    out = CandidateSet()
    if not gate.kept:
        return out

    eligible_chunks = coll.chunks_for(gate.kept)
    eligible_ids = [c.chunk_id for c in eligible_chunks]
    if not eligible_ids:
        return out

    out.chunks = {c.chunk_id: c for c in eligible_chunks}
    out.labels = sib.labels
    req_candidates: dict[str, list[str]] = {}
    
    # 1. Per-Requirement Hybrid Scoring
    for req in contract.requirements:
        query_text = req.text
        if req.keywords:
            query_text += " " + " ".join(req.keywords)
            
        dense_raw = coll.dense_scores(query_text, eligible_ids)
        bm25_raw = coll.bm25_scores(query_text, eligible_ids) if settings.LEX_WEIGHT > 0 else {}
        
        dense_norm = _normalize_scores(dense_raw)
        bm25_norm = _normalize_scores(bm25_raw)
        
        combined_scores = {}
        w = settings.LEX_WEIGHT
        for cid in eligible_ids:
            d_s = dense_norm.get(cid, 0.0)
            b_s = bm25_norm.get(cid, 0.0)
            combined_scores[cid] = (1.0 - w) * d_s + w * b_s
            
        sorted_cids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        is_exhaustive = req.type in ["list", "comparison", "condition"] or req.exhaustive
        max_cap = settings.CAND_MAX_LIST if is_exhaustive else settings.CAND_MAX_FACT
        z_floor = 0.0 if is_exhaustive else settings.CAND_Z_FLOOR
        
        kept_for_req = []
        for i, (cid, score) in enumerate(sorted_cids):
            if i < settings.CAND_MIN or score >= z_floor:
                kept_for_req.append(cid)
                if len(kept_for_req) >= max_cap:
                    break
            else:
                break
                
        req_candidates[req.id] = kept_for_req

    # 2. Fast embedding lookup from FAISS (sub-millisecond vs 7+ seconds on CPU)
    unique_cand_ids = set()
    for cids in req_candidates.values():
        unique_cand_ids.update(cids)
        
    emb_map = {}
    missing_cids = []
    for cid in unique_cand_ids:
        vec = coll.get_embedding(cid) if hasattr(coll, "get_embedding") else None
        if vec is not None:
            emb_map[cid] = vec
        else:
            missing_cids.append(cid)
            
    if missing_cids:
        missing_texts = [out.chunks[cid].content for cid in missing_cids]
        m_embs = coll.embed(missing_texts)
        for i, cid in enumerate(missing_cids):
            emb_map[cid] = m_embs[i]

    # 3. MMR Selection & Saturation with Ledger
    selected_embs = []
    for req in contract.requirements:
        cids = req_candidates.get(req.id, [])
        if not cids:
            out.ledger[req.id] = {"count": 0, "stop_reason": "no_candidates"}
            continue
            
        is_exhaustive = req.type in ("list", "comparison", "condition") or req.exhaustive
        req_limit = settings.MAX_SELECTED_CHUNKS if is_exhaustive else 2
        added = 0
        
        for cid in cids:
            if cid in out.candidates:
                continue
                
            emb = emb_map.get(cid)
            if emb is not None and selected_embs:
                # Fast vectorized cosine sim against all selected
                sims = np.dot(selected_embs, emb)
                if np.any(sims >= settings.DEDUPE_COSINE):
                    continue
                
            chunk = out.chunks[cid]
            tokens = len(chunk.content) // 4
            
            if out.tokens_used + tokens > settings.CONTEXT_TOKEN_BUDGET:
                out.ledger[req.id] = {"count": added, "stop_reason": "budget"}
                break
                
            out.candidates.add(cid)
            if emb is not None:
                selected_embs.append(emb)
            out.per_requirement.setdefault(req.id, []).append(cid)
            out.tokens_used += tokens
            added += 1
            
            if added >= req_limit:
                out.ledger[req.id] = {"count": added, "stop_reason": "saturated"}
                break
                
        if req.id not in out.ledger:
            out.ledger[req.id] = {"count": added, "stop_reason": "exhausted"}
            
        if out.tokens_used >= settings.CONTEXT_TOKEN_BUDGET or len(out.candidates) >= settings.MAX_SELECTED_CHUNKS:
            break
            
    return out

# file: app/select.py

import numpy as np
from app.schemas import EvidenceContract
from app.adjudicate import AdjudicationResult, CandidateSet
from app.config import settings

class Selection:
    def __init__(self):
        self.selected = []
        self.ledger = {}
        self.tokens_used = 0

def _cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

def select_chunks(
    contract: EvidenceContract, 
    adjudication: AdjudicationResult, 
    candidates: CandidateSet
) -> Selection:
    """Saturating MMR selector bypassing the LLM adjudicator to maximize context recall and speed."""
    selection = Selection()
    selected_set = set()
    selected_vectors = []
    
    for req in contract.requirements:
        cids = candidates.per_requirement.get(req.id, [])
        if not cids:
            selection.ledger[req.id] = {"count": 0, "stop_reason": "no_candidates"}
            continue
            
        is_exhaustive = req.type in ("list", "comparison", "condition") or req.exhaustive
        req_limit = settings.MAX_SELECTED_CHUNKS if is_exhaustive else 2
        added = 0
        
        for cid in cids:
            if cid in selected_set:
                continue
                
            chunk = candidates.chunks.get(cid)
            if not chunk or not hasattr(chunk, 'embedding') or chunk.embedding is None:
                continue # Skip if no embedding available for MMR
                
            # MMR Discard: Skip if >= DEDUPE_COSINE similar to ANY already selected chunk
            is_redundant = False
            for sv in selected_vectors:
                if _cosine_sim(chunk.embedding, sv) >= settings.DEDUPE_COSINE:
                    is_redundant = True
                    break
            
            if is_redundant:
                continue
                
            tokens = len(chunk.content) // 4
            
            if selection.tokens_used + tokens > settings.CONTEXT_TOKEN_BUDGET:
                selection.ledger[req.id] = {"count": added, "stop_reason": "budget"}
                break
                
            selection.selected.append(cid)
            selected_set.add(cid)
            selected_vectors.append(chunk.embedding)
            selection.tokens_used += tokens
            added += 1
            
            if added >= req_limit:
                selection.ledger[req.id] = {"count": added, "stop_reason": "saturated"}
                break
                
        if req.id not in selection.ledger:
            selection.ledger[req.id] = {"count": added, "stop_reason": "exhausted"}
            
        if selection.tokens_used > settings.CONTEXT_TOKEN_BUDGET or len(selection.selected) >= settings.MAX_SELECTED_CHUNKS:
            break
            
    return selection
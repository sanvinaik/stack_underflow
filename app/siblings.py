import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.schemas import DocCard, SiblingSet, Scope
from app.config import settings
from app.gate import GateResult

@dataclass
class SiblingResolution:
    kept: list[str]
    per_doc_caps: dict[str, int] = field(default_factory=dict)
    labels: dict[str, dict[str, Any]] = field(default_factory=dict)

def _normalize(val: Any) -> str:
    if val is None:
        return ""
    return str(val).strip().lower()

def build_sibling_sets(cards: list[DocCard], embeddings: np.ndarray) -> list[SiblingSet]:
    """
    Builds connected components of documents that are near-duplicates.
    Edges form if (normalized issuer and doc_type match) OR (summary cosine >= SIBLING_COSINE).
    Computes discriminating axes for each set.
    """
    n = len(cards)
    if n < 2:
        return []

    # Adjacency list for connected components (no graph library allowed)
    adj: dict[str, list[str]] = {c.doc_id: [] for c in cards}
    
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = cards[i], cards[j]
            is_sibling = False
            
            # Rule 1: Same normalized issuer and doc_type (and both known)
            if (c1.issuer.status == "known" and c2.issuer.status == "known" and 
                c1.doc_type.status == "known" and c2.doc_type.status == "known"):
                if (_normalize(c1.issuer.value) == _normalize(c2.issuer.value) and
                    _normalize(c1.doc_type.value) == _normalize(c2.doc_type.value)):
                    is_sibling = True
                    
            # Rule 2: Summary cosine similarity
            if not is_sibling and embeddings.size > 0:
                # embeddings are L2 normalized in store.py, so inner product is cosine
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= settings.SIBLING_COSINE:
                    is_sibling = True
                    
            if is_sibling:
                adj[c1.doc_id].append(c2.doc_id)
                adj[c2.doc_id].append(c1.doc_id)

    # Find connected components via DFS
    visited = set()
    components = []
    for doc_id in adj:
        if doc_id not in visited:
            comp = []
            stack = [doc_id]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    comp.append(node)
                    stack.extend(adj[node])
            if len(comp) > 1:
                components.append(comp)

    # Extract axes for each component
    card_map = {c.doc_id: c for c in cards}
    sets = []
    
    fields_to_check = ["fiscal_period_end", "period_year", "issuer", "jurisdiction", "sector", "doc_type"]
    
    for comp_docs in components:
        axes = {}
        # Collect all possible extra_attributes keys in this component
        extra_keys = set()
        for d_id in comp_docs:
            extra_keys.update(card_map[d_id].extra_attributes.keys())
            
        all_fields = fields_to_check + list(extra_keys)
        
        for field_name in all_fields:
            val_map = {}
            distinct_norm_vals = set()
            
            for d_id in comp_docs:
                c = card_map[d_id]
                if field_name in fields_to_check:
                    f = getattr(c, field_name)
                else:
                    f = c.extra_attributes.get(field_name)
                    
                if f and f.status == "known" and f.value is not None:
                    val_map[d_id] = f.value
                    distinct_norm_vals.add(_normalize(f.value))
                    
            # Axis requires more than one distinct known value
            if len(distinct_norm_vals) > 1:
                axes[field_name] = val_map
                
        sets.append(SiblingSet(doc_ids=comp_docs, axes=axes))
        
    return sets

def resolve_siblings(gate: GateResult, sets: list[SiblingSet], scope: Scope) -> SiblingResolution:
    """
    Applies sibling caps and labels if ambiguous scope leaves multiple members of a set alive.
    """
    surviving_set = set(gate.kept)
    resolution = SiblingResolution(kept=gate.kept)
    
    for s_set in sets:
        # Find which members of this sibling set survived the gate
        active_members = [d for d in s_set.doc_ids if d in surviving_set]
        n_active = len(active_members)
        
        if n_active > 1:
            # Ambiguous scope: keep all active members, apply candidate cap
            cap = math.ceil(settings.ADJ_MAX_CANDIDATES / n_active)
            for d_id in active_members:
                resolution.per_doc_caps[d_id] = cap
                
                # Assign labels for composer/adjudicator prompt headers
                doc_labels = {}
                for axis_name, axis_values in s_set.axes.items():
                    if d_id in axis_values:
                        doc_labels[axis_name] = axis_values[d_id]
                if doc_labels:
                    resolution.labels[d_id] = doc_labels
                    
    return resolution

# requirements:
# numpy>=1.24.0

# contract:
# input: build_sibling_sets(cards, embeddings) and resolve_siblings(gate, sets, scope)
# output: Sibling sets computing near-duplicates at index time; resolution dictating chunk caps and doc labels at query time.
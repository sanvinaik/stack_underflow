from dataclasses import dataclass, field
from typing import Any

from app.schemas import DocCard, Scope, ScopeField

@dataclass
class GateResult:
    kept: list[str]
    excluded: list[dict]
    unresolved: bool = False
    multipliers: dict[str, float] = field(default_factory=dict)

def _normalize(val: Any) -> str:
    if val is None:
        return ""
    return str(val).strip().lower()

def _field_matches(scope_val: Any, card_val: Any) -> bool:
    s = _normalize(scope_val)
    c = _normalize(card_val)
    if not s or not c:
        return False
    return s == c or s in c or c in s

def _card_named_in_scope(card: DocCard, scope_val: Any) -> bool:
    s = _normalize(scope_val)
    if not s or len(s) < 3:
        return False
    doc_id = _normalize(card.doc_id)
    filename = _normalize(card.filename)
    title = _normalize(card.title.value) if card.title else ""
    return s in doc_id or doc_id in s or s in filename or s in title

def _check_year(scope_val: Any, card: DocCard) -> tuple[bool, str]:
    """Matches fiscal_period_end, then period_year, then publication_date."""
    sy = _normalize(scope_val)
    if card.fiscal_period_end.status == "known":
        return _normalize(card.fiscal_period_end.value) == sy, "known"
    if card.period_year.status == "known":
        return _normalize(card.period_year.value) == sy, "known"
    if card.publication_date.status == "known":
        return sy in _normalize(card.publication_date.value), "known"
    return False, "unknown"

def _evaluate_field(scope_field: ScopeField | None, card_status: str, match: bool, mode: str) -> tuple[bool, float, str]:
    """Returns (keep, multiplier, reason)."""
    if not scope_field or scope_field.basis != "explicit":
        return True, 1.0, ""
        
    if card_status == "known":
        if match:
            return True, 1.0, ""
        if mode == "hard":
            return False, 0.0, "Mismatched explicit scope"
        return True, 0.1, "Soft mode kept mismatched scope"
        
    # inferred, unknown, conflicting stay eligible with a 0.5 penalty
    return True, 0.5, ""

def apply_gate(cards: list[DocCard], scope: Scope, mode: str) -> GateResult:
    kept = []
    excluded = []
    multipliers = {}
    
    for card in cards:
        keep = True
        doc_mult = 1.0
        exclude_reasons = []
        
        # Override: if card doc_id, filename, or title matches scope field, it is targeted
        is_specifically_named = False
        for s_field in [scope.issuer, scope.doc_type, scope.entities, scope.sector]:
            if s_field and _card_named_in_scope(card, s_field.value):
                is_specifically_named = True
                break
                
        # Check Issuer
        if scope.issuer:
            match = _field_matches(scope.issuer.value, card.issuer.value) or is_specifically_named
            k, m, r = _evaluate_field(scope.issuer, card.issuer.status, match, mode)
            keep &= k; doc_mult *= m
            if not k: exclude_reasons.append("issuer: " + r)
            
        # Check Jurisdiction
        if keep and scope.jurisdiction:
            match = _field_matches(scope.jurisdiction.value, card.jurisdiction.value)
            k, m, r = _evaluate_field(scope.jurisdiction, card.jurisdiction.status, match, mode)
            keep &= k; doc_mult *= m
            if not k: exclude_reasons.append("jurisdiction: " + r)
            
        # Check Sector
        if keep and scope.sector:
            match = _field_matches(scope.sector.value, card.sector.value)
            k, m, r = _evaluate_field(scope.sector, card.sector.status, match, mode)
            keep &= k; doc_mult *= m
            if not k: exclude_reasons.append("sector: " + r)
            
        # Check DocType
        if keep and scope.doc_type:
            match = _field_matches(scope.doc_type.value, card.doc_type.value) or is_specifically_named
            k, m, r = _evaluate_field(scope.doc_type, card.doc_type.status, match, mode)
            keep &= k; doc_mult *= m
            if not k: exclude_reasons.append("doc_type: " + r)
            
        # Check Entities (Multi-entity keeps if any matches)
        if keep and scope.entities and scope.entities.basis == "explicit":
            s_ent = _normalize(scope.entities.value)
            match = is_specifically_named or any(_field_matches(s_ent, e) for e in card.entities)
            if not match and card.entities:
                if mode == "hard":
                    keep = False
                    exclude_reasons.append("entities: Mismatched explicit scope")
                else:
                    doc_mult *= 0.1
            elif not match:
                doc_mult *= 0.5
                
        # Check Year
        if keep and scope.fiscal_period_end:
            match, status = _check_year(scope.fiscal_period_end.value, card)
            k, m, r = _evaluate_field(scope.fiscal_period_end, status, match, mode)
            keep &= k; doc_mult *= m
            if not k: exclude_reasons.append("year: " + r)
            
        if keep:
            kept.append(card.doc_id)
            multipliers[card.doc_id] = doc_mult
        else:
            excluded.append({
                "doc": card.doc_id,
                "reason": "; ".join(exclude_reasons)
            })
            
    # Fallback: if zero documents survived the gate, run ungated
    unresolved = False
    if not kept and cards:
        unresolved = True
        kept = [c.doc_id for c in cards]
        multipliers = {c.doc_id: 1.0 for c in cards}
        excluded.clear()
        
    return GateResult(
        kept=kept,
        excluded=excluded,
        unresolved=unresolved,
        multipliers=multipliers
    )

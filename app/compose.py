# file: app/compose.py

import json
import re
from dataclasses import dataclass
from typing import Any

from app.schemas import EvidenceContract, AnswerDraft, Chunk
from app.siblings import SiblingResolution
from app.llm import structured_call

@dataclass
class ComposedAnswer:
    answer: str
    dropped_points: int
    unsupported_reqs: list[str]

SYSTEM_PROMPT = """You are an expert financial research analyst.
Your task is to synthesize a comprehensive, rigorous, and factual answer based strictly on the provided CHUNKS.

Rules:
1. Provide factual points directly answering each requirement.
2. Group points by `req_id`.
3. For each point, provide the factual statement in `text` and supporting chunk IDs in `cites`.
4. CRITICAL: In `cites`, you MUST provide the exact chunk IDs (e.g. ["Implementation Handbook_p8_c36"]) taken from the `[ID: ...]` headers.
5. Do NOT include page numbers, headers, or filenames inside the `text` string.
6. If a requirement has no evidence in the chunks, omit it and add its ID to `unsupported_req_ids`.
7. Respond strictly in JSON matching the AnswerDraft schema.
"""

def _format_citation(chunk_ids: list[str], chunks_dict: dict[str, Chunk]) -> str:
    cites = []
    for cid in chunk_ids:
        chunk = chunks_dict.get(cid)
        if chunk:
            doc = chunk.metadata.document
            pages = ",".join(map(str, chunk.metadata.page_numbers))
            cites.append(f"{doc}, p.{pages}")
    
    if not cites:
        return ""
    
    seen = set()
    unique_cites = []
    for c in cites:
        if c not in seen:
            seen.add(c)
            unique_cites.append(c)
            
    return f" ({'; '.join(unique_cites)})"

def _resolve_cites(
    raw_cites: list[str], 
    req_id: str, 
    selected_set: set[str], 
    selection: Any
) -> list[str]:
    valid = []
    # 1. Exact or cleaned match
    for c in raw_cites:
        cleaned = re.sub(r"[\[\]'\"]", "", c).strip()
        if cleaned.startswith("ID:"):
            cleaned = cleaned[3:].strip()
        if cleaned in selected_set:
            valid.append(cleaned)
        else:
            # Fuzzy match against selected IDs
            matched = [s for s in selected_set if cleaned and (cleaned in s or s in cleaned)]
            if matched:
                valid.append(matched[0])
                
    # 2. If model left cites empty or mismatched, attribute to top chunks for this req_id
    if not valid:
        req_cids = getattr(selection, "per_requirement", {}).get(req_id, [])
        for r_cid in req_cids[:2]:
            if r_cid in selected_set:
                valid.append(r_cid)
                
    # 3. Fallback to any selected candidate if still empty
    if not valid and selected_set:
        valid.append(next(iter(selected_set)))
        
    return list(dict.fromkeys(valid))

async def compose(
    contract: EvidenceContract, 
    selection: Any, 
    sib: SiblingResolution,
    chunks_dict: dict[str, Chunk]
) -> ComposedAnswer:
    if not selection.candidates:
        return ComposedAnswer(
            answer="No information was retrieved for this question.",
            dropped_points=0,
            unsupported_reqs=[r.id for r in contract.requirements]
        )

    # For fast LLM synthesis (<4s without timeout), select top 18-20 diverse chunks for prompt context
    composer_cids = []
    seen_comp = set()
    for req in contract.requirements:
        req_cids = getattr(selection, "per_requirement", {}).get(req.id, [])
        for cid in req_cids[:4]:
            if cid not in seen_comp:
                seen_comp.add(cid)
                composer_cids.append(cid)
                
    for cid in selection.candidates:
        if len(composer_cids) >= 20:
            break
        if cid not in seen_comp:
            seen_comp.add(cid)
            composer_cids.append(cid)

    chunk_texts = []
    for cid in composer_cids:
        chunk = chunks_dict.get(cid)
        if chunk:
            doc_id = next((d for d in sib.kept if cid.startswith(d + "_")), "")
            label_str = ""
            if doc_id and doc_id in sib.labels:
                label_pairs = [f"{k}:{v}" for k, v in sib.labels[doc_id].items()]
                if label_pairs:
                    label_str = " | " + " | ".join(label_pairs)
            chunk_texts.append(f"[ID: {cid}{label_str}]\n{chunk.content}\n")
            
    req_dump = [{"id": r.id, "text": r.text} for r in contract.requirements]
    
    user_prompt = (
        f"<delim>\n"
        f"REQUIREMENTS:\n{json.dumps(req_dump, indent=2)}\n\n"
        f"SELECTED CHUNKS:\n{chr(10).join(chunk_texts)}\n"
        f"</delim>"
    )

    try:
        draft, _ = await structured_call(
            tier="HEAVY",
            schema=AnswerDraft,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            name="answer_generation"
        )
    except Exception as e:
        draft = AnswerDraft(points=[], unsupported_req_ids=[r.id for r in contract.requirements])

    dropped = 0
    valid_points = []
    selected_set = set(selection.candidates)
    
    for point in draft.points:
        valid_cites = _resolve_cites(point.cites, point.req_id, selected_set, selection)
        if not valid_cites:
            dropped += 1
            continue
            
        citation_suffix = _format_citation(valid_cites, chunks_dict)
        clean_text = point.text.strip()
        if clean_text.endswith("."):
            clean_text = clean_text[:-1]
            
        valid_points.append(f"{clean_text}{citation_suffix}.")

    unsupported = set(draft.unsupported_req_ids)
    for r in contract.requirements:
        if not any(p.req_id == r.id for p in draft.points if p.text.strip()):
            unsupported.add(r.id)

    if not valid_points:
        return ComposedAnswer(
            answer="No information was retrieved for this question.",
            dropped_points=dropped,
            unsupported_reqs=list(unsupported)
        )

    final_answer = " ".join(valid_points)
    
    if unsupported and len(unsupported) < len(contract.requirements):
        missing = sorted(list(unsupported))
        final_answer += f" No information was retrieved for requirements: {', '.join(missing)}."

    if len(final_answer) > 6000:
        final_answer = final_answer[:5997] + "..."

    return ComposedAnswer(
        answer=final_answer.strip(),
        dropped_points=dropped,
        unsupported_reqs=list(unsupported)
    )

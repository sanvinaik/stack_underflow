import asyncio
import json
import traceback
from app.store import load_collection
from app.compile import compile_query
from app.gate import apply_gate
from app.siblings import resolve_siblings
from app.retrieve import retrieve_and_select
from app.compose import SYSTEM_PROMPT, compose
from app.llm import structured_call
from app.schemas import AnswerDraft

async def main():
    coll = load_collection('dev_collection')
    q = 'What are the key highlights or topics discussed in the Implementation Handbook?'
    contract = await compile_query(q, {})
    gate = apply_gate(coll.cards, contract.scope, 'hard')
    sib = resolve_siblings(gate, coll.siblings, contract.scope)
    sel = retrieve_and_select(coll, contract, gate, sib)
    
    print(f"Contract requirements: {[r.id for r in contract.requirements]}")
    print(f"Selected candidates count: {len(sel.candidates)}")
    
    chunk_texts = []
    for cid in sel.candidates:
        chunk = sel.chunks.get(cid)
        if chunk:
            chunk_texts.append(f"[ID: {cid}]\n{chunk.content}\n")
    
    req_dump = [{"id": r.id, "text": r.text} for r in contract.requirements]
    user_prompt = (
        f"<delim>\n"
        f"REQUIREMENTS:\n{json.dumps(req_dump, indent=2)}\n\n"
        f"SELECTED CHUNKS:\n{chr(10).join(chunk_texts)}\n"
        f"</delim>"
    )
    print(f"Prompt length chars: {len(user_prompt)}")
    
    try:
        draft, meta = await structured_call(
            tier="HEAVY",
            schema=AnswerDraft,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            name="answer_generation"
        )
        print("Success draft points count:", len(draft.points))
        for p in draft.points:
            print("Point:", p.text[:80], "cites:", p.cites, "req_id:", p.req_id)
        print("Unsupported:", draft.unsupported_req_ids)
    except Exception as e:
        print("Exception occurred:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

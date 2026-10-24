import asyncio
import os
import time
import statistics
import sys
from langfuse.openai import AsyncOpenAI
from pydantic import ValidationError

# Ensure the app module can be found if script is run from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.schemas import IndexResponse

async def run_single_call(client: AsyncOpenAI, model: str, name: str) -> dict:
    start_time = time.perf_counter()
    tokens = 0
    empty_content = False
    success = False
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a testing bot. You must output JSON matching this schema: {'status': 'success'}."},
                {"role": "user", "content": "Return the exact success status JSON."}
            ],
            response_format={"type": "json_object"},
            max_tokens=10000,
            name=name
        )
        
        content = response.choices[0].message.content
        if response.usage:
            tokens = response.usage.total_tokens
            
        if not content:
            empty_content = True
        else:
            try:
                # Validate against a known schema from app.schemas
                IndexResponse.model_validate_json(content)
                success = True
            except ValidationError:
                pass
                
    except Exception as e:
        print(f"Call failed: {e}")
        empty_content = True
        
    duration_ms = (time.perf_counter() - start_time) * 1000
    return {
        "duration_ms": duration_ms,
        "tokens": tokens,
        "empty": empty_content,
        "success": success
    }

async def main():
    base_url = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/inference/openai/v1")
    api_key = os.getenv("OPENCODE_API_KEY")
    model = os.getenv("OPENCODE_MODEL_FAST", "deepseek-v4.1-flash")
    
    if model.endswith("-free"):
        raise ValueError(f"Model {model} is rejected (cannot end with '-free').")
        
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    
    print(f"Starting smoke test against {base_url} using model {model}...")
    
    # 1. 20 sequential JSON-mode calls
    print("\n--- Running 20 Sequential Calls ---")
    sequential_results = []
    for i in range(20):
        res = await run_single_call(client, model, name=f"smoke_seq_{i}")
        sequential_results.append(res)
        print(f"Call {i+1}/20: {res['duration_ms']:.2f}ms | Empty: {res['empty']} | Success: {res['success']}")
        
    # Calculate sequential metrics
    durations = sorted([r["duration_ms"] for r in sequential_results])
    p50 = durations[len(durations)//2]
    p95 = durations[int(len(durations) * 0.95)]
    empty_count = sum(1 for r in sequential_results if r["empty"])
    total_tokens = sum(r["tokens"] for r in sequential_results)
    
    print(f"\nSequential Metrics:")
    print(f"p50 Latency: {p50:.2f} ms")
    print(f"p95 Latency: {p95:.2f} ms")
    print(f"Empty Content Rate: {empty_count / 20 * 100:.1f}%")
    print(f"Total Tokens Used: {total_tokens}")
    
    # 2. 10 parallel calls for shard pattern rate limits
    print("\n--- Running 10 Parallel Calls ---")
    parallel_tasks = [run_single_call(client, model, name=f"smoke_par_{i}") for i in range(10)]
    
    parallel_start = time.perf_counter()
    parallel_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
    parallel_duration = (time.perf_counter() - parallel_start) * 1000
    
    success_count = sum(1 for r in parallel_results if isinstance(r, dict) and r.get("success"))
    print(f"Parallel Execution Time (10 shards): {parallel_duration:.2f} ms")
    print(f"Parallel Success Rate: {success_count / 10 * 100:.1f}%")
    
    # Flush Langfuse telemetry before exiting
    client.flush()
    print("\nSmoke test complete. Langfuse telemetry flushed.")

if __name__ == "__main__":
    asyncio.run(main())

# requirements:
# langfuse>=2.0.0
# pydantic>=2.0.0

# contract:
# input: Execution from CLI. Requires OPENCODE_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST in env.
# output: Console prints detailing p50/p95 latency, empty-content rates, and parallel shard completion times.
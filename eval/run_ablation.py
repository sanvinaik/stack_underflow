# file: eval/run_ablation.py

"""Run SG-ECP ablation evaluations against the local FastAPI server.

Place this file at <project-root>/eval/run_ablation.py, replacing the old file.
Expected benchmark: <project-root>/data/benchmark_100.jsonl (or DATA_DIR override).
The API server must already be running and have the requested collection indexed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

# Resolve project root whether this script is run from the project root or eval/.
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = (
    SCRIPT_PATH.parent.parent
    if SCRIPT_PATH.parent.name.lower() == "eval"
    else SCRIPT_PATH.parent
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load the project's .env when python-dotenv is installed. Environment variables
# already set in the shell take precedence over values in the .env file.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

try:
    from eval.official_eval import OpenCodeClient, RAGMetricsEvaluator
except ImportError as exc:
    raise SystemExit(
        "Could not import eval.official_eval. Put this script in the project's "
        "eval/ folder and ensure eval/official_eval.py exists. "
        f"Import error: {exc}"
    ) from exc


METRIC_COLUMNS = [
    "answer_precision",
    "answer_recall",
    "context_precision",
    "context_recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablations over the SG-ECP HTTP API and compute evaluation metrics."
    )
    parser.add_argument(
        "--ablation",
        required=True,
        choices=["B0_20", "B0_80", "B1", "B2", "B3"],
        help="Ablation identifier. Configure the API server for this ablation before running.",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the running FastAPI server (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--collection",
        default="dev_collection",
        help="Indexed collection name. Must match the collection you uploaded.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N benchmark rows (use --limit 1 for a smoke test).",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=0,
        help="Zero-based benchmark row to start at (default: 0).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path. Default: project root/ablation_results_<ablation>.csv",
    )
    return parser.parse_args()


def resolve_data_dir() -> Path:
    raw = os.getenv("DATA_DIR")
    path = Path(raw).expanduser() if raw else Path("data")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_first_nonempty(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_gold_chunks(record: dict[str, Any]) -> list[str]:
    """Read reference chunks from common benchmark field names."""
    raw_chunks = None
    for key in ("chunks", "benchmark_chunks", "expected_chunks", "gold_chunks"):
        value = record.get(key)
        if isinstance(value, list):
            raw_chunks = value
            break

    texts: list[str] = []
    for item in raw_chunks or []:
        if isinstance(item, str) and item.strip():
            texts.append(item.strip())
        elif isinstance(item, dict):
            content = item.get("content") or item.get("text")
            if isinstance(content, str) and content.strip():
                texts.append(content.strip())
    return texts


def extract_retrieved_chunks(raw_chunks: Any) -> list[str]:
    if not isinstance(raw_chunks, list):
        return []
    texts: list[str] = []
    for item in raw_chunks:
        if isinstance(item, str) and item.strip():
            texts.append(item.strip())
        elif isinstance(item, dict):
            content = item.get("content") or item.get("text")
            if isinstance(content, str) and content.strip():
                texts.append(content.strip())
    return texts


def extract_trace(raw_trace: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_trace, list):
        return []
    return [step for step in raw_trace if isinstance(step, dict)]


def classify_failure(trace: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    if (
        metrics.get("answer_recall") == 1.0
        and metrics.get("context_recall") == 1.0
    ):
        return "none"

    for step in trace:
        step_name = step.get("step_name", "")
        output = step.get("output", {})
        if not isinstance(output, dict):
            output = {}

        if step_name == "scope_gate" and output.get("scope_unresolved"):
            return "gate_unresolved"
        if step_name == "coverage_selection":
            ledger = output.get("ledger", {})
            if isinstance(ledger, dict) and any(
                isinstance(req, dict) and req.get("stop_reason") == "budget"
                for req in ledger.values()
            ):
                return "selector_budget_cut"
        if step_name == "evidence_adjudication":
            if output.get("adjudicator_skipped"):
                return "adjudicator_fallback"
            if output.get("adjudication_partial"):
                return "adjudicator_timeout"
        if step_name == "answer_generation" and output.get("unsupported_requirements"):
            return "composer_unsupported_req"

    if metrics.get("context_recall", 1.0) < 1.0:
        return "retrieval_miss"
    if metrics.get("answer_recall", 1.0) < 1.0:
        return "composer_omission"
    return "unknown"


def estimate_cost(trace: list[dict[str, Any]]) -> float:
    """Rough placeholder estimate; replace with provider usage/pricing if available."""
    total_tokens = 0
    for step in trace:
        output = step.get("output", {})
        if isinstance(output, dict):
            try:
                total_tokens += int(output.get("tokens_used", 0) or 0)
            except (TypeError, ValueError):
                pass
    return (total_tokens / 1_000_000) * 0.15


async def run_query_http(
    client: httpx.AsyncClient, base_url: str, collection: str, query: str
) -> dict[str, Any]:
    started = time.perf_counter()
    url = f"{base_url.rstrip('/')}/query"
    try:
        response = await client.post(
            url,
            json={"collection_name": collection, "query": query},
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("The /query endpoint returned JSON that is not an object.")
        data["latency_s"] = time.perf_counter() - started
        return data
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:1000]
        return {
            "error": f"HTTP {exc.response.status_code} from {url}: {body}",
            "latency_s": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "error": f"Request to {url} failed: {type(exc).__name__}: {exc}",
            "latency_s": time.perf_counter() - started,
        }


def base_row(q_id: str, series: str, ablation: str) -> dict[str, Any]:
    return {
        "id": q_id,
        "series": series,
        "ablation": ablation,
        "query": "",
        "latency_s": None,
        "cost_usd": None,
        "answer_precision": None,
        "answer_recall": None,
        "context_precision": None,
        "context_recall": None,
        "failure_category": "unknown",
        "error": "",
    }


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


async def main() -> None:
    args = parse_args()

    api_key = os.getenv("OPENCODE_API_KEY", "").strip()
    if not api_key or api_key in {"YOUR_OPENCODE_API_KEY_HERE", "YOUR_API_KEY_HERE"}:
        raise SystemExit(
            "OPENCODE_API_KEY is missing. Set it in your shell or project .env file; "
            "do not hard-code it into this script."
        )

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1.")
    if args.start_at < 0:
        raise SystemExit("--start-at cannot be negative.")

    data_dir = resolve_data_dir()
    benchmark_file = data_dir / "benchmark_100.jsonl"
    if not benchmark_file.is_file():
        raise SystemExit(
            f"Benchmark file not found: {benchmark_file}\n"
            "Set DATA_DIR to the folder containing benchmark_100.jsonl."
        )

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else PROJECT_ROOT / f"ablation_results_{args.ablation}.csv"
    )
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()

    try:
        evaluator = RAGMetricsEvaluator(OpenCodeClient(api_key=api_key))
    except Exception as exc:
        raise SystemExit(f"Could not initialize the official evaluator: {exc}") from exc

    records: list[dict[str, Any]] = []
    try:
        with benchmark_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict):
                        records.append(parsed)
                    else:
                        print(f"Skipping line {line_number}: JSON value is not an object.")
                except json.JSONDecodeError as exc:
                    print(f"Skipping malformed JSON on line {line_number}: {exc}")
    except OSError as exc:
        raise SystemExit(f"Could not read benchmark file {benchmark_file}: {exc}") from exc

    if not records:
        raise SystemExit(f"No valid benchmark records found in {benchmark_file}.")

    records = records[args.start_at :]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No benchmark rows remain after applying --start-at/--limit.")

    results: list[dict[str, Any]] = []
    print(f"Project root : {PROJECT_ROOT}", flush=True)
    print(f"Benchmark    : {benchmark_file}", flush=True)
    print(f"API URL      : {args.url.rstrip('/')}", flush=True)
    print(f"Collection   : {args.collection}", flush=True)
    print(f"Ablation     : {args.ablation}", flush=True)
    print(f"Questions    : {len(records)}", flush=True)
    print(f"Output CSV   : {output_path}", flush=True)
    print("Starting evaluation...", flush=True)

    async with httpx.AsyncClient() as client:
        for idx, record in enumerate(records, start=1):
            q_id_value = record.get("id", f"Q{idx}")
            q_id = str(q_id_value)
            series = q_id[0].upper() if q_id else "U"
            query = get_first_nonempty(record, ("query", "question"))
            row = base_row(q_id, series, args.ablation)
            row["query"] = query
            print(f"[{idx}/{len(records)}] {q_id}: {query[:100] or '[EMPTY QUERY]'}", flush=True)

            if not query:
                row["error"] = "Benchmark record has no non-empty 'query' or 'question' field."
                row["failure_category"] = "benchmark_format_error"
                results.append(row)
                save_results(results, output_path)
                continue

            benchmark_answer = get_first_nonempty(
                record, ("expected_answer", "benchmark_answer", "reference_answer", "gold_answer")
            )
            if not benchmark_answer:
                row["error"] = (
                    "No reference answer found. Expected one of: expected_answer, "
                    "benchmark_answer, reference_answer, gold_answer."
                )
                row["failure_category"] = "benchmark_format_error"
                results.append(row)
                save_results(results, output_path)
                print(f"  ERROR: {row['error']}", flush=True)
                continue

            # Request to your local API
            api_result = await run_query_http(client, args.url, args.collection, query)
            row["latency_s"] = api_result.get("latency_s")

            # ---> NEW PRINT STATEMENT TO DIAGNOSE WAITING TIME <---
            print(f"  -> API finished in {row['latency_s']:.2f}s. Starting official LLM evaluation...", flush=True)

            if "error" in api_result:
                row["error"] = api_result["error"]
                row["failure_category"] = "api_fault"
                results.append(row)
                save_results(results, output_path)
                print(f"  API ERROR: {row['error']}", flush=True)
                continue

            answer_value = api_result.get("answer", "")
            if isinstance(answer_value, str):
                answer = answer_value
            elif answer_value is None:
                answer = ""
            else:
                answer = json.dumps(answer_value, ensure_ascii=False)

            # PROTECTIVE SLICE: Cap chunks fed to the evaluator to prevent it from hanging[cite: 4]
            raw_chunks = api_result.get("chunks", [])
            retrieved_chunks = extract_retrieved_chunks(raw_chunks)[:10]
            
            benchmark_chunks = extract_gold_chunks(record)
            trace = extract_trace(api_result.get("trace", []))
            row["cost_usd"] = estimate_cost(trace)

            try:
                # The evaluator is synchronous (uses requests); run it in a thread so
                # it does not block the asyncio event loop used by httpx.
                metrics = await asyncio.to_thread(
                    evaluator.evaluate_single,
                    query=query,
                    ai_answer=answer,
                    benchmark_answer=benchmark_answer,
                    retrieved_chunks=retrieved_chunks,
                    benchmark_chunks=benchmark_chunks,
                )

                for metric_name in METRIC_COLUMNS:
                    value = metrics.get(metric_name)
                    row[metric_name] = float(value) if value is not None else None
                row["failure_category"] = classify_failure(trace, metrics)

            except Exception as exc:
                row["error"] = f"Evaluator failed: {type(exc).__name__}: {exc}"
                row["failure_category"] = "evaluation_fault"
                print(f"  EVALUATION ERROR: {row['error']}", flush=True)

            results.append(row)
            save_results(results, output_path)  # Preserve progress if a later question fails.

            if not row["error"]:
                print(
                    "  Metrics: "
                    f"AnsP={row['answer_precision']:.3f} "
                    f"AnsR={row['answer_recall']:.3f} "
                    f"CtxP={row['context_precision']:.3f} "
                    f"CtxR={row['context_recall']:.3f}",
                    flush=True,
                )

    df = pd.DataFrame(results)
    successful = df.dropna(subset=METRIC_COLUMNS) if not df.empty else pd.DataFrame()
    failed_count = int(df["error"].astype(bool).sum()) if not df.empty else 0

    print("\n=== Evaluation Summary ===", flush=True)
    print(f"Rows processed : {len(df)}", flush=True)
    print(f"Metric rows    : {len(successful)}", flush=True)
    print(f"Rows with error: {failed_count}", flush=True)

    if not successful.empty:
        print("\n--- Overall Metrics ---", flush=True)
        print(successful[METRIC_COLUMNS].mean().to_frame("mean").T.to_string(index=False), flush=True)

        print("\n--- Metrics by Series ---", flush=True)
        grouped = successful.groupby("series")[METRIC_COLUMNS + ["latency_s", "cost_usd"]].mean()
        print(grouped.to_string(), flush=True)
    else:
        print(
            "No rows produced all four metrics. Inspect the per-question ERROR messages "
            "and the CSV for details.",
            flush=True,
        )

    if failed_count:
        print("\n--- Failures ---", flush=True)
        for _, failed_row in df[df["error"].astype(bool)].iterrows():
            print(f"{failed_row['id']}: {failed_row['error']}", flush=True)

    print(f"\nSaved results to: {output_path}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Any completed rows were already saved to CSV.", flush=True)
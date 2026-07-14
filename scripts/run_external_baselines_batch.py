#!/usr/bin/env python3
"""Prepare, submit, inspect, and collect frozen external-baseline Batch jobs."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import AsyncOpenAI, OpenAI

from scripts import run_arch_smoke_comparison as arch
from scripts import run_baselines as verified


MODES = ("GPT4O_LONGCTX", "LINC", "LOGIC_LM")
BATCH_PRICE_IN = 0.075
BATCH_PRICE_OUT = 0.30


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def compose_prompt(mode: str, row: dict[str, Any], context: str) -> tuple[str, str]:
    if mode == "GPT4O_LONGCTX":
        system = (
            "You are a product-passport reasoning assistant. Answer using ONLY the provided context. "
            "Return a concise answer containing all requested values. If insufficient, say INSUFFICIENT EVIDENCE."
        )
        user = f"Domain: {row['domain']}\nContext:\n{context}\n\nQuestion: {row['query']}"
    elif mode == "LINC":
        system = (
            "You are a neuro-symbolic reasoning agent in the style of LINC. Extract premises from the supplied "
            "context, express the question as a goal, reason step by step, and return one final answer containing "
            "all requested values. Use only the supplied context."
        )
        user = (
            f"Domain: {row['domain']}\nContext:\n{context}\n\nQuestion: {row['query']}\n\n"
            "Output format:\nPremises: ...\nGoal: ...\nReasoning: ...\nFinal: <one line>"
        )
    else:
        system = (
            "You are Logic-LM. Classify the question, extract relevant premises from the provided context, "
            "reason over them, and return a single final answer containing all requested values. Use only context."
        )
        user = (
            f"Domain: {row['domain']}\nContext:\n{context}\n\nQuestion: {row['query']}\n\n"
            "Output format:\nKind: RULE|LOOKUP|COMPOSE\nReasoning: ...\nAnswer: <one line>"
        )
    return system, user


def parse_answer(mode: str, raw: str) -> str:
    if mode == "LINC":
        match = re.search(r"Final\s*:\s*(.+)", raw, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else raw.strip()
    if mode == "LOGIC_LM":
        match = re.search(r"Answer\s*:\s*(.+)", raw, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else raw.strip()
    return raw.strip()


def request(custom_id: str, model: str, system: str, user: str, max_tokens: int) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
    }


def prepare(args: argparse.Namespace) -> int:
    input_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    estimated_in = 0
    estimated_out = 0
    commit = git_commit()

    keep = set(json.loads(args.verified_ids.read_text(encoding="utf-8")))
    rows = verified.load_release_benchmark(args.verified_benchmark)
    rows = sorted([r for r in rows if f"{r['id']}|{r['domain']}" in keep], key=lambda r: (r["id"], r["domain"]))
    if len(rows) != 6270:
        raise SystemExit(f"verified row gate failed: {len(rows)} != 6270")
    doc_cache: dict[str, list[dict[str, str]]] = {}
    mem_cache: dict[str, list[dict[str, str]]] = {}
    for mode in MODES:
        for index, row in enumerate(rows):
            context = verified.build_retrieved_context(
                row, verified.DEFAULT_DOCS_ROOT, doc_cache, mem_cache, args.max_context_chars
            )
            system, user = verified.PROMPT_BUILDERS[mode](row["query"], row["type"], row["domain"], context)
            custom_id = f"v-{mode.lower()}-{index:05d}"
            max_tokens = verified.MODE_MAX_OUTPUT[mode]
            input_rows.append(request(custom_id, args.model, system, user, max_tokens))
            manifest.append(
                {
                    "custom_id": custom_id,
                    "benchmark": "verified_release_6270",
                    "mode": mode,
                    "row": row,
                    "max_tokens": max_tokens,
                    "max_context_chars": args.max_context_chars,
                    "harness_commit": commit,
                }
            )
            estimated_in += approx_tokens(system) + approx_tokens(user) + 16
            estimated_out += max_tokens

    compose_rows = arch.load_benchmark_rows(args.compositional_benchmark)
    if len(compose_rows) != 3000:
        raise SystemExit(f"compositional row gate failed: {len(compose_rows)} != 3000")
    doc_cache = {}
    mem_cache = {}
    for mode in MODES:
        for index, row in enumerate(compose_rows):
            context = arch.context_for(row, doc_cache, mem_cache)
            system, user = compose_prompt(mode, row, context)
            custom_id = f"c-{mode.lower()}-{index:05d}"
            input_rows.append(request(custom_id, args.model, system, user, 220))
            manifest.append(
                {
                    "custom_id": custom_id,
                    "benchmark": "compositional_3000",
                    "mode": mode,
                    "row": row,
                    "max_tokens": 220,
                    "max_context_chars": None,
                    "harness_commit": commit,
                }
            )
            estimated_in += approx_tokens(system) + approx_tokens(user) + 16
            estimated_out += 220

    if len(input_rows) != 27810 or len({r["custom_id"] for r in input_rows}) != len(input_rows):
        raise SystemExit("request count or custom_id uniqueness gate failed")
    estimated_cost = estimated_in * BATCH_PRICE_IN / 1_000_000 + estimated_out * BATCH_PRICE_OUT / 1_000_000
    if estimated_cost > args.budget_usd:
        raise SystemExit(f"estimated Batch cost ${estimated_cost:.4f} exceeds cap ${args.budget_usd:.4f}")
    write_jsonl(args.input_jsonl, input_rows)
    write_jsonl(args.manifest_jsonl, manifest)
    summary = {
        "harness_commit": commit,
        "requested_model": args.model,
        "requests": len(input_rows),
        "verified_requests": 18810,
        "compositional_requests": 9000,
        "estimated_input_tokens": estimated_in,
        "max_output_tokens": estimated_out,
        "estimated_batch_cost_usd": estimated_cost,
        "budget_cap_usd": args.budget_usd,
    }
    args.prepare_report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def submit(args: argparse.Namespace) -> int:
    if args.state_json.exists():
        raise SystemExit(f"refusing to overwrite existing state: {args.state_json}")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with args.input_jsonl.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"evaluation": "IJCKG2026-frozen-external-baselines", "commit": git_commit()},
    )
    state = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "status": batch.status,
        "endpoint": batch.endpoint,
        "completion_window": batch.completion_window,
        "harness_commit": git_commit(),
    }
    args.state_json.parent.mkdir(parents=True, exist_ok=True)
    args.state_json.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, indent=2))
    return 0


def estimated_request_input_tokens(row: dict[str, Any]) -> int:
    messages = row["body"].get("messages") or []
    return sum(approx_tokens(str(message.get("content") or "")) for message in messages) + 16


def shard(args: argparse.Namespace) -> int:
    requests = read_jsonl(args.input_jsonl)
    metadata = read_jsonl(args.manifest_jsonl)
    if len(requests) != len(metadata):
        raise SystemExit("request/manifest length mismatch")
    by_id = {row["custom_id"]: row for row in metadata}
    shards: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        index = len(shards)
        directory = args.output_dir / f"shard_{index:03d}"
        input_path = directory / "input.jsonl"
        manifest_path = directory / "manifest.jsonl"
        write_jsonl(input_path, current)
        write_jsonl(manifest_path, [by_id[row["custom_id"]] for row in current])
        shards.append(
            {
                "index": index,
                "requests": len(current),
                "estimated_input_tokens": current_tokens,
                "input_jsonl": str(input_path),
                "manifest_jsonl": str(manifest_path),
                "state_json": str(directory / "state.json"),
                "raw_output": str(directory / "output.jsonl"),
            }
        )
        current = []
        current_tokens = 0

    for row in requests:
        tokens = estimated_request_input_tokens(row)
        if tokens > args.max_estimated_input_tokens:
            raise SystemExit(f"single request exceeds shard token cap: {row['custom_id']} tokens={tokens}")
        if current and current_tokens + tokens > args.max_estimated_input_tokens:
            flush()
        current.append(row)
        current_tokens += tokens
    flush()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "SHARDS.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_requests": len(requests),
                "max_estimated_input_tokens": args.max_estimated_input_tokens,
                "shards": shards,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"shards": len(shards), "requests": len(requests), "index": str(index_path)}, indent=2))
    return 0


def run_shards(args: argparse.Namespace) -> int:
    index = json.loads(args.shards_json.read_text(encoding="utf-8"))
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    terminal_bad = {"failed", "expired", "cancelled"}
    for item in index["shards"]:
        state_path = Path(item["state_json"])
        raw_output = Path(item["raw_output"])
        if raw_output.exists():
            print(f"[batch-shards] shard={item['index']} already downloaded", flush=True)
            continue
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            batch = client.batches.retrieve(state["batch_id"])
        else:
            with Path(item["input_jsonl"]).open("rb") as handle:
                uploaded = client.files.create(file=handle, purpose="batch")
            batch = client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "evaluation": "IJCKG2026-frozen-external-baselines",
                    "commit": git_commit(),
                    "shard": str(item["index"]),
                },
            )
            state = {
                "batch_id": batch.id,
                "input_file_id": uploaded.id,
                "status": batch.status,
                "harness_commit": git_commit(),
                "shard": item["index"],
            }
            state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            print(f"[batch-shards] submitted shard={item['index']} batch={batch.id}", flush=True)
        while batch.status not in {"completed", *terminal_bad}:
            counts = batch.request_counts.model_dump() if batch.request_counts else {}
            print(f"[batch-shards] shard={item['index']} status={batch.status} counts={counts}", flush=True)
            time.sleep(args.poll_seconds)
            batch = client.batches.retrieve(batch.id)
        if batch.status in terminal_bad:
            detail = batch.errors.model_dump() if batch.errors else None
            raise SystemExit(f"shard {item['index']} ended with {batch.status}: {detail}")
        if not batch.output_file_id:
            raise SystemExit(f"shard {item['index']} completed without output_file_id")
        raw_output.write_text(client.files.content(batch.output_file_id).text, encoding="utf-8")
        state.update({"status": batch.status, "output_file_id": batch.output_file_id})
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        print(f"[batch-shards] completed shard={item['index']} output={raw_output}", flush=True)
    return 0


def combine_shards(args: argparse.Namespace) -> int:
    index = json.loads(args.shards_json.read_text(encoding="utf-8"))
    request_count = 0
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    with args.raw_output.open("w", encoding="utf-8") as output, args.manifest_jsonl.open("w", encoding="utf-8") as manifest:
        for item in index["shards"]:
            raw_path = Path(item["raw_output"])
            manifest_path = Path(item["manifest_jsonl"])
            if not raw_path.exists():
                raise SystemExit(f"missing shard output: {raw_path}")
            raw_text = raw_path.read_text(encoding="utf-8")
            output.write(raw_text)
            if raw_text and not raw_text.endswith("\n"):
                output.write("\n")
            manifest_text = manifest_path.read_text(encoding="utf-8")
            manifest.write(manifest_text)
            if manifest_text and not manifest_text.endswith("\n"):
                manifest.write("\n")
            request_count += sum(1 for line in raw_text.splitlines() if line.strip())
    print(f"combined shard outputs: {request_count} rows -> {args.raw_output}")
    return 0 if request_count == index["source_requests"] else 3


async def run_live_async(args: argparse.Namespace) -> int:
    requests = read_jsonl(args.input_jsonl)
    if args.limit is not None:
        requests = requests[: args.limit]
    completed: set[str] = set()
    prior_in = prior_out = 0
    if args.raw_output.exists():
        for row in read_jsonl(args.raw_output):
            response = row.get("response") or {}
            if response.get("status_code") == 200:
                completed.add(row["custom_id"])
                usage = (response.get("body") or {}).get("usage") or {}
                prior_in += int(usage.get("prompt_tokens", 0))
                prior_out += int(usage.get("completion_tokens", 0))
    remaining = [row for row in requests if row["custom_id"] not in completed]
    running_in, running_out = prior_in, prior_out
    running_cost = running_in * args.price_in / 1_000_000 + running_out * args.price_out / 1_000_000
    if running_cost >= args.budget_usd:
        raise SystemExit(f"existing live output cost ${running_cost:.4f} already reaches cap ${args.budget_usd:.4f}")
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=args.timeout)
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for row in remaining:
        queue.put_nowait(row)
    lock = asyncio.Lock()
    stop = asyncio.Event()
    failures: list[dict[str, str]] = []
    completed_this_run = 0
    handle = args.raw_output.open("a", encoding="utf-8")

    class RollingTokenLimiter:
        def __init__(self, tokens_per_minute: int) -> None:
            self.limit = tokens_per_minute
            self.events: deque[tuple[float, int]] = deque()
            self.total = 0
            self.guard = asyncio.Lock()

        async def acquire(self, tokens: int) -> None:
            while True:
                delay = 0.0
                async with self.guard:
                    now = time.monotonic()
                    while self.events and now - self.events[0][0] >= 60.0:
                        _, expired = self.events.popleft()
                        self.total -= expired
                    if self.total + tokens <= self.limit:
                        self.events.append((now, tokens))
                        self.total += tokens
                        return
                    if self.events:
                        delay = max(0.05, 60.05 - (now - self.events[0][0]))
                await asyncio.sleep(delay)

    limiter = RollingTokenLimiter(args.tpm_limit)
    request_guard = asyncio.Lock()
    next_request_at = 0.0

    async def pace_request() -> None:
        """Space requests globally when an account-level rolling RPD cap applies."""
        nonlocal next_request_at
        if args.request_interval <= 0:
            return
        async with request_guard:
            now = time.monotonic()
            delay = max(0.0, next_request_at - now)
            if delay:
                await asyncio.sleep(delay)
            next_request_at = time.monotonic() + args.request_interval
    if args.initial_delay:
        print(f"[live] initial cooldown={args.initial_delay:.1f}s", flush=True)
        await asyncio.sleep(args.initial_delay)

    async def worker() -> None:
        nonlocal running_in, running_out, running_cost, completed_this_run
        while not queue.empty() and not stop.is_set():
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            last_error: Exception | None = None
            response = None
            estimated_tokens = estimated_request_input_tokens(item)
            for attempt in range(args.attempts):
                try:
                    await limiter.acquire(estimated_tokens)
                    await pace_request()
                    response = await client.chat.completions.create(**item["body"])
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < args.attempts:
                        await asyncio.sleep(min(15.0, 1.5 * (2**attempt)))
            if response is None:
                async with lock:
                    failures.append({"custom_id": item["custom_id"], "error": f"{type(last_error).__name__}: {last_error}"})
                    stop.set()
                queue.task_done()
                return
            body = response.model_dump()
            usage = body.get("usage") or {}
            output = {
                "custom_id": item["custom_id"],
                "execution_api": "live",
                "response": {
                    "status_code": 200,
                    "request_id": getattr(response, "_request_id", ""),
                    "body": body,
                },
                "error": None,
            }
            async with lock:
                handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                running_in += int(usage.get("prompt_tokens", 0))
                running_out += int(usage.get("completion_tokens", 0))
                running_cost = running_in * args.price_in / 1_000_000 + running_out * args.price_out / 1_000_000
                completed_this_run += 1
                if completed_this_run % 100 == 0:
                    print(
                        f"[live] completed={len(completed) + completed_this_run}/{len(requests)} "
                        f"cost=${running_cost:.4f} failures={len(failures)}",
                        flush=True,
                    )
                if running_cost > args.budget_usd:
                    stop.set()
            queue.task_done()

    try:
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    finally:
        handle.close()
        await client.close()
    report = {
        "requests": len(requests),
        "previously_completed": len(completed),
        "completed_this_run": completed_this_run,
        "remaining": len(requests) - len(completed) - completed_this_run,
        "tokens_in": running_in,
        "tokens_out": running_out,
        "cost_usd": running_cost,
        "budget_cap_usd": args.budget_usd,
        "failures": failures,
    }
    args.live_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not failures and report["remaining"] == 0 else 3


def run_live(args: argparse.Namespace) -> int:
    return asyncio.run(run_live_async(args))


def fetch_batch(state_path: Path) -> tuple[OpenAI, Any]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return client, client.batches.retrieve(state["batch_id"])


def status(args: argparse.Namespace) -> int:
    _, batch = fetch_batch(args.state_json)
    payload = {
        "id": batch.id,
        "status": batch.status,
        "request_counts": batch.request_counts.model_dump() if batch.request_counts else None,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "errors": batch.errors.model_dump() if batch.errors else None,
        "usage": batch.usage.model_dump() if getattr(batch, "usage", None) else None,
    }
    print(json.dumps(payload, indent=2))
    return 0 if batch.status not in {"failed", "expired", "cancelled"} else 2


def verified_row(meta: dict[str, Any], body: dict[str, Any], request_id: str, execution_api: str) -> dict[str, Any]:
    source = meta["row"]
    mode = meta["mode"]
    raw = body["choices"][0]["message"]["content"] or ""
    answer = parse_answer(mode, raw)
    usage = body.get("usage") or {}
    success = max(0, verified.score_answer(answer, source["expected_contains"], source["type"], source["query"]))
    return {
        "id": source["id"], "mode": mode, "type": source["type"], "domain": source["domain"],
        "query": source["query"], "product": source.get("product", ""), "session": source.get("session", "s1"),
        "success": success,
        "steps": json.dumps([{"source": mode, "text": raw, "request_id": request_id, "execution_api": execution_api}]),
        "correct": success, "latency_ms": "", "confidence": 0.85 if success else 0.45,
        "confidence_raw": 0.85 if success else 0.45, "confidence_cal": 0.85 if success else 0.45,
        "cost_retrieval_calls": 1, "cost_rule_checks": 0,
        "cost_tokens_in": int(usage.get("prompt_tokens", 0)), "cost_tokens_out": int(usage.get("completion_tokens", 0)),
        "n_steps": 1, "answer": answer, "expected_contains": source["expected_contains"], "cost_usd_running": "",
        "model": body.get("model", ""), "temperature": "0.0", "max_context_chars": meta["max_context_chars"],
        "max_output_tokens": meta["max_tokens"], "harness_commit": meta["harness_commit"],
    }


def compositional_row(meta: dict[str, Any], body: dict[str, Any], request_id: str, execution_api: str) -> dict[str, Any]:
    source = meta["row"]
    mode = meta["mode"]
    raw = body["choices"][0]["message"]["content"] or ""
    answer = parse_answer(mode, raw)
    usage = body.get("usage") or {}
    return {
        "id": source["id"], "domain": source["domain"], "subtype": source.get("subtype", source.get("type", "")),
        "mode": mode, "required_sources": "+".join(source.get("required_sources") or []),
        "expected_groups": json.dumps(source["expected_groups"], ensure_ascii=False),
        "success": arch.score_answer(answer, source["expected_groups"]), "llm_used": 1, "answer": answer,
        "tokens_in": int(usage.get("prompt_tokens", 0)), "tokens_out": int(usage.get("completion_tokens", 0)),
        "answer_trace": json.dumps({"raw_answer": raw, "request_id": request_id, "execution_api": execution_api}),
        "model": body.get("model", ""), "temperature": "0.0", "max_output_tokens": meta["max_tokens"],
        "harness_commit": meta["harness_commit"],
    }


def materialize(content: str, manifest_path: Path, output_dir: Path, execution_api: str, run_id: str) -> int:
    output = {row["custom_id"]: row for row in (json.loads(line) for line in content.splitlines() if line.strip())}
    manifest = read_jsonl(manifest_path)
    failures = []
    verified_by_mode = {mode: [] for mode in MODES}
    compose_rows = []
    total_in = total_out = 0
    resolved_models: set[str] = set()
    for meta in manifest:
        item = output.get(meta["custom_id"])
        if not item or item.get("error") or not item.get("response") or item["response"].get("status_code") != 200:
            failures.append({"custom_id": meta["custom_id"], "output": item})
            continue
        response = item["response"]
        body = response["body"]
        usage = body.get("usage") or {}
        total_in += int(usage.get("prompt_tokens", 0))
        total_out += int(usage.get("completion_tokens", 0))
        resolved_models.add(str(body.get("model", "")))
        if meta["benchmark"] == "verified_release_6270":
            verified_by_mode[meta["mode"]].append(
                verified_row(meta, body, response.get("request_id", ""), execution_api)
            )
        else:
            compose_rows.append(compositional_row(meta, body, response.get("request_id", ""), execution_api))
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode, rows in verified_by_mode.items():
        path = output_dir / "verified" / f"{mode}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=verified.EVAL_CSV_FIELDS)
            writer.writeheader(); writer.writerows(rows)
    compose_path = output_dir / "compositional" / "external_baselines.csv"
    compose_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "domain", "subtype", "mode", "required_sources", "expected_groups", "success", "llm_used", "answer", "tokens_in", "tokens_out", "answer_trace", "model", "temperature", "max_output_tokens", "harness_commit"]
    with compose_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(compose_rows)
    report = {
        "run_id": run_id, "execution_api": execution_api, "status": "completed", "expected": len(manifest), "collected": len(output),
        "failures": len(failures), "verified_rows_by_mode": {k: len(v) for k, v in verified_by_mode.items()},
        "compositional_rows": len(compose_rows), "resolved_models": sorted(resolved_models),
        "tokens_in": total_in, "tokens_out": total_out,
        "estimated_actual_cost_usd": total_in * (BATCH_PRICE_IN if execution_api == "batch" else 0.15) / 1_000_000
        + total_out * (BATCH_PRICE_OUT if execution_api == "batch" else 0.60) / 1_000_000,
    }
    (output_dir / "EXTERNAL_COLLECTION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "external_failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not failures and all(len(v) == 6270 for v in verified_by_mode.values()) and len(compose_rows) == 9000 else 3


def collect(args: argparse.Namespace) -> int:
    client, batch = fetch_batch(args.state_json)
    if batch.status != "completed" or not batch.output_file_id:
        raise SystemExit(f"batch is not collectable: status={batch.status}")
    content = client.files.content(batch.output_file_id).text
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(content, encoding="utf-8")
    return materialize(content, args.manifest_jsonl, args.output_dir, "batch", batch.id)


def collect_local(args: argparse.Namespace) -> int:
    content = args.raw_output.read_text(encoding="utf-8")
    return materialize(content, args.manifest_jsonl, args.output_dir, "live", "concurrent-live")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--verified-benchmark", type=Path, required=True)
    prep.add_argument("--verified-ids", type=Path, required=True)
    prep.add_argument("--compositional-benchmark", type=Path, required=True)
    prep.add_argument("--input-jsonl", type=Path, required=True)
    prep.add_argument("--manifest-jsonl", type=Path, required=True)
    prep.add_argument("--prepare-report", type=Path, required=True)
    prep.add_argument("--model", default="gpt-4o-mini")
    prep.add_argument("--max-context-chars", type=int, default=12000)
    prep.add_argument("--budget-usd", type=float, default=4.50)
    prep.set_defaults(function=prepare)
    send = sub.add_parser("submit")
    send.add_argument("--input-jsonl", type=Path, required=True)
    send.add_argument("--state-json", type=Path, required=True)
    send.set_defaults(function=submit)
    split = sub.add_parser("shard")
    split.add_argument("--input-jsonl", type=Path, required=True)
    split.add_argument("--manifest-jsonl", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--max-estimated-input-tokens", type=int, default=1_650_000)
    split.set_defaults(function=shard)
    run = sub.add_parser("run-shards")
    run.add_argument("--shards-json", type=Path, required=True)
    run.add_argument("--poll-seconds", type=int, default=30)
    run.set_defaults(function=run_shards)
    combine = sub.add_parser("combine-shards")
    combine.add_argument("--shards-json", type=Path, required=True)
    combine.add_argument("--raw-output", type=Path, required=True)
    combine.add_argument("--manifest-jsonl", type=Path, required=True)
    combine.set_defaults(function=combine_shards)
    live = sub.add_parser("run-live")
    live.add_argument("--input-jsonl", type=Path, required=True)
    live.add_argument("--raw-output", type=Path, required=True)
    live.add_argument("--live-report", type=Path, required=True)
    live.add_argument("--concurrency", type=int, default=20)
    live.add_argument("--attempts", type=int, default=4)
    live.add_argument("--timeout", type=float, default=45.0)
    live.add_argument("--budget-usd", type=float, default=6.0)
    live.add_argument("--limit", type=int, default=None)
    live.add_argument("--price-in", type=float, default=0.15)
    live.add_argument("--price-out", type=float, default=0.60)
    live.add_argument("--tpm-limit", type=int, default=170000)
    live.add_argument("--initial-delay", type=float, default=0.0)
    live.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="minimum seconds between API requests across all workers (for rolling RPD limits)",
    )
    live.set_defaults(function=run_live)
    check = sub.add_parser("status")
    check.add_argument("--state-json", type=Path, required=True)
    check.set_defaults(function=status)
    done = sub.add_parser("collect")
    done.add_argument("--state-json", type=Path, required=True)
    done.add_argument("--manifest-jsonl", type=Path, required=True)
    done.add_argument("--raw-output", type=Path, required=True)
    done.add_argument("--output-dir", type=Path, required=True)
    done.set_defaults(function=collect)
    local = sub.add_parser("collect-local")
    local.add_argument("--manifest-jsonl", type=Path, required=True)
    local.add_argument("--raw-output", type=Path, required=True)
    local.add_argument("--output-dir", type=Path, required=True)
    local.set_defaults(function=collect_local)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())

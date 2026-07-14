#!/usr/bin/env python3
"""Prepare, submit, inspect, and collect frozen external-baseline Batch jobs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

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


def verified_row(meta: dict[str, Any], body: dict[str, Any], request_id: str) -> dict[str, Any]:
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
        "steps": json.dumps([{"source": mode, "text": raw, "request_id": request_id, "execution_api": "batch"}]),
        "correct": success, "latency_ms": "", "confidence": 0.85 if success else 0.45,
        "confidence_raw": 0.85 if success else 0.45, "confidence_cal": 0.85 if success else 0.45,
        "cost_retrieval_calls": 1, "cost_rule_checks": 0,
        "cost_tokens_in": int(usage.get("prompt_tokens", 0)), "cost_tokens_out": int(usage.get("completion_tokens", 0)),
        "n_steps": 1, "answer": answer, "expected_contains": source["expected_contains"], "cost_usd_running": "",
        "model": body.get("model", ""), "temperature": "0.0", "max_context_chars": meta["max_context_chars"],
        "max_output_tokens": meta["max_tokens"], "harness_commit": meta["harness_commit"],
    }


def compositional_row(meta: dict[str, Any], body: dict[str, Any], request_id: str) -> dict[str, Any]:
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
        "answer_trace": json.dumps({"raw_answer": raw, "request_id": request_id, "execution_api": "batch"}),
        "model": body.get("model", ""), "temperature": "0.0", "max_output_tokens": meta["max_tokens"],
        "harness_commit": meta["harness_commit"],
    }


def collect(args: argparse.Namespace) -> int:
    client, batch = fetch_batch(args.state_json)
    if batch.status != "completed" or not batch.output_file_id:
        raise SystemExit(f"batch is not collectable: status={batch.status}")
    content = client.files.content(batch.output_file_id).text
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(content, encoding="utf-8")
    output = {row["custom_id"]: row for row in (json.loads(line) for line in content.splitlines() if line.strip())}
    manifest = read_jsonl(args.manifest_jsonl)
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
            verified_by_mode[meta["mode"]].append(verified_row(meta, body, response.get("request_id", "")))
        else:
            compose_rows.append(compositional_row(meta, body, response.get("request_id", "")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode, rows in verified_by_mode.items():
        path = args.output_dir / "verified" / f"{mode}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=verified.EVAL_CSV_FIELDS)
            writer.writeheader(); writer.writerows(rows)
    compose_path = args.output_dir / "compositional" / "external_baselines.csv"
    compose_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "domain", "subtype", "mode", "required_sources", "expected_groups", "success", "llm_used", "answer", "tokens_in", "tokens_out", "answer_trace", "model", "temperature", "max_output_tokens", "harness_commit"]
    with compose_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(compose_rows)
    report = {
        "batch_id": batch.id, "status": batch.status, "expected": len(manifest), "collected": len(output),
        "failures": len(failures), "verified_rows_by_mode": {k: len(v) for k, v in verified_by_mode.items()},
        "compositional_rows": len(compose_rows), "resolved_models": sorted(resolved_models),
        "tokens_in": total_in, "tokens_out": total_out,
        "estimated_actual_batch_cost_usd": total_in * BATCH_PRICE_IN / 1_000_000 + total_out * BATCH_PRICE_OUT / 1_000_000,
    }
    (args.output_dir / "BATCH_COLLECTION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "batch_failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not failures and all(len(v) == 6270 for v in verified_by_mode.values()) and len(compose_rows) == 9000 else 3


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
    check = sub.add_parser("status")
    check.add_argument("--state-json", type=Path, required=True)
    check.set_defaults(function=status)
    done = sub.add_parser("collect")
    done.add_argument("--state-json", type=Path, required=True)
    done.add_argument("--manifest-jsonl", type=Path, required=True)
    done.add_argument("--raw-output", type=Path, required=True)
    done.add_argument("--output-dir", type=Path, required=True)
    done.set_defaults(function=collect)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())

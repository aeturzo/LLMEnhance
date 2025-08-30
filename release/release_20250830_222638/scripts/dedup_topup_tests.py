#!/usr/bin/env python3
"""
Unify dataset hygiene + auto top-up for a domain.

- Dedup by (query, product, type)
- Normalize IDs -> {domain}_{typ}_{00001}
- Auto top-up: call scripts/gen_dataset.py with deficits ONLY, merge results, re-dedup
- Repeat up to --max-iters times until targets are met

Usage examples:
  # battery defaults (180/140/100)
  python scripts/dedup_topup_tests.py --domain battery

  # lexmark with explicit targets
  python scripts/dedup_topup_tests.py --domain lexmark --targets recall=150,logic=120,open=90

  # viessmann, more attempts if generator produces duplicates
  python scripts/dedup_topup_tests.py --domain viessmann --max-iters 5
"""
from __future__ import annotations
import argparse, json, os, random, shutil, subprocess, time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_SCRIPT = REPO_ROOT / "scripts" / "gen_dataset.py"

DEFAULTS = {
    "battery":   {"recall": 180, "logic": 140, "open": 100},
    "lexmark":   {"recall": 150, "logic": 120, "open":  90},
    "viessmann": {"recall": 150, "logic": 120, "open":  90},
    "textiles":  {"recall": 180, "logic": 140, "open": 100},
}

def norm_str(s: str | None) -> str:
    return (s or "").strip().lower()

def read_jsonl(p: Path) -> List[dict]:
    if not p.exists(): return []
    out: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                out.append(json.loads(t))
    return out

def write_jsonl(p: Path, rows: List[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def dedup_rows(rows: List[dict]) -> List[dict]:
    seen: set[Tuple[str,str,str]] = set()
    out: List[dict] = []
    for r in rows:
        q = norm_str(r.get("query"))
        prod = norm_str(r.get("product"))
        typ = norm_str(r.get("type") or "open")
        key = (q, prod, typ)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def normalize_ids(rows: List[dict], domain: str) -> List[dict]:
    used: set[str] = set()
    seq: Dict[str,int] = {}
    out: List[dict] = []
    for r in rows:
        typ = norm_str(r.get("type") or "open")
        rid = r.get("id")
        if (not rid) or (rid in used):
            seq[typ] = seq.get(typ, 0) + 1
            rid = f"{domain}_{typ[:3]}_{seq[typ]:05d}"
        used.add(rid)
        r = dict(r)
        r["id"] = rid
        out.append(r)
    return out

def count_by_type(rows: List[dict]) -> Dict[str,int]:
    c: Dict[str,int] = {}
    for r in rows:
        t = norm_str(r.get("type") or "open")
        c[t] = c.get(t, 0) + 1
    # ensure keys exist
    for k in ("recall","logic","open"):
        c.setdefault(k, 0)
    return c

def parse_targets(s: str | None, domain: str) -> Dict[str,int]:
    if s:
        out: Dict[str,int] = {"recall":0,"logic":0,"open":0}
        for part in s.split(","):
            if not part.strip(): continue
            k, v = part.split("=")
            out[norm_str(k)] = int(v.strip())
        return out
    return DEFAULTS.get(domain, {"recall":180,"logic":140,"open":100})

def run_generator(domain: str, need: Dict[str,int]) -> Path:
    """
    Call scripts/gen_dataset.py with ONLY the missing counts.
    This will overwrite tests/{domain}/tests.jsonl with a file that contains exactly those numbers (if generator respects counts).
    We'll immediately rename that to tests/{domain}/tests.generated.jsonl and merge it into the main set.
    """
    tests_dir = REPO_ROOT / "tests" / domain
    tests_dir.mkdir(parents=True, exist_ok=True)
    out_file = tests_dir / "tests.jsonl"
    gen_env = os.environ.copy()
    # run in repo root so relative paths in generator work
    cmd = [
        "python", str(GEN_SCRIPT),
        "--domain", domain,
        "--n_recall", str(max(0, need.get("recall", 0))),
        "--n_logic",  str(max(0, need.get("logic", 0))),
        "--n_open",   str(max(0, need.get("open", 0))),
    ]
    print(f"[gen] calling: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), env=gen_env, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise SystemExit(f"[gen] generator failed (exit {res.returncode}). See logs above.")
    # move freshly generated file aside so we can merge
    gen_out = tests_dir / f"tests.generated.{int(time.time())}.jsonl"
    if not out_file.exists():
        raise SystemExit(f"[gen] expected generator to write {out_file}, but it does not exist.")
    shutil.move(str(out_file), str(gen_out))
    print(f"[gen] generated -> {gen_out}")
    return gen_out

def merge_unique(old_rows: List[dict], new_rows: List[dict]) -> List[dict]:
    return dedup_rows(old_rows + new_rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["battery","lexmark","viessmann","textiles"])
    ap.add_argument("--targets", default=None, help="e.g. recall=180,logic=140,open=100")
    ap.add_argument("--max-iters", type=int, default=3, help="max top-up rounds")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    domain = args.domain
    targets = parse_targets(args.targets, domain)

    tests_file = REPO_ROOT / "tests" / domain / "tests.jsonl"
    if not tests_file.exists():
        raise SystemExit(f"Input not found: {tests_file} (Generate once with scripts/gen_dataset.py)")

    # Step 1: load + dedup + normalize IDs
    base_rows = normalize_ids(dedup_rows(read_jsonl(tests_file)), domain)
    write_jsonl(tests_file, base_rows)

    # Step 2: iterate top-up if needed
    for it in range(1, args.max_iters + 1):
        counts = count_by_type(base_rows)
        need = {k: max(0, targets[k] - counts.get(k, 0)) for k in targets}
        print(f"[iter {it}] domain={domain} counts={counts} targets={targets} need={need}")
        if all(v <= 0 for v in need.values()):
            break

        # copy current to a safe place (pre-merge snapshot)
        snapshot = tests_file.with_suffix(f".pre.{int(time.time())}.jsonl")
        shutil.copyfile(tests_file, snapshot)
        print(f"[iter {it}] snapshot -> {snapshot.name}")

        # run generator for deficits only; merge; re-dedup; renumber IDs
        gen_out = run_generator(domain, need)
        new_rows = read_jsonl(gen_out)
        merged = merge_unique(base_rows, new_rows)
        base_rows = normalize_ids(merged, domain)
        write_jsonl(tests_file, base_rows)
        print(f"[iter {it}] merged {len(new_rows)} new → total {len(base_rows)}")

    # Final report
    final_counts = count_by_type(base_rows)
    print(f"[done] domain={domain} final counts={final_counts} targets={targets}")
    shortfalls = {k: max(0, targets[k] - final_counts.get(k, 0)) for k in targets}
    if any(v > 0 for v in shortfalls.values()):
        print("[warn] still below targets after max-iters. You can re-run with higher --max-iters "
              "or regenerate more examples (ontology may be limiting).")
    else:
        print("[ok] targets met or exceeded.")

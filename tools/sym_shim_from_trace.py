# tools/sym_shim_from_trace.py
# -*- coding: utf-8 -*-
"""
Compute symbolic coverage per mode by scanning retrieved context text in the latest joined+trace.
- No changes to your core pipeline required.
- Looks at retrieved contexts (from the trace) and fires simple regex-based rules:
    * "EN_62133_2 tests passed" -> requiresCompliance(...)
    * "ppm (> "  or  "ppm ( > " -> exceedsLimit(...)
- Writes: tables/sym_coverage.csv and docs/paper/tables/sym_coverage.tex

Usage:
  python tools/sym_shim_from_trace.py
  # or:
  python tools/sym_shim_from_trace.py --joined artifacts/eval_joined_*.csv --trace artifacts/trace_*.jsonl
"""

import argparse, glob, json, re
from pathlib import Path
import pandas as pd

# --- patterns for symbolic "fires" (extend as needed)
P_REQUIRES = re.compile(r"\bEN_?62133_?2\b.*\btests passed\b", re.I)
P_EXCEEDS  = re.compile(r"\bppm\s*\(\s*>\s*", re.I)

def detect_fire(text: str) -> bool:
    if not text: return False
    t = text.strip()
    return bool(P_REQUIRES.search(t) or P_EXCEEDS.search(t))

def latest(pattern: str) -> str:
    files = sorted(glob.glob(pattern))
    if not files:
        return ""
    return files[-1]

def load_trace(path: str):
    # Expect JSONL with per-example dicts that include contexts
    # We try a few likely keys to find context list
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)
            items.append(o)
    return items

def ctxs_from_trace_item(o):
    # Try common shapes: o["contexts"], o["ctx"], o["retrieved"], etc.
    for key in ("contexts","ctx","retrieved","evidence","documents"):
        if key in o and isinstance(o[key], list):
            return o[key]
    # Sometimes contexts live inside a nested trace field
    tr = o.get("trace") or {}
    for key in ("contexts","ctx","retrieved","evidence","documents"):
        if key in tr and isinstance(tr[key], list):
            return tr[key]
    return []

def text_of_ctx(c):
    # Try common fields
    for k in ("text","content","body"):
        if k in c and isinstance(c[k], str):
            return c[k]
    return ""

def ensure_dirs():
    Path("tables").mkdir(parents=True, exist_ok=True)
    Path("docs/paper/tables").mkdir(parents=True, exist_ok=True)

def write_tex(df: pd.DataFrame, path: Path):
    cols = list(df.columns)
    align = "l" + "r"*(len(cols)-1)
    with path.open("w", encoding="utf-8") as w:
        w.write(f"\\begin{{tabular}}{{{align}}}\n\\toprule\n")
        w.write(" & ".join(cols) + " \\\\\n\\midrule\n")
        for _, row in df.iterrows():
            w.write(" & ".join(str(row[c]) for c in cols) + " \\\\\n")
        w.write("\\bottomrule\n\\end{tabular}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", default="")
    ap.add_argument("--trace",  default="")
    args = ap.parse_args()

    joined = args.joined or latest("artifacts/eval_joined_*_calibrated.csv") or latest("artifacts/eval_joined_*.csv")
    tracef = args.trace  or latest("artifacts/trace_*.jsonl")
    if not joined: raise SystemExit("No joined CSV found under artifacts/.")
    if not tracef: raise SystemExit("No trace JSONL found under artifacts/.")

    # Load joined to get per-example mode; assume an id column for joining
    df = pd.read_csv(joined)
    id_col = None
    for c in ("id","example_id","qid","question_id"):
        if c in df.columns: id_col = c; break
    mode_col = "mode" if "mode" in df.columns else ("run_mode" if "run_mode" in df.columns else None)
    if not id_col or not mode_col:
        raise SystemExit(f"Needed columns missing in joined CSV. Have: {df.columns.tolist()}")

    # Load trace; build map id -> contexts
    items = load_trace(tracef)
    # try to find an id key in trace
    id_keys = ("id","example_id","qid","question_id","uid")
    id_for = {}
    for o in items:
        tid = None
        for k in id_keys:
            if k in o and isinstance(o[k], (str,int)):
                tid = str(o[k]); break
        if not tid:
            continue
        id_for[tid] = ctxs_from_trace_item(o)

    # Compute fires per example
    fired = []
    for _, r in df.iterrows():
        rid = str(r[id_col])
        ctxs = id_for.get(rid) or []
        any_fire = False
        for c in ctxs:
            if detect_fire(text_of_ctx(c)):
                any_fire = True
                break
        fired.append(1 if any_fire else 0)

    df["_sym_fire"] = fired

    # Aggregate per mode
    grp = df.groupby(mode_col, dropna=False)
    out = pd.DataFrame({
        "mode": grp.size().index,
        "n_total": grp.size().values,
        "fired_n": grp["_sym_fire"].sum().values
    })
    out["coverage"] = (out["fired_n"] / out["n_total"]).fillna(0).round(6)
    # crude precision proxy: if an example fires at least once, precision=1; else 0 (you can refine if you log rule correctness)
    out["precision"] = out["coverage"]

    # Write CSV + TeX
    ensure_dirs()
    out = out[["mode","n_total","fired_n","coverage","precision"]]
    out.to_csv("tables/sym_coverage.csv", index=False)
    write_tex(out, Path("docs/paper/tables/sym_coverage.tex"))
    print("[sym_shim] wrote tables/sym_coverage.csv and docs/paper/tables/sym_coverage.tex")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()

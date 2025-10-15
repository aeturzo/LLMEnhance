import os, sys, json, glob, textwrap
from pathlib import Path

ROOT = Path(".").resolve()
OUT  = Path("debug_bundle.txt")

def add_file(p):
    try:
        s = Path(p).read_text(encoding="utf-8")
    except Exception as e:
        s = f"[ERR reading {p}: {e}]"
    return f"\n===== BEGIN FILE: {p} =====\n{s}\n===== END FILE: {p} =====\n"

def add_cmd(title, cmd, content):
    return f"\n===== {title}: {cmd} =====\n{content}\n"

sections = []

# 0) repo tree (top 2 levels)
tree = []
for d, dirs, files in os.walk(".", topdown=True):
    drel = os.path.relpath(d, ".")
    depth = drel.count(os.sep)
    if depth > 2: 
        dirs[:] = []
        continue
    tree.append(drel + "/")
    for f in files:
        tree.append(os.path.join(drel, f))
sections.append(add_cmd("TREE", "top 2 levels", "\n".join(sorted(tree))))

# 1) grep to find exporter & accept/join
def run_grep(pattern):
    out=[]
    for p in ROOT.rglob("*.py"):
        if any(skip in str(p) for skip in [".git","__pycache__","site-packages",".venv","env","build"]):
            continue
        try:
            s = p.read_text(encoding="utf-8")
            if pattern in s:
                for i, line in enumerate(s.splitlines(), 1):
                    if pattern in line:
                        out.append(f"{p}:{i}:{line}")
        except Exception:
            pass
    return "\n".join(out) or "[no matches]"

sections.append(add_cmd("GREP", 'eval_summary / paper tables', 
    run_grep("eval_summary")))
sections.append(add_cmd("GREP", 'docs/paper/tables', 
    run_grep("docs/paper/tables")))
sections.append(add_cmd("GREP", 'eval_joined / Acceptance OK / join / accept', 
    "\n".join(filter(None, [
        run_grep("eval_joined"),
        run_grep("Acceptance OK"),
        run_grep(" join("),
        run_grep("accept")
    ]))))

# 2) include key files if present
candidates = [
 "scripts/run_paper_pipeline.py",
 "run_eval_all.py",
 "scripts/build_corpus.py",
 "backend/retrieval/hybrid.py",
]
for c in candidates:
    if Path(c).exists():
        sections.append(add_file(c))

# 3) include any likely exporter / joiner files we discovered
hits = set()
for line in sections[-3].splitlines():  # last GREP block
    if line.startswith("[") or not line.strip(): 
        continue
    path = line.split(":",1)[0]
    if path.endswith(".py"): hits.add(path)
for h in sorted(hits):
    if Path(h).exists() and h not in candidates:
        sections.append(add_file(h))

# 4) artifacts + joined CSV shapes
def safe_read_csv(path, n=5):
    try:
        import pandas as pd
        df = pd.read_csv(path)
        head = df.head(n).to_csv(index=False)
        cols = ", ".join(df.columns.astype(str).tolist())
        return f"[{path}] cols: {cols}\nHEAD:\n{head}"
    except Exception as e:
        return f"[ERR reading {path}: {e}]"

# sizes
def wc_glob(globpat):
    lines=[]
    for p in sorted(glob.glob(globpat)):
        try:
            n = sum(1 for _ in open(p, encoding="utf-8"))
            lines.append(f"{n:7d}  {p}")
        except Exception as e:
            lines.append(f"[ERR] {p}: {e}")
    return "\n".join(lines) or f"[no files matching {globpat}]"

sections.append(add_cmd("WC", "artifacts/eval_*_*.csv", wc_glob("artifacts/eval_*_*.csv")))
sections.append(add_cmd("WC", "artifacts/eval_joined_*.csv", wc_glob("artifacts/eval_joined_*.csv")))
sections.append(add_cmd("WC", "docs/paper/tables/*", wc_glob("docs/paper/tables/*")))

# 5) sample rows from a joined CSV (first available)
joined = sorted(glob.glob("artifacts/eval_joined_*.csv"))
if joined:
    sections.append(add_cmd("JOINED SAMPLE", joined[-1], safe_read_csv(joined[-1], n=10)))

# 6) one gold item + retriever smoke rows (if backend importable)
try:
    gold = None
    for dom in ["battery","lexmark","viessmann"]:
        p = Path(f"tests/{dom}/test.jsonl")
        if p.exists():
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        gold = line.strip()
                        break
            if gold: break
    if gold:
        sections.append(add_cmd("GOLD SAMPLE", str(p), gold))
except Exception as e:
    sections.append(add_cmd("GOLD SAMPLE", "error", str(e)))

try:
    sys.path.insert(0, str(ROOT))
    from backend.retrieval.hybrid import HybridRetriever
    r = HybridRetriever("backend/corpus/dpp_corpus.jsonl")
    rows = r.search("EN 62133-2 tests passed")
    import json as _json
    sections.append(add_cmd("RETRIEVER ROWS", "search('EN 62133-2 tests passed')", 
        "\n".join(_json.dumps(x, ensure_ascii=False) for x in rows[:5])))
except Exception as e:
    sections.append(add_cmd("RETRIEVER ROWS", "error", str(e)))

OUT.write_text("\n".join(sections), encoding="utf-8")
print(f"[OK] wrote {OUT} ({OUT.stat().st_size} bytes)")

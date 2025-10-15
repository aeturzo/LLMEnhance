# tools/snapshot_repo.py
# Create a readable single-file snapshot for debugging/review.
import os, glob, json, textwrap
from pathlib import Path

ROOT = Path(".").resolve()
OUT  = Path("repo_snapshot.txt")

def add_title(t): return f"\n===== {t} =====\n"
def add_file(p):
    try: s = Path(p).read_text(encoding="utf-8")
    except Exception as e: s = f"[ERR reading {p}: {e}]"
    return f"\n----- BEGIN FILE: {p} -----\n{s}\n----- END FILE: {p} -----\n"

def grep(pattern):
    out=[]
    for p in ROOT.rglob("*.py"):
        if any(x in str(p) for x in [".git","__pycache__","site-packages",".venv","env","build"]):
            continue
        try:
            for i,l in enumerate(p.read_text(encoding="utf-8").splitlines(),1):
                if pattern in l:
                    out.append(f"{p}:{i}:{l}")
        except: pass
    return "\n".join(out) or "[no matches]"

sections=[]

# 0) tree (top 2 levels)
tree=[]
for d,dirs,files in os.walk(".", topdown=True):
    drel=os.path.relpath(d,".")
    if drel.count(os.sep)>2: dirs[:]=[]; continue
    tree.append(drel+"/")
    for f in files: tree.append(os.path.join(drel,f))
sections.append(add_title("TREE (top 2 levels)") + "\n".join(sorted(tree)))

# 1) greps to locate exporter/join and paper tables
sections.append(add_title("GREP eval_summary") + grep("eval_summary"))
sections.append(add_title("GREP paper tables") + grep("docs/paper/tables"))
sections.append(add_title("GREP joined/accept") +
               "\n\n".join([grep("eval_joined"), grep("Acceptance OK"), grep(" join("), grep("accept")]))

# 2) key scripts if present
for f in ["scripts/run_paper_pipeline.py","run_eval_all.py","scripts/build_corpus.py",
          "backend/retrieval/hybrid.py","tools/make_summary_from_joined.py","scripts/gen_synth.py"]:
    if Path(f).exists(): sections.append(add_file(f))

# 3) include files the greps pointed at
hits=set()
for block in sections[-3].splitlines():
    if ":" in block and block.endswith(".py"):
        hits.add(block.split(":",1)[0])
for h in sorted(hits):
    if Path(h).exists():
        sections.append(add_file(h))

# 4) artifact sizes + small samples
def wc(globpat):
    lines=[]
    for p in sorted(glob.glob(globpat)):
        try: n=sum(1 for _ in open(p,encoding="utf-8")); lines.append(f"{n:7d}  {p}")
        except Exception as e: lines.append(f"[ERR] {p}: {e}")
    return "\n".join(lines) or f"[no files match {globpat}]"

sections.append(add_title("WC artifacts/eval_*_*.csv") + wc("artifacts/eval_*_*.csv"))
sections.append(add_title("WC artifacts/eval_joined_*.csv") + wc("artifacts/eval_joined_*.csv"))
sections.append(add_title("WC docs/paper/tables/*") + wc("docs/paper/tables/*"))

# sample joined (first + last few lines) without pulling huge data
joined = sorted(glob.glob("artifacts/eval_joined_*_calibrated.csv") or glob.glob("artifacts/eval_joined_*.csv"))
if joined:
    p = joined[-1]
    head = "".join(list(open(p,encoding="utf-8"))[:10])
    tail = "".join(list(open(p,encoding="utf-8"))[-10:])
    sections.append(add_title(f"SAMPLE {p} (HEAD)") + head)
    sections.append(add_title(f"SAMPLE {p} (TAIL)") + tail)

OUT.write_text("\n".join(sections), encoding="utf-8")
print(f"[OK] wrote {OUT} ({OUT.stat().st_size} bytes)")

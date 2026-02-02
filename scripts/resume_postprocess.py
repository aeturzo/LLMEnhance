#!/usr/bin/env python3
"""
Resume post-processing (stats/plots/paper assets) from existing artifacts
without deleting or re-running evals.

- Selects the latest eval_joined per domain.
- Pools into a single eval_joined_pooled_<timestamp>.csv under a resume dir.
- Runs compute_stats, calibration, selective risk, sym_coverage, plots,
  and publication checks.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
import subprocess

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def latest(path: Path, pattern: str) -> Path | None:
    files = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def latest_per_domain(artifacts: Path, domains: list[str]) -> dict[str, Path]:
    """Pick newest eval_joined per domain, preferring filenames that contain domain."""
    out: dict[str, Path] = {}
    all_joined = sorted(artifacts.glob("eval_joined_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for dom in domains:
        cand = [p for p in all_joined if dom in p.stem]
        if not cand:
            picked = None
            for p in all_joined:
                try:
                    df = pd.read_csv(p, nrows=50)
                    if "domain" in df.columns and dom in set(df["domain"].dropna().unique()):
                        picked = p
                        break
                except Exception:
                    continue
            if picked:
                cand = [picked]
        if cand:
            out[dom] = cand[0]
    return out


def plot_risk_coverage(rc_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(rc_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for mode, g in df.groupby("mode"):
        g_sorted = g[g["domain"] == "pooled"].sort_values("coverage") if "domain" in g.columns else g.sort_values("coverage")
        plt.plot(g_sorted["coverage"], g_sorted["risk"], label=mode)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1-accuracy)")
    plt.title("Risk-Coverage (pooled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "risk_coverage.png")
    plt.close()


def plot_reliability(joined_path: Path, out_dir: Path, bins: int = 10) -> None:
    df = pd.read_csv(joined_path)
    conf = pd.to_numeric(df.get("confidence_cal", df.get("confidence", 0.0)), errors="coerce")
    suc = df["correct"].astype(int)
    bins_edges = pd.interval_range(start=0, end=1, periods=bins)
    dfb = pd.DataFrame({"conf": conf, "suc": suc})
    dfb["bin"] = pd.cut(dfb["conf"].fillna(0.0), bins_edges, include_lowest=True)
    gb = dfb.groupby("bin")
    x = [i.mid for i in gb["conf"].agg("mean").index]
    y = gb["suc"].mean().fillna(0)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.plot(x, y, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "reliability.png")
    plt.close()


def plot_memory_scaling(acc_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(acc_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    mem_modes = ["MEM", "MEM_ONLY", "MEMSYM"]
    for mode in mem_modes:
        g = df[(df["mode"] == mode) & (df["type"] == "all") & (df["domain"] != "pooled")]
        if g.empty:
            continue
        plt.bar(g["domain"] + f" ({mode})", g["acc"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Memory scaling (per domain)")
    plt.tight_layout()
    plt.savefig(out_dir / "memory_scaling.png")
    plt.close()


def plot_stub(title: str, filename: Path):
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    plt.text(0.5, 0.5, f"{title}\n(no data)", ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["battery", "lexmark", "viessmann"])
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--docs", default=str(Path("docs") / "paper"))
    ap.add_argument("--allow_partial", action="store_true")
    args = ap.parse_args()

    artifacts = (ROOT / args.artifacts).resolve()
    tables = (ROOT / args.tables).resolve()
    figures = (ROOT / args.figures).resolve()
    docs = (ROOT / args.docs).resolve()
    docs_fig = docs / "figures"
    docs_tab = docs / "tables"

    picked = latest_per_domain(artifacts, args.domains)
    missing = [d for d in args.domains if d not in picked]
    if missing and not args.allow_partial:
        raise SystemExit(f"Missing eval_joined for domains: {missing}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    resume_dir = artifacts / f"_resume_{ts}"
    resume_dir.mkdir(parents=True, exist_ok=True)

    # Pool latest per-domain joined
    dfs = []
    for dom, fp in picked.items():
        print(f"Using {dom}: {fp}")
        dfs.append(pd.read_csv(fp))
    if not dfs:
        raise SystemExit("No eval_joined files found to pool.")

    pooled = resume_dir / f"eval_joined_pooled_{ts}.csv"
    pd.concat(dfs).to_csv(pooled, index=False)
    print(f"Wrote {pooled}")

    # Build pooled trace if per-domain traces are available (best effort)
    def _extract_ts(fp: Path) -> str | None:
        name = fp.stem  # eval_joined_YYYYMMDD_HHMMSS
        parts = name.split("_")
        if len(parts) >= 3 and len(parts[-2]) == 8 and len(parts[-1]) == 6:
            return f"{parts[-2]}_{parts[-1]}"
        return None

    trace_paths = []
    for dom, fp in picked.items():
        ts_dom = _extract_ts(fp)
        if ts_dom:
            t = artifacts / f"trace_{ts_dom}.jsonl"
            if t.exists():
                trace_paths.append(t)

    pooled_trace = resume_dir / f"trace_{ts}.jsonl"
    if trace_paths:
        with pooled_trace.open("wb") as out_f:
            for t in trace_paths:
                data = t.read_bytes()
                out_f.write(data)
                if data and not data.endswith(b"\n"):
                    out_f.write(b"\n")
        print(f"Wrote {pooled_trace}")
    else:
        # fallback to newest trace if we cannot match per-domain traces
        trace = latest(artifacts, "trace_*.jsonl")
        if trace:
            shutil.copy2(trace, resume_dir / trace.name)

    # Compute stats and post-processing
    run(["python", str(ROOT / "scripts" / "compute_stats.py"), "--artifacts", str(resume_dir), "--out", str(tables), "--joined", str(pooled)])
    run(["python", str(ROOT / "backend" / "eval" / "report_dataset_counts.py"), "--out", str(tables / "dataset_counts.csv")])
    run(["python", str(ROOT / "backend" / "eval" / "calibrate.py"),
         "--artifacts", str(resume_dir), "--tables", str(tables), "--bins", "10",
         "--joined", str(pooled), "--out", str(resume_dir)])
    run(["python", str(ROOT / "backend" / "eval" / "selective.py"), "--artifacts", str(resume_dir), "--out", str(tables)])
    if pooled_trace.exists():
        run(["python", str(ROOT / "backend" / "eval" / "sym_coverage.py"), "--trace", str(pooled_trace), "--out", str(tables)])
    else:
        run(["python", str(ROOT / "backend" / "eval" / "sym_coverage.py"), "--joined", str(pooled), "--out", str(tables)])

    # Plots
    rc_path = tables / "risk_coverage.csv"
    acc_path = tables / "acc_ci.csv"
    joined_cal = resume_dir / f"{pooled.stem}_calibrated.csv"
    joined_for_plot = joined_cal if joined_cal.exists() else pooled

    if rc_path.exists():
        plot_risk_coverage(rc_path, figures)
    if joined_for_plot.exists():
        plot_reliability(joined_for_plot, figures)
    if acc_path.exists():
        plot_memory_scaling(acc_path, figures)
    else:
        plot_stub("Memory scaling", figures / "memory_scaling.png")
    plot_stub("Robustness (paraphrase/noise)", figures / "robustness.png")

    # Copy assets to docs/paper
    docs_fig.mkdir(parents=True, exist_ok=True)
    docs_tab.mkdir(parents=True, exist_ok=True)
    for f in figures.glob("*.png"):
        (docs_fig / f.name).write_bytes(f.read_bytes())
    for f in tables.glob("*.csv"):
        if f.is_file():
            (docs_tab / f.name).write_bytes(f.read_bytes())

    # Optional LaTeX tables
    for f in ["acc_ci.csv", "risk_coverage.csv", "aurc.csv", "mcnemar.csv", "effect_sizes.csv", "cost_by_mode.csv", "ece_by_mode.csv", "selective_targets.csv", "dataset_counts.csv"]:
        fp = tables / f
        if fp.exists() and fp.is_file():
            df = pd.read_csv(fp)
            tex = df.to_latex(index=False)
            (docs_tab / f.replace(".csv", ".tex")).write_text(tex, encoding="utf-8")

    # Publication checks
    run(["python", str(ROOT / "backend" / "eval" / "publication_checks.py")])

    print("Resume post-processing complete.")


if __name__ == "__main__":
    main()

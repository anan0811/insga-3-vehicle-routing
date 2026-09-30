"""Scalability_test_runner.py — Scalability test: INSGA3 only.

Scales : 50, 80, 100, 150, 200 customers
         1 instance per scale, N_RUNS=3 runs per (instance × satellite).
Sat    : optimal selection from sat∈{1,2,3} → lowest avg cost over N_RUNS runs.
Params : Test A (50 cust)  cp=0.65, mp=0.60
         Test B (≥80 cust) cp=0.80, mp=1.00

Output : Scalability_test_results.TXT  (same directory as this script)
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent           # 論文/experiments/
EXP_DIR    = HERE.parent.parent / "論文實驗"            # My-paper--main/論文實驗/
INSGA3_DIR = HERE.parent / "INSGA-III"                 # 論文/INSGA-III/

OUTPUT_FILE = HERE / "Scalability_test_results.TXT"

# ── Config ─────────────────────────────────────────────────────────────────────
N_RUNS    = 3
N_WORKERS = int(os.getenv("SCALE_WORKERS", "6"))

CP_A, MP_A = 0.65, 0.60   # Test A — 50 customers
CP_B, MP_B = 0.80, 1.00   # Test B — ≥80 customers
SAT_COUNTS = [1, 2, 3]

# ── Instance definitions ───────────────────────────────────────────────────────
# (n_cust, [(label, row_start, row_end), ...])
SCALE_GROUPS = [
    ( 50, [("A1", 3000, 3049)]),
    ( 80, [("B1", 3500, 3579)]),
    (100, [("C1", 4300, 4399)]),
    (150, [("D1", 4600, 4749)]),
    (200, [("E1", 5050, 5249)]),
]


def _params(n_cust: int):
    return (CP_A, MP_A) if n_cust <= 50 else (CP_B, MP_B)


# ── Worker (top-level required for Windows ProcessPoolExecutor) ────────────────

def _worker_insga3(args):
    label, run, rs, re, sat, cp, mp = args
    import sys, random, time, importlib.util
    from pathlib import Path
    _exp = Path(__file__).resolve().parent.parent.parent / "論文實驗"
    _i3  = Path(__file__).resolve().parent.parent / "INSGA-III"
    for p in (str(_exp), str(_i3)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from HNSGA3_vrp import _get_instance_rows_cache
    spec = importlib.util.spec_from_file_location("insga3_mod", _i3 / "insga3.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    random.seed(sum(ord(c) for c in label) * 1000 + run)
    _get_instance_rows_cache()
    m = mod.INSGA3Algorithm()
    m.start_customer_number = rs
    m.end_customer_number   = re
    m.number_satellite      = sat
    m.crossover_probability = cp
    m.mut_prob              = mp
    t0 = time.time()
    m.runMain()
    fe = m.FE_all_cost
    se = m.SE_all_cost
    if fe is None or se is None:
        raise ValueError("no solution")
    return dict(inst=label, run=run, sat=sat,
                cost=fe + se,
                nv=m.all_number_vehicles, co2=m.SE_all_carbon,
                rt=time.time() - t0)


def _avg(lst): return sum(lst) / len(lst) if lst else float("nan")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    from HNSGA3_vrp import _get_instance_rows_cache
    _get_instance_rows_cache()

    # Build all tasks
    all_tasks = []  # (label, run, rs, re, sat, cp, mp, n_cust)
    for n_cust, instances in SCALE_GROUPS:
        cp, mp = _params(n_cust)
        for label, rs, re in instances:
            for sat in SAT_COUNTS:
                for run in range(1, N_RUNS + 1):
                    all_tasks.append((label, run, rs, re, sat, cp, mp, n_cust))

    total = len(all_tasks)
    print(f"\n  Scalability Test (INSGA3) — {total} tasks, {N_WORKERS} workers")
    print(f"  Scales: {[g[0] for g in SCALE_GROUPS]} customers")
    print(f"  1 instance/scale × {len(SAT_COUNTS)} sat counts × {N_RUNS} runs\n")

    # raw[(label, sat)] = [result, ...]
    raw: dict = {}
    wall = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        future_map = {}
        for t in all_tasks:
            label, run, rs, re, sat, cp, mp, _ = t
            f = pool.submit(_worker_insga3, (label, run, rs, re, sat, cp, mp))
            future_map[f] = t

        done = 0
        for fut in as_completed(future_map):
            done += 1
            t = future_map[fut]
            label = t[0]
            try:
                r = fut.result()
                raw.setdefault((label, r["sat"]), []).append(r)
                print(
                    f"  [{done:>3}/{total}] {label}  sat={r['sat']}"
                    f"  run={r['run']}  cost={r['cost']:.1f}  rt={r['rt']:.1f}s"
                    f"  (wall={time.time()-wall:.0f}s)",
                    flush=True,
                )
            except Exception as e:
                print(f"  ERROR {label}: {e}", flush=True)

    # Find optimal satellite per label
    opt: dict[str, int] = {}
    for _, instances in SCALE_GROUPS:
        for label, *_ in instances:
            best_sat, best_cost = None, float("inf")
            for sat in SAT_COUNTS:
                runs = raw.get((label, sat), [])
                if runs:
                    c = _avg([r["cost"] for r in runs])
                    if c < best_cost:
                        best_cost, best_sat = c, sat
            opt[label] = best_sat

    # Output
    sep = "-" * 60
    lines = [
        "Scalability Test Results — INSGA3",
        f"Scales: {[g[0] for g in SCALE_GROUPS]} customers  |  1 instance per scale",
        f"Runs per (instance × satellite): {N_RUNS}  |  Sat: optimal from {{1,2,3}}",
        f"Test A (50 cust): cp={CP_A} mp={MP_A}",
        f"Test B (>=80 cust): cp={CP_B} mp={MP_B}",
        sep,
        f"  {'Cust':>6}  {'Opt-Sat':>7}  {'Avg RT(s)':>10}  {'Avg Cost($)':>12}",
        sep,
    ]

    for n_cust, instances in SCALE_GROUPS:
        for label, *_ in instances:
            sat = opt.get(label)
            if sat is None:
                continue
            runs = raw.get((label, sat), [])
            if runs:
                avg_rt   = _avg([r["rt"]   for r in runs])
                avg_cost = _avg([r["cost"] for r in runs])
                lines.append(
                    f"  {n_cust:>6}  {sat:>7}  {avg_rt:>10.1f}  {avg_cost:>12.1f}"
                )

    lines += [sep, f"Total wall time: {time.time()-wall:.0f}s"]
    text = "\n".join(lines) + "\n"
    OUTPUT_FILE.write_text(text, encoding="utf-8")
    print(f"\n  Results -> {OUTPUT_FILE}")
    print(text.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()

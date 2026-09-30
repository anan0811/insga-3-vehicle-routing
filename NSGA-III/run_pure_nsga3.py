"""run_pure_nsga3.py — Table 6 experiment for Pure NSGA-III.

Pure NSGA-III: Random init + OX crossover + Inversion mutation + NO VNS
               + NSGA-III reference-point niching selection

Uses TK-BASE's optimal satellite counts per instance:
  [2,3,1,2,2,1,1,3,1,2, 2,1,3,1,1,1,2,1,1,3]

Parameters (same as TK-BASE):
  Test A (inst 1-10,  50 cust): cp=0.65, mp=0.60
  Test B (inst 11-20, 80 cust): cp=0.80, mp=1.00

Instances: rows 3000-4299 (50/80 customers)

Output:
  TABLE6_PURE_NSGA3.TXT
"""
from __future__ import annotations

import os
import sys
import time
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE       = Path(__file__).resolve().parent            # 論文/NSGA-III/
_EXP_DIR    = _HERE.parent.parent / "論文實驗"
_INSGA3_DIR = _HERE.parent / "INSGA-III"

for _p in [str(_HERE), str(_EXP_DIR), str(_INSGA3_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

N_RUNS    = 10
N_WORKERS = int(os.getenv("TABLE6_WORKERS", "6"))

# TK-BASE optimal satellite counts (from SAT_SWEEP_TK_BASE.TXT)
OPT_SAT = [2, 3, 1, 2, 2, 1, 1, 3, 1, 2,
           2, 1, 3, 1, 1, 1, 2, 1, 1, 3]

# Same parameters as TK-BASE
PARAMS_TEST_A = (0.65, 0.60)
PARAMS_TEST_B = (0.80, 1.00)

OUTPUT_FILE = _HERE / "Instance_PURE_NSGA3.TXT"
OUTPUT_RAW  = _HERE / "Instance_PURE_NSGA3_RAW.TXT"

INSTANCES = [
    ( 1, 3000, 3049), ( 2, 3050, 3099), ( 3, 3100, 3149),
    ( 4, 3150, 3199), ( 5, 3200, 3249), ( 6, 3250, 3299),
    ( 7, 3300, 3349), ( 8, 3350, 3399), ( 9, 3400, 3449),
    (10, 3450, 3499),
    (11, 3500, 3579), (12, 3580, 3659), (13, 3660, 3739),
    (14, 3740, 3819), (15, 3820, 3899), (16, 3900, 3979),
    (17, 3980, 4059), (18, 4060, 4139), (19, 4140, 4219),
    (20, 4220, 4299),
]


def _worker(args):
    inst_id, run_num, row_start, row_end, n_sat, cp, mp = args
    import random, time, sys
    from pathlib import Path

    _here       = Path(__file__).resolve().parent
    _exp_dir    = _here.parent.parent / "論文實驗"
    _insga3_dir = _here.parent / "INSGA-III"
    for _p in [str(_here), str(_exp_dir), str(_insga3_dir)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from nsga3 import NSGA3Algorithm
    from INSGA3_vrp import _get_instance_rows_cache

    random.seed(inst_id * 1000 + run_num)
    _get_instance_rows_cache()

    m = NSGA3Algorithm()
    m.start_customer_number = row_start
    m.end_customer_number   = row_end
    m.number_satellite      = n_sat
    m.crossover_probability = cp
    m.mut_prob              = mp

    t0 = time.time()
    m.runMain()
    return {
        "inst_id":    inst_id,
        "run_num":    run_num,
        "total_cost": m.FE_all_cost + m.SE_all_cost,
        "vehicles":   m.all_number_vehicles,
        "carbon":     m.SE_all_carbon,
        "elapsed":    time.time() - t0,
    }


def _avg(lst): return sum(lst) / len(lst) if lst else float("nan")
def _std(lst): return statistics.stdev(lst) if len(lst) >= 2 else 0.0


def main():
    from INSGA3_vrp import _get_instance_rows_cache
    _get_instance_rows_cache()

    tasks = []
    for idx, (inst_id, rs, re) in enumerate(INSTANCES):
        n_sat  = OPT_SAT[idx]
        cp, mp = PARAMS_TEST_A if inst_id <= 10 else PARAMS_TEST_B
        for run in range(1, N_RUNS + 1):
            tasks.append((inst_id, run, rs, re, n_sat, cp, mp))

    total = len(tasks)
    print(f"\n  Pure NSGA-III Table 6 — {total} tasks, {N_WORKERS} workers")
    print(f"  Opt sats (TK-BASE): {OPT_SAT}")
    print(f"  Test A (inst 1-10):  cp={PARAMS_TEST_A[0]} mp={PARAMS_TEST_A[1]}")
    print(f"  Test B (inst 11-20): cp={PARAMS_TEST_B[0]} mp={PARAMS_TEST_B[1]}\n")

    raw  = {inst_id: [] for inst_id, *_ in INSTANCES}
    wall = time.time()
    done = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        fm = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(fm):
            done += 1
            try:
                r = fut.result()
                raw[r["inst_id"]].append(r)
                print(
                    f"  [{done:>3}/{total}] inst={r['inst_id']:02d}"
                    f" run={r['run_num']:02d}  cost={r['total_cost']:.1f}"
                    f"  NV={r['vehicles']}  {r['elapsed']:.1f}s"
                    f"  (wall={time.time()-wall:.0f}s)",
                    flush=True,
                )
            except Exception as e:
                t = fm[fut]
                print(f"  ERROR inst={t[0]} run={t[1]}: {e}", flush=True)

    sep = "-" * 72
    lines = [
        "Table 6: Pure NSGA-III Results",
        "Algorithm: Random init + OX crossover + Inversion mutation + NO VNS",
        "Selection: NSGA-III reference-point niching",
        f"Opt sats (TK-BASE): {OPT_SAT}",
        f"Test A (inst 1-10):  cp={PARAMS_TEST_A[0]} mp={PARAMS_TEST_A[1]}",
        f"Test B (inst 11-20): cp={PARAMS_TEST_B[0]} mp={PARAMS_TEST_B[1]}",
        f"Runs per instance: {N_RUNS}",
        sep,
        f"  {'Inst':>5}  {'Sat':>4}  {'Cost($)':>10}  {'Std':>8}  "
        f"{'NV':>4}  {'Carbon':>10}  {'RT(s)':>7}",
        sep,
    ]

    all_costs, all_stds, all_nvs, all_carbons, all_rts = [], [], [], [], []

    for idx, (inst_id, *_) in enumerate(INSTANCES):
        n_sat   = OPT_SAT[idx]
        results = raw.get(inst_id, [])
        if not results:
            lines.append(f"  {inst_id:>5}  {n_sat:>4}  {'NO DATA':>10}")
            continue
        costs   = [r["total_cost"] for r in results]
        nvs     = [r["vehicles"]   for r in results]
        carbons = [r["carbon"]     for r in results]
        rts     = [r["elapsed"]    for r in results]
        avg_c = _avg(costs);  std_c = _std(costs)
        avg_n = _avg(nvs);    avg_co = _avg(carbons);  avg_r = _avg(rts)
        lines.append(
            f"  {inst_id:>5}  {n_sat:>4}  {avg_c:>10.1f}  {std_c:>8.1f}  "
            f"{int(round(avg_n)):>4}  {avg_co:>10.4f}  {avg_r:>7.1f}"
        )
        all_costs.append(avg_c);  all_stds.append(std_c)
        all_nvs.append(avg_n);    all_carbons.append(avg_co);  all_rts.append(avg_r)

    lines += [
        sep,
        f"  {'Avg':>5}       {_avg(all_costs):>10.1f}  {_avg(all_stds):>8.1f}  "
        f"{int(round(_avg(all_nvs))):>4}  {_avg(all_carbons):>10.4f}  {_avg(all_rts):>7.1f}",
        sep,
        f"\nTotal wall time: {time.time()-wall:.0f}s",
    ]

    text = "\n".join(lines) + "\n"
    OUTPUT_FILE.write_text(text, encoding="utf-8")
    print(f"\n  Results -> {OUTPUT_FILE}")
    print(text)

    # Save raw per-run data
    raw_lines = []
    for idx, (inst_id, *_) in enumerate(INSTANCES):
        for r in sorted(raw.get(inst_id, []), key=lambda x: x["run_num"]):
            raw_lines.append(
                f"inst={r['inst_id']:02d} run={r['run_num']:02d}"
                f" cost={r['total_cost']:.1f} NV={r['vehicles']}"
                f" carbon={r['carbon']:.4f} elapsed={r['elapsed']:.1f}s"
            )
    OUTPUT_RAW.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    print(f"  RAW data -> {OUTPUT_RAW}")


if __name__ == "__main__":
    main()

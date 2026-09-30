"""run_pure_nsga2.py — Table 6 experiment for Pure NSGA-II.

Pure NSGA-II: Random init + OX crossover + Inversion mutation + NO VNS
               + NSGA-II crowding-distance selection

Parameters from JHENG (2025) Table 5 NSGA-II sensitivity analysis:
  Test A (inst 1-10,  50 cust, 2 sat): cp=0.85, mp=0.5
  Test B (inst 11-20, 80 cust, 3 sat): cp=0.60, mp=0.6

Output:
  TABLE6_PURE_NSGA2.TXT     — paper-style per-instance summary
  TABLE6_PURE_NSGA2_RAW.TXT — all 200 raw run results
"""

import os
import sys
import time
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE       = Path(__file__).resolve().parent   # 論文/NSGA-II/
_INSGA3_DIR = _HERE.parent / "INSGA-III"
for _p in [str(_HERE), str(_INSGA3_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

N_RUNS    = 10
N_WORKERS = int(os.getenv("TABLE6_WORKERS", "6"))

# JHENG Table 5 optimal parameters for NSGA-II
PARAMS_TEST_A = (0.85, 0.5)
PARAMS_TEST_B = (0.60, 0.6)

OUTPUT_FILE     = _HERE / "Instance_PURE_NSGA2.TXT"
OUTPUT_RAW_FILE = _HERE / "Instance_PURE_NSGA2_RAW.TXT"

INSTANCES = [
    ( 1, 3000, 3049, 2), ( 2, 3050, 3099, 2), ( 3, 3100, 3149, 2),
    ( 4, 3150, 3199, 2), ( 5, 3200, 3249, 2), ( 6, 3250, 3299, 2),
    ( 7, 3300, 3349, 2), ( 8, 3350, 3399, 2), ( 9, 3400, 3449, 2),
    (10, 3450, 3499, 2),
    (11, 3500, 3579, 3), (12, 3580, 3659, 3), (13, 3660, 3739, 3),
    (14, 3740, 3819, 3), (15, 3820, 3899, 3), (16, 3900, 3979, 3),
    (17, 3980, 4059, 3), (18, 4060, 4139, 3), (19, 4140, 4219, 3),
    (20, 4220, 4299, 3),
]


def _worker(args):
    inst_id, run_num, row_start, row_end, n_sat, cp, mp = args
    import random, time, sys
    from pathlib import Path

    _here       = Path(__file__).resolve().parent
    _insga3_dir = _here.parent / "INSGA-III"
    for _p in [str(_here), str(_insga3_dir)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from pure_nsga2 import PureNSGA2Algorithm
    from INSGA3_vrp import _get_instance_rows_cache

    random.seed(inst_id * 1000 + run_num)
    _get_instance_rows_cache()

    m = PureNSGA2Algorithm()
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
    for inst_id, rs, re, ns in INSTANCES:
        cp, mp = PARAMS_TEST_A if inst_id <= 10 else PARAMS_TEST_B
        for run in range(1, N_RUNS + 1):
            tasks.append((inst_id, run, rs, re, ns, cp, mp))

    raw      = {inst_id: [] for inst_id, *_ in INSTANCES}
    raw_lines = []
    wall = time.time()

    print(f"\n  Pure NSGA-II Table 6 — {len(tasks)} tasks, {N_WORKERS} workers")
    print(f"  Test A (inst 1-10):  cp={PARAMS_TEST_A[0]} mp={PARAMS_TEST_A[1]}")
    print(f"  Test B (inst 11-20): cp={PARAMS_TEST_B[0]} mp={PARAMS_TEST_B[1]}\n")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        fm = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(fm):
            try:
                r = fut.result()
                raw[r["inst_id"]].append(r)
                raw_lines.append(
                    f"inst={r['inst_id']:02d} run={r['run_num']:02d} "
                    f"cost={r['total_cost']:.1f} NV={r['vehicles']} "
                    f"carbon={r['carbon']:.4f} elapsed={r['elapsed']:.1f}s"
                )
                print(
                    f"  [inst {r['inst_id']:02d} run {r['run_num']:02d}/{N_RUNS}]"
                    f"  cost={r['total_cost']:.1f}  NV={r['vehicles']}"
                    f"  co2={r['carbon']:.1f}  {r['elapsed']:.1f}s"
                    f"  (wall={time.time()-wall:.0f}s)",
                    flush=True,
                )
            except Exception as e:
                t = fm[fut]
                print(f"  ERROR inst={t[0]} run={t[1]}: {e}", flush=True)
                raw_lines.append(f"inst={t[0]:02d} run={t[1]:02d} ERROR: {e}")

    lines = [
        "Table 6: Pure NSGA-II Results",
        "Algorithm: Random init + OX crossover + Inversion mutation + NO VNS",
        "Selection: NSGA-II crowding-distance",
        f"Test A (inst 1-10)  cp={PARAMS_TEST_A[0]} mp={PARAMS_TEST_A[1]}  "
        f"| Test B (inst 11-20) cp={PARAMS_TEST_B[0]} mp={PARAMS_TEST_B[1]}",
        "10 runs per instance",
        "-" * 72,
        f"  {'Instance':>8}    {'Cost($)':>10}    {'Std':>8}    "
        f"{'NV':>4}    {'Carbon':>10}    {'RT(s)':>7}",
        "-" * 72,
    ]

    all_costs, all_nvs, all_carbons, all_rts, all_stds = [], [], [], [], []

    for inst_id, *_ in INSTANCES:
        results = raw.get(inst_id, [])
        if not results:
            lines.append(f"  {inst_id:>8}    {'NO DATA':>10}")
            continue
        costs   = [r["total_cost"] for r in results]
        nvs     = [r["vehicles"]   for r in results]
        carbons = [r["carbon"]     for r in results]
        times   = [r["elapsed"]    for r in results]

        avg_cost   = _avg(costs)
        std_cost   = _std(costs)
        avg_nv     = _avg(nvs)
        avg_carbon = _avg(carbons)
        avg_rt     = _avg(times)

        lines.append(
            f"  {inst_id:>8}    {avg_cost:>10.1f}    {std_cost:>8.1f}    "
            f"{int(round(avg_nv)):>4}    {avg_carbon:>10.4f}    {avg_rt:>7.1f}"
        )
        all_costs.append(avg_cost)
        all_stds.append(std_cost)
        all_nvs.append(avg_nv)
        all_carbons.append(avg_carbon)
        all_rts.append(avg_rt)

    lines += [
        "-" * 72,
        f"  {'Average':>8}    {_avg(all_costs):>10.1f}    {_avg(all_stds):>8.1f}    "
        f"{int(round(_avg(all_nvs))):>4}    {_avg(all_carbons):>10.4f}    {_avg(all_rts):>7.1f}",
        "-" * 72,
        f"\nTotal wall time: {time.time()-wall:.0f}s",
    ]

    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUTPUT_RAW_FILE.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    print(f"\nResults → {OUTPUT_FILE}")
    for l in lines:
        print(l)


if __name__ == "__main__":
    main()

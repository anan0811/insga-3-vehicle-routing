"""run_pure_nsga2_jheng_inst.py — Pure NSGA-II on JHENG's original instances.

Instance definition matches JHENG (2025):
  rows 0-1999, 100 customers per instance, 20 instances
  Satellite count per instance from JHENG's FAST_RUN_SATELLITES[0:20]

Parameters from JHENG Table 5 (NSGA-II sensitivity):
  Test A (inst 1-10):  cp=0.85, mp=0.5
  Test B (inst 11-20): cp=0.60, mp=0.6
"""

import os
import sys
import time
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE       = Path(__file__).resolve().parent          # 論文/NSGA-II/
_INSGA3_DIR = _HERE.parent / "INSGA-III"
for _p in [str(_HERE), str(_INSGA3_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

N_RUNS    = 10
N_WORKERS = int(os.getenv("TABLE6_WORKERS", "6"))

PARAMS_TEST_A = (0.85, 0.5)   # JHENG Table 5 optimal for NSGA-II, inst 1-10
PARAMS_TEST_B = (0.60, 0.6)   # JHENG Table 5 optimal for NSGA-II, inst 11-20

OUTPUT_FILE     = _HERE / "TABLE6_PURE_NSGA2_JHENG_INST.TXT"
OUTPUT_RAW_FILE = _HERE / "TABLE6_PURE_NSGA2_JHENG_INST_RAW.TXT"

# JHENG FAST_RUN_SATELLITES[0:20]
_SAT = [2, 3, 2, 1, 1, 3, 3, 2, 1, 2,
        3, 3, 3, 1, 1, 3, 3, 1, 3, 2]

# Each instance: (inst_id, row_start, row_end, n_satellites)
INSTANCES = [
    (i, (i - 1) * 100, i * 100 - 1, _SAT[i - 1])
    for i in range(1, 21)
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

    raw       = {inst_id: [] for inst_id, *_ in INSTANCES}
    raw_lines = []
    wall      = time.time()

    print(f"\n  Pure NSGA-II — JHENG instances — {len(tasks)} tasks, {N_WORKERS} workers")
    print(f"  Inst 1-20: rows 0-1999, 100 customers each")
    print(f"  Satellites per inst: {_SAT}")
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
        "Table 6: Pure NSGA-II Results — JHENG Instances",
        "Algorithm: Random init + OX crossover + Inversion mutation + NO VNS",
        "Selection: NSGA-II crowding-distance",
        "Instances: rows 0-1999, 100 customers each (same as JHENG 2025)",
        f"Satellites: {_SAT[:10]} (inst 1-10)",
        f"           {_SAT[10:]} (inst 11-20)",
        f"Test A (inst 1-10)  cp={PARAMS_TEST_A[0]} mp={PARAMS_TEST_A[1]}  "
        f"| Test B (inst 11-20) cp={PARAMS_TEST_B[0]} mp={PARAMS_TEST_B[1]}",
        "10 runs per instance",
        "-" * 72,
        f"  {'Instance':>8}  {'Sat':>3}  {'Cost($)':>10}  {'Std':>8}  "
        f"{'NV':>4}  {'Carbon':>10}  {'RT(s)':>7}",
        "-" * 72,
    ]

    all_costs, all_nvs, all_carbons, all_rts, all_stds = [], [], [], [], []

    for inst_id, rs, re, ns in INSTANCES:
        results = raw.get(inst_id, [])
        if not results:
            lines.append(f"  {inst_id:>8}  {ns:>3}  {'NO DATA':>10}")
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
            f"  {inst_id:>8}  {ns:>3}  {avg_cost:>10.1f}  {std_cost:>8.1f}  "
            f"{int(round(avg_nv)):>4}  {avg_carbon:>10.4f}  {avg_rt:>7.1f}"
        )
        all_costs.append(avg_cost)
        all_stds.append(std_cost)
        all_nvs.append(avg_nv)
        all_carbons.append(avg_carbon)
        all_rts.append(avg_rt)

    lines += [
        "-" * 72,
        f"  {'Average':>8}       {_avg(all_costs):>10.1f}  {_avg(all_stds):>8.1f}  "
        f"{int(round(_avg(all_nvs))):>4}  {_avg(all_carbons):>10.4f}  {_avg(all_rts):>7.1f}",
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

"""run_nsga2_table6.py — Table 6 experiment for NSGA-II (Variant A).

NSGA-II uses identical operators to HNSGA-III (NN init, OBX crossover,
hybrid mutation, 2-opt, stagnation-triggered VNS); only the environmental
selection is replaced: NSGA-III niching → NSGA-II crowding distance.

Best parameters reuse HNSGA-III Table-5 results:
  Test A (inst 1-10,  50 cust, 2 sat): cp=0.65, mp=0.6
  Test B (inst 11-20, 80 cust, 3 sat): cp=0.80, mp=0.5

Output files (this directory):
  TABLE6_NSGA2.TXT   — paper-style per-instance summary
  TABLE6_NSGA2_RAW.TXT — all 200 raw run results
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent          # 論文/NSGA-II/
_NSGA_VRP   = _HERE.parent.parent / "nsga_vrp"
_INSGA3_DIR = _HERE.parent / "INSGA-III"
for _p in [str(_NSGA_VRP), str(_INSGA3_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Configuration ─────────────────────────────────────────────────────────────
N_RUNS    = 10
N_WORKERS = int(os.getenv("TABLE6_WORKERS", "6"))

PARAMS_TEST_A = (0.65, 0.6)   # best from HNSGA-III Table-5, Test A
PARAMS_TEST_B = (0.80, 0.5)   # best from HNSGA-III Table-5, Test B

OUTPUT_FILE     = _HERE / "TABLE6_NSGA2.TXT"
OUTPUT_RAW_FILE = _HERE / "TABLE6_NSGA2_RAW.TXT"

# ── Instance definitions ──────────────────────────────────────────────────────
INSTANCES = [
    # Instances 1-10: 50 customers, 2 satellites
    ( 1, 3000, 3049, 2),
    ( 2, 3050, 3099, 2),
    ( 3, 3100, 3149, 2),
    ( 4, 3150, 3199, 2),
    ( 5, 3200, 3249, 2),
    ( 6, 3250, 3299, 2),
    ( 7, 3300, 3349, 2),
    ( 8, 3350, 3399, 2),
    ( 9, 3400, 3449, 2),
    (10, 3450, 3499, 2),
    # Instances 11-20: 80 customers, 3 satellites
    (11, 3500, 3579, 3),
    (12, 3580, 3659, 3),
    (13, 3660, 3739, 3),
    (14, 3740, 3819, 3),
    (15, 3820, 3899, 3),
    (16, 3900, 3979, 3),
    (17, 3980, 4059, 3),
    (18, 4060, 4139, 3),
    (19, 4140, 4219, 3),
    (20, 4220, 4299, 3),
]


# ── Worker (must be top-level for ProcessPoolExecutor on Windows) ─────────────
def _worker_task(args):
    inst_id, run_num, row_start, row_end, n_satellites, cp, mp = args

    import sys, random, time
    from pathlib import Path
    _here = Path(__file__).resolve().parent
    _nsga_vrp   = _here.parent.parent / "nsga_vrp"
    _insga3_dir = _here.parent / "INSGA-III"
    for _p in [str(_nsga_vrp), str(_insga3_dir)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from nsga2 import NSGA2Algorithm
    from INSGA3_vrp import _get_instance_rows_cache

    random.seed(inst_id * 1000 + run_num)
    _get_instance_rows_cache()

    model = NSGA2Algorithm()
    model.start_customer_number = row_start
    model.end_customer_number   = row_end
    model.number_satellite      = n_satellites
    model.crossover_probability = cp
    model.mut_prob              = mp

    t0 = time.time()
    model.runMain()
    elapsed = time.time() - t0

    total_cost = model.FE_all_cost + model.SE_all_cost
    return {
        "inst_id":    inst_id,
        "run_num":    run_num,
        "total_cost": total_cost,
        "fe_cost":    model.FE_all_cost,
        "se_cost":    model.SE_all_cost,
        "vehicles":   model.all_number_vehicles,
        "carbon":     model.SE_all_carbon,
        "elapsed":    elapsed,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
def _avg(lst):
    return sum(lst) / len(lst) if lst else float("nan")

def _std(lst):
    if len(lst) < 2:
        return 0.0
    import statistics
    return statistics.stdev(lst)

def _params_for(inst_id):
    return PARAMS_TEST_A if inst_id <= 10 else PARAMS_TEST_B


# ── Main experiment ───────────────────────────────────────────────────────────
def main():
    from INSGA3_vrp import _get_instance_rows_cache
    _get_instance_rows_cache()

    all_tasks = []
    for inst_id, row_start, row_end, n_sat in INSTANCES:
        cp, mp = _params_for(inst_id)
        for run in range(1, N_RUNS + 1):
            all_tasks.append((inst_id, run, row_start, row_end, n_sat, cp, mp))

    total_tasks = len(all_tasks)
    wall_start  = time.time()
    raw: dict[int, list] = {inst_id: [] for inst_id, *_ in INSTANCES}
    raw_lines = []

    print(f"\n  NSGA-II Table 6 — {total_tasks} runs, {N_WORKERS} workers\n", flush=True)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        future_map = {pool.submit(_worker_task, t): t for t in all_tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                r = future.result()
                raw[r["inst_id"]].append(r)
                raw_lines.append(
                    f"inst={r['inst_id']:02d} run={r['run_num']:02d} "
                    f"cost={r['total_cost']:.1f} NV={r['vehicles']} "
                    f"carbon={r['carbon']:.4f} elapsed={r['elapsed']:.1f}s"
                )
                print(
                    f"  [inst {r['inst_id']:02d} run {r['run_num']:02d}/{N_RUNS}]"
                    f"  cost={r['total_cost']:.1f}  NV={r['vehicles']}"
                    f"  {r['elapsed']:.1f}s"
                    f"  (wall={time.time()-wall_start:.0f}s)",
                    flush=True,
                )
            except Exception as exc:
                inst_id_t, run_num_t = task[0], task[1]
                print(f"  ERROR inst={inst_id_t} run={run_num_t}: {exc}", flush=True)
                raw_lines.append(f"inst={inst_id_t:02d} run={run_num_t:02d} ERROR: {exc}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    header = (
        f"Table 6: NSGA-II (Variant A) Results\n"
        f"Operators: NN init + OBX crossover + hybrid mutation + 2-opt + VNS\n"
        f"Selection: NSGA-II crowding-distance (replaces NSGA-III niching)\n"
        f"Test A (inst 1-10)  cp={PARAMS_TEST_A[0]} mp={PARAMS_TEST_A[1]}\n"
        f"Test B (inst 11-20) cp={PARAMS_TEST_B[0]} mp={PARAMS_TEST_B[1]}\n"
        f"10 runs per instance\n"
    )
    lines = [header, "-" * 70]
    lines.append(
        f"  {'Instance':>8}    {'Cost($)':>10}    {'Std':>8}    "
        f"{'NV':>4}    {'Carbon':>10}    {'RT(s)':>7}"
    )
    lines.append("-" * 70)

    all_costs, all_nvs, all_carbons, all_rts = [], [], [], []
    table_rows = []

    for inst_id, row_start, row_end, n_sat in INSTANCES:
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
        all_nvs.append(avg_nv)
        all_carbons.append(avg_carbon)
        all_rts.append(avg_rt)
        table_rows.append((inst_id, avg_cost, std_cost, avg_nv, avg_carbon, avg_rt))

    lines.append("-" * 70)
    if all_costs:
        lines.append(
            f"  {'Average':>8}    {_avg(all_costs):>10.1f}    {_avg([r[2] for r in table_rows]):>8.1f}    "
            f"{int(round(_avg(all_nvs))):>4}    {_avg(all_carbons):>10.4f}    {_avg(all_rts):>7.1f}"
        )
    lines.append("-" * 70)
    lines.append(f"\nTotal wall time: {time.time()-wall_start:.0f}s")

    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUTPUT_RAW_FILE.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    print(f"\nResults → {OUTPUT_FILE}")
    print(f"Raw data → {OUTPUT_RAW_FILE}")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()

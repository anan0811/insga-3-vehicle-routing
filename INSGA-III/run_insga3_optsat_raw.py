"""run_insga3_optsat_raw.py — INSGA3 with per-instance optimal satellites.

Optimal satellite counts (from satellite sweep / TK-BASE results):
  [2, 3, 1, 2, 2, 1, 1, 3, 1, 2, 2, 1, 3, 1, 1, 1, 2, 1, 1, 3]

Parameters:
  Test A (inst 1-10):  cp=0.65, mp=0.60
  Test B (inst 11-20): cp=0.80, mp=1.00
  10 runs per instance, 6 parallel workers

Output:
  INSGA3_optsat.TXT       (avg/std table)
  INSGA3_optsat_RAW.TXT   (per-run raw data)
"""
from __future__ import annotations

import os
import sys
import time
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE    = Path(__file__).resolve().parent          # 論文/INSGA-III/

N_RUNS    = 10
N_WORKERS = int(os.getenv("INSGA3_WORKERS", "6"))

CP_A, MP_A = 0.65, 0.60
CP_B, MP_B = 0.80, 1.00

OPT_SAT = [2, 3, 1, 2, 2, 1, 1, 3, 1, 2,
           2, 1, 3, 1, 1, 1, 2, 1, 1, 3]

OUTPUT_AVG = HERE / "INSGA3_optsat.TXT"
OUTPUT_RAW = HERE / "INSGA3_optsat_RAW.TXT"

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
    import sys, random, time
    from pathlib import Path

    _here    = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    from INSGA3_vrp import INSGA3Algorithm, _get_instance_rows_cache

    random.seed(inst_id * 1000 + run_num)
    _get_instance_rows_cache()

    m = INSGA3Algorithm()
    m.start_customer_number = row_start
    m.end_customer_number   = row_end
    m.number_satellite      = n_sat
    m.crossover_probability = cp
    m.mut_prob              = mp

    t0 = time.time()
    m.runMain()
    return {
        "inst_id": inst_id, "run_num": run_num,
        "cost":    m.FE_all_cost + m.SE_all_cost,
        "nv":      m.all_number_vehicles,
        "carbon":  m.SE_all_carbon,
        "elapsed": time.time() - t0,
    }


def _avg(lst): return sum(lst) / len(lst) if lst else float("nan")
def _std(lst):
    if len(lst) < 2: return 0.0
    mu = _avg(lst)
    return math.sqrt(sum((x - mu) ** 2 for x in lst) / (len(lst) - 1))


def main():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from INSGA3_vrp import _get_instance_rows_cache
    _get_instance_rows_cache()

    tasks = []
    for idx, (inst_id, rs, re) in enumerate(INSTANCES):
        n_sat  = OPT_SAT[idx]
        cp, mp = (CP_A, MP_A) if inst_id <= 10 else (CP_B, MP_B)
        for run in range(1, N_RUNS + 1):
            tasks.append((inst_id, run, rs, re, n_sat, cp, mp))

    total = len(tasks)
    print(f"\n  INSGA3 (optimal sat) — {total} tasks, {N_WORKERS} workers")
    print(f"  Opt sats: {OPT_SAT}")
    print(f"  Test A: cp={CP_A} mp={MP_A}  |  Test B: cp={CP_B} mp={MP_B}\n")

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
                    f" run={r['run_num']:02d}  cost={r['cost']:.1f}"
                    f"  nv={r['nv']}  co2={r['carbon']:.4f}"
                    f"  {r['elapsed']:.1f}s  (wall={time.time()-wall:.0f}s)",
                    flush=True,
                )
            except Exception as e:
                t = fm[fut]
                print(f"  ERROR inst={t[0]} run={t[1]}: {e}", flush=True)

    sep = "-" * 72
    lines = [
        "INSGA3 Results (Optimal Satellite Counts)",
        f"Optimal satellites: {OPT_SAT}",
        f"Test A (inst 1-10):  cp={CP_A} mp={MP_A}",
        f"Test B (inst 11-20): cp={CP_B} mp={MP_B}",
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
        costs   = [r["cost"]    for r in results]
        nvs     = [r["nv"]      for r in results]
        carbons = [r["carbon"]  for r in results]
        rts     = [r["elapsed"] for r in results]
        a, s   = _avg(costs), _std(costs)
        nv_avg = round(_avg(nvs))
        c, rt  = _avg(carbons), _avg(rts)
        lines.append(
            f"  {inst_id:>5}  {n_sat:>4}  {a:>10.1f}  {s:>8.1f}"
            f"  {nv_avg:>4}  {c:>10.4f}  {rt:>7.1f}"
        )
        all_costs.append(a); all_stds.append(s)
        all_nvs.append(nv_avg); all_carbons.append(c); all_rts.append(rt)

    lines += [
        sep,
        f"  {'Avg':>5}       {_avg(all_costs):>10.1f}  {_avg(all_stds):>8.1f}"
        f"  {round(_avg(all_nvs)):>4}  {_avg(all_carbons):>10.4f}  {_avg(all_rts):>7.1f}",
        sep,
        f"\nTotal wall time: {time.time()-wall:.0f}s",
    ]

    OUTPUT_AVG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Avg results -> {OUTPUT_AVG}")
    print("\n".join(lines))

    # Save raw per-run data
    raw_lines = []
    for idx, (inst_id, *_) in enumerate(INSTANCES):
        for r in sorted(raw.get(inst_id, []), key=lambda x: x["run_num"]):
            raw_lines.append(
                f"inst={r['inst_id']:02d} run={r['run_num']:02d}"
                f" cost={r['cost']:.1f} NV={r['nv']}"
                f" carbon={r['carbon']:.4f} elapsed={r['elapsed']:.1f}s"
            )
    OUTPUT_RAW.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    print(f"  RAW data  -> {OUTPUT_RAW}")


if __name__ == "__main__":
    main()

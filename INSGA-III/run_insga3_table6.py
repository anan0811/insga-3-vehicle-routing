"""run_insga3_table6.py — INSGA3 實驗，輸出 Table 6 格式 (Avg/Std/NV/CO2/RT)。"""
import os, sys, time, math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE    = Path(__file__).resolve().parent                          # 論文/INSGA-III/

N_RUNS    = 10
N_WORKERS = int(os.getenv("INSGA3_WORKERS", "12"))
CP_A, MP_A = 0.65, 0.60   # Test A (inst 1-10)
CP_B, MP_B = 0.80, 1.00   # Test B (inst 11-20)
OUTPUT_FILE = HERE / "INSGA3_instance.TXT"

INSTANCES = []
for i in range(1, 11):
    start = 3000 + (i - 1) * 50
    INSTANCES.append((i, start, start + 49, 2))
for i in range(11, 21):
    start = 3500 + (i - 11) * 80
    INSTANCES.append((i, start, start + 79, 3))


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
    for inst_id, rs, re, ns in INSTANCES:
        cp, mp = (CP_A, MP_A) if inst_id <= 10 else (CP_B, MP_B)
        for run in range(1, N_RUNS + 1):
            tasks.append((inst_id, run, rs, re, ns, cp, mp))

    raw  = {inst_id: [] for inst_id, *_ in INSTANCES}
    wall = time.time()
    print(f"\n  INSGA3 Instance — {len(tasks)} tasks, {N_WORKERS} workers\n")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        fm = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(fm):
            try:
                r = fut.result()
                raw[r["inst_id"]].append(r)
                print(f"  [inst {r['inst_id']:02d} run {r['run_num']:02d}/{N_RUNS}]"
                      f"  cost={r['cost']:.1f}  nv={r['nv']}  co2={r['carbon']:.4f}"
                      f"  {r['elapsed']:.1f}s  (wall={time.time()-wall:.0f}s)", flush=True)
            except Exception as e:
                t = fm[fut]
                print(f"  ERROR inst={t[0]} run={t[1]}: {e}")

    sep  = "-" * 72
    hdr  = f"{'Inst':>6}  {'Sat':>3}  {'Cost($)':>10}  {'Std':>8}  {'NV':>4}  {'Carbon':>10}  {'RT(s)':>6}"
    lines = [
        "INSGA3 Instance Results (Fixed Satellite Counts)",
        f"Algorithm: Triple-Init NSGA-III (Chaos+KMedoids+TS-KMedoids, no local search)",
        f"Fixed sats: inst 1-10 = sat 2, inst 11-20 = sat 3",
        f"Test A (inst 1-10):  cp={CP_A} mp={MP_A}",
        f"Test B (inst 11-20): cp={CP_B} mp={MP_B}",
        f"Runs per instance: {N_RUNS}",
        sep, hdr, sep,
    ]
    all_cost, all_std, all_nv, all_co2, all_rt = [], [], [], [], []

    for inst_id, _, _, n_sat in INSTANCES:
        results = raw.get(inst_id, [])
        if not results:
            continue
        costs  = [r["cost"]    for r in results]
        nvs    = [r["nv"]      for r in results]
        co2s   = [r["carbon"]  for r in results]
        rts    = [r["elapsed"] for r in results]
        a, s   = _avg(costs), _std(costs)
        nv_avg = round(_avg(nvs))
        c, rt  = _avg(co2s), _avg(rts)
        lines.append(f"{inst_id:>6}  {n_sat:>3}  {a:>10.1f}  {s:>8.1f}  {nv_avg:>4}  {c:>10.4f}  {rt:>6.1f}")
        all_cost.append(a); all_std.append(s)
        all_nv.append(nv_avg); all_co2.append(c); all_rt.append(rt)

    lines += [
        sep,
        f"{'Avg':>6}  {'':>3}  {_avg(all_cost):>10.1f}  {_avg(all_std):>8.1f}"
        f"  {round(_avg(all_nv)):>4}  {_avg(all_co2):>10.4f}  {_avg(all_rt):>6.1f}",
        sep,
    ]
    total_wall = time.time() - wall
    lines.append(f"\nTotal wall time: {total_wall:.0f}s")

    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nResults → {OUTPUT_FILE}")
    for l in lines:
        print(l)


if __name__ == "__main__":
    main()

"""run_case_study_insga3.py — 30-day real-world case study using INSGA3.

Uses INSGA3Algorithm (論文/INSGA-III/insga3.py) — NOT the old HNSGA3.

Customer data mapping (real_data.xlsx):
  Day  1 → rows   0– 99  (ID   1–100)
  Day  2 → rows 100–199  (ID 101–200)
  ...
  Day 30 → rows 2900–2999 (ID 2901–3000)

Parameters (from INSGA3 benchmark sensitivity analysis):
  Days  1–10: cp=0.65, mp=0.60  (Test A — consistent with benchmark inst 1–10)
  Days 11–30: cp=0.80, mp=1.00  (Test B — consistent with benchmark inst 11–20)

Output (all in same directory as this script):
  TABLE13_INSGA3.TXT         Paper-style table (Day | Sat=1 | Sat=2 | Sat=3 | Best)
  TABLE13_FULL_INSGA3.CSV    Per-day × per-satellite: cost_avg, cost_std, nv_avg,
                             co2_avg, co2_std, fe_avg, se_avg, rt_avg  (N_RUNS runs)
  TABLE13_RAW_INSGA3.CSV     Per-run raw data (all individual runs)
  routes/day{d:02d}_sat{s}_run{r:02d}.json  Route/solution data per run
  progress.json              Checkpoint for auto-resume on restart
  summary.txt                30-day totals and dynamic-vs-fixed comparison

Additional outputs (#7 extras):
  - FE cost / SE cost / carbon per satellite per day
  - Satellite coordinates (cluster centres) per run
  - Customer group assignments per satellite per run
  - Best SE chromosome per satellite × period (morning/afternoon)
  - 30-day dynamic-optimal vs fixed-satellite total cost comparison
  - Standard deviation across runs for cost and carbon
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
HERE     = Path(__file__).resolve().parent          # 論文/實驗/Case test/
ROOT     = HERE.parent.parent.parent                # My-paper--main/
INSGA3_DIR = ROOT / "論文" / "INSGA-III"
EXP_DIR    = ROOT / "論文實驗"

for p in (str(INSGA3_DIR), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Config ────────────────────────────────────────────────────────────────────
N_RUNS      = 3          # runs per (day, satellite) — for avg ± std
N_WORKERS   = int(os.getenv("CASE_WORKERS", "3"))
SAT_VALUES  = [1, 2, 3]
N_DAYS      = 30

CP_A, MP_A  = 0.65, 0.60   # days  1–10
CP_B, MP_B  = 0.80, 1.00   # days 11–30

OUTPUT_DIR    = HERE
ROUTES_DIR    = HERE / "routes"
PROGRESS_FILE = HERE / "progress.json"
TXT_FILE      = HERE / "TABLE13_INSGA3.TXT"
FULL_CSV      = HERE / "TABLE13_FULL_INSGA3.CSV"
RAW_CSV       = HERE / "TABLE13_RAW_INSGA3.CSV"
SUMMARY_FILE  = HERE / "summary.txt"

ROUTES_DIR.mkdir(parents=True, exist_ok=True)


# ── Parameter selection ───────────────────────────────────────────────────────
def _params(day: int) -> tuple[float, float]:
    return (CP_A, MP_A) if day <= 10 else (CP_B, MP_B)


# ── Progress helpers ──────────────────────────────────────────────────────────
def _load_progress() -> dict:
    """Load checkpoint. Structure: {day: {sat: [run_result, ...]}}"""
    if PROGRESS_FILE.exists():
        raw = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        return {int(d): {int(s): v for s, v in sd.items()} for d, sd in raw.items()}
    return {}


def _save_progress(prog: dict) -> None:
    PROGRESS_FILE.write_text(
        json.dumps(prog, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Worker (top-level for pickling) ──────────────────────────────────────────
def _worker(args):
    """
    Run one INSGA3 trial. Returns a result dict including route data.
    Must be top-level for ProcessPoolExecutor pickling.
    """
    day, n_sat, run_num, cp, mp = args

    import sys, random, time, importlib.util
    from pathlib import Path

    _root     = Path(__file__).resolve().parent.parent.parent.parent
    _insga3   = _root / "論文" / "INSGA-III"
    _exp      = _root / "論文實驗"
    for p in (str(_insga3), str(_exp)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from HNSGA3_vrp import _get_instance_rows_cache
    _get_instance_rows_cache()

    spec = importlib.util.spec_from_file_location("insga3", _insga3 / "insga3.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    random.seed(day * 10000 + n_sat * 100 + run_num)

    row_start = (day - 1) * 100
    row_end   = row_start + 99

    m = mod.INSGA3Algorithm()
    m.start_customer_number = row_start
    m.end_customer_number   = row_end
    m.number_satellite      = n_sat
    m.crossover_probability = cp
    m.mut_prob              = mp

    t0 = time.time()
    m.runMain()
    elapsed = time.time() - t0

    # ── Extract route/solution data ───────────────────────────────────────────
    def _best_chromosome(pop):
        """Return best chromosome from a population, or None."""
        if not pop:
            return None
        best = m._safe_select(pop)
        if best is None:
            return None
        chrom = best[0]
        return list(chrom) if hasattr(chrom, "__iter__") else None

    routes = {
        "day":           day,
        "n_sat":         n_sat,
        "run":           run_num,
        "satellite_coords": [list(c) for c in m.centers] if m.centers is not None else [],
        "customer_group":   [int(g) for g in m.customer_group] if m.customer_group is not None else [],
        "customer_ids":     [int(r[0]) for r in m.json_instance],
        "chromosomes": {
            "sat1_morning":   _best_chromosome(m.all_morning_customer_id_and_fitness1),
            "sat1_afternoon": _best_chromosome(m.all_afternoon_customer_id_and_fitness1),
            "sat2_morning":   _best_chromosome(m.all_morning_customer_id_and_fitness2),
            "sat2_afternoon": _best_chromosome(m.all_afternoon_customer_id_and_fitness2),
            "sat3_morning":   _best_chromosome(m.all_morning_customer_id_and_fitness3),
            "sat3_afternoon": _best_chromosome(m.all_afternoon_customer_id_and_fitness3),
        },
    }

    return {
        "day":      day,
        "n_sat":    n_sat,
        "run":      run_num,
        "cost":     m.FE_all_cost + m.SE_all_cost,
        "fe_cost":  m.FE_all_cost,
        "se_cost":  m.SE_all_cost,
        "nv":       m.all_number_vehicles,
        "carbon":   m.SE_all_carbon,
        "elapsed":  elapsed,
        "routes":   routes,
    }


# ── Statistics helpers ────────────────────────────────────────────────────────
def _avg(lst): return sum(lst) / len(lst) if lst else float("nan")
def _std(lst):
    if len(lst) < 2:
        return 0.0
    mu = _avg(lst)
    return math.sqrt(sum((x - mu) ** 2 for x in lst) / (len(lst) - 1))


# ── Output writers ────────────────────────────────────────────────────────────
def _write_txt(agg: dict) -> None:
    """Write paper-style TABLE13 text file."""
    lines = [
        "Table 13: Optimal Number of Active Satellites per Planning Day (INSGA3)",
        "",
        f"  {'Day':>4}  {'Sat=1 ($)':>12}  {'Sat=2 ($)':>12}  {'Sat=3 ($)':>12}  {'Optimal':>7}",
        "-" * 65,
    ]
    dist = {1: 0, 2: 0, 3: 0}
    total_optimal = 0.0
    totals = {s: 0.0 for s in SAT_VALUES}

    for day in range(1, N_DAYS + 1):
        row = agg.get(day, {})
        costs = {s: _avg(row.get(s, {}).get("costs", [float("nan")])) for s in SAT_VALUES}
        best_sat = min(SAT_VALUES, key=lambda s: costs[s])
        dist[best_sat] += 1
        total_optimal += costs[best_sat]
        for s in SAT_VALUES:
            totals[s] += costs[s]

        parts = []
        for s in SAT_VALUES:
            c = costs[s]
            tag = f"*{c:,.1f}" if s == best_sat else f"{c:,.1f}"
            parts.append(f"{tag:>12}")
        lines.append(f"  {day:>4}  {'  '.join(parts)}  {best_sat:>7}")

    lines += [
        "-" * 65,
        "",
        f"  Distribution: Sat=1: {dist[1]} days ({dist[1]/N_DAYS*100:.0f}%),  "
        f"Sat=2: {dist[2]} days ({dist[2]/N_DAYS*100:.0f}%),  "
        f"Sat=3: {dist[3]} days ({dist[3]/N_DAYS*100:.0f}%)",
        "",
        f"  30-day total (dynamic optimal): ${total_optimal:,.1f}",
        f"  30-day total (fixed Sat=1):     ${totals[1]:,.1f}  "
        f"(+${totals[1]-total_optimal:,.1f}, +{(totals[1]/total_optimal-1)*100:.1f}%)",
        f"  30-day total (fixed Sat=2):     ${totals[2]:,.1f}  "
        f"(+${totals[2]-total_optimal:,.1f}, +{(totals[2]/total_optimal-1)*100:.1f}%)",
        f"  30-day total (fixed Sat=3):     ${totals[3]:,.1f}  "
        f"(+${totals[3]-total_optimal:,.1f}, +{(totals[3]/total_optimal-1)*100:.1f}%)",
        "",
        "  * denotes minimum cost (optimal satellite count) for that day.",
        f"  Each cost = average of {N_RUNS} independent INSGA3 runs.",
    ]
    TXT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_full_csv(agg: dict) -> None:
    """Write full CSV with avg ± std for all metrics per day × satellite."""
    header = (
        "day,n_sat,"
        "cost_avg,cost_std,"
        "fe_avg,fe_std,"
        "se_avg,se_std,"
        "nv_avg,nv_std,"
        "co2_avg,co2_std,"
        "rt_avg,rt_std,"
        "n_runs,is_optimal"
    )
    rows = [header]
    for day in range(1, N_DAYS + 1):
        day_data = agg.get(day, {})
        costs = {s: _avg(day_data.get(s, {}).get("costs", [float("nan")])) for s in SAT_VALUES}
        best_sat = min(SAT_VALUES, key=lambda s: costs[s])
        for s in SAT_VALUES:
            sd = day_data.get(s, {})
            c_list = sd.get("costs", [])
            fe_list = sd.get("fe_costs", [])
            se_list = sd.get("se_costs", [])
            nv_list = sd.get("nvs", [])
            co2_list = sd.get("carbons", [])
            rt_list = sd.get("rts", [])
            rows.append(
                f"{day},{s},"
                f"{_avg(c_list):.2f},{_std(c_list):.2f},"
                f"{_avg(fe_list):.2f},{_std(fe_list):.2f},"
                f"{_avg(se_list):.2f},{_std(se_list):.2f},"
                f"{_avg(nv_list):.2f},{_std(nv_list):.2f},"
                f"{_avg(co2_list):.4f},{_std(co2_list):.4f},"
                f"{_avg(rt_list):.1f},{_std(rt_list):.1f},"
                f"{len(c_list)},{1 if s == best_sat else 0}"
            )
    FULL_CSV.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_raw_csv(all_results: list) -> None:
    """Write per-run raw CSV."""
    header = "day,n_sat,run,cost,fe_cost,se_cost,nv,carbon,elapsed_s"
    rows = [header]
    for r in sorted(all_results, key=lambda x: (x["day"], x["n_sat"], x["run"])):
        rows.append(
            f"{r['day']},{r['n_sat']},{r['run']},"
            f"{r['cost']:.2f},{r['fe_cost']:.2f},{r['se_cost']:.2f},"
            f"{r['nv']},{r['carbon']:.4f},{r['elapsed']:.1f}"
        )
    RAW_CSV.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_summary(agg: dict, wall_seconds: float) -> None:
    """Write 30-day aggregate summary."""
    lines = [
        "=" * 70,
        "  30-Day Real-World Case Study Summary — INSGA3",
        "=" * 70,
        f"  Runs per (day, satellite): {N_RUNS}",
        f"  Total wall time: {wall_seconds/3600:.2f} h ({wall_seconds:.0f} s)",
        "",
        f"  {'Day':>4}  {'Best Sat':>8}  {'Opt Cost ($)':>14}  "
        f"{'NV':>4}  {'CO2 (kg)':>10}  {'RT (s)':>7}",
        "-" * 70,
    ]
    total_cost = total_nv = total_co2 = total_rt = 0.0
    dist = {1: 0, 2: 0, 3: 0}

    for day in range(1, N_DAYS + 1):
        day_data = agg.get(day, {})
        costs = {s: _avg(day_data.get(s, {}).get("costs", [float("nan")])) for s in SAT_VALUES}
        best_sat = min(SAT_VALUES, key=lambda s: costs[s])
        dist[best_sat] += 1

        sd = day_data.get(best_sat, {})
        opt_cost = costs[best_sat]
        nv  = _avg(sd.get("nvs", [0]))
        co2 = _avg(sd.get("carbons", [0]))
        rt  = _avg(sd.get("rts", [0]))
        total_cost += opt_cost
        total_nv   += nv
        total_co2  += co2
        total_rt   += rt
        lines.append(
            f"  {day:>4}  {best_sat:>8}  {opt_cost:>14,.1f}  "
            f"{nv:>4.1f}  {co2:>10.4f}  {rt:>7.1f}"
        )

    lines += [
        "-" * 70,
        f"  {'Total':>4}             {total_cost:>14,.1f}  "
        f"{total_nv:>4.1f}  {total_co2:>10.4f}  {total_rt:>7.1f}",
        "",
        f"  Satellite distribution: Sat=1: {dist[1]} days, "
        f"Sat=2: {dist[2]} days, Sat=3: {dist[3]} days",
        "=" * 70,
    ]
    SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    from HNSGA3_vrp import _get_instance_rows_cache
    _get_instance_rows_cache()

    progress = _load_progress()
    all_results: list = []

    # Rebuild all_results from existing progress
    for day_d, sat_d in progress.items():
        for sat_s, run_list in sat_d.items():
            for r in run_list:
                all_results.append(r)

    # Build task list (skip already completed)
    tasks = []
    for day in range(1, N_DAYS + 1):
        cp, mp = _params(day)
        for n_sat in SAT_VALUES:
            done_runs = len(progress.get(day, {}).get(n_sat, []))
            for run_num in range(done_runs + 1, N_RUNS + 1):
                tasks.append((day, n_sat, run_num, cp, mp))

    total   = len(tasks)
    done    = sum(len(r) for d in progress.values() for r in d.values())
    grand   = N_DAYS * len(SAT_VALUES) * N_RUNS
    wall_t0 = time.time()

    print(f"\n  INSGA3 Case Study — {N_DAYS} days × {len(SAT_VALUES)} sats × {N_RUNS} runs")
    print(f"  Already done: {done}/{grand}   Remaining: {total}")
    print(f"  Workers: {N_WORKERS}   Output: {OUTPUT_DIR}\n")

    if tasks:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            fm = {pool.submit(_worker, t): t for t in tasks}
            for fut in as_completed(fm):
                try:
                    r = fut.result()

                    # Save route JSON
                    route_path = ROUTES_DIR / f"day{r['day']:02d}_sat{r['n_sat']}_run{r['run']:02d}.json"
                    route_path.write_text(
                        json.dumps(r["routes"], ensure_ascii=False, indent=2),
                        encoding="utf-8"
                    )

                    # Strip routes from result before storing in progress
                    r_slim = {k: v for k, v in r.items() if k != "routes"}
                    all_results.append(r_slim)

                    day, n_sat = r["day"], r["n_sat"]
                    progress.setdefault(day, {}).setdefault(n_sat, []).append(r_slim)
                    _save_progress(progress)

                    elapsed_wall = time.time() - wall_t0
                    done += 1
                    print(
                        f"  [{done:>3}/{grand}] day={day:02d} sat={n_sat} run={r['run']}"
                        f"  cost={r['cost']:,.1f}  nv={r['nv']}"
                        f"  co2={r['carbon']:.4f}  rt={r['elapsed']:.1f}s"
                        f"  (wall={elapsed_wall:.0f}s)",
                        flush=True,
                    )
                except Exception as e:
                    t = fm[fut]
                    print(f"  ERROR day={t[0]} sat={t[1]} run={t[2]}: {e}", flush=True)

    wall_total = time.time() - wall_t0

    # ── Aggregate results ─────────────────────────────────────────────────────
    agg: dict = {}   # agg[day][sat] = {costs, fe_costs, se_costs, nvs, carbons, rts}
    for r in all_results:
        day, sat = r["day"], r["n_sat"]
        agg.setdefault(day, {}).setdefault(sat, {
            "costs": [], "fe_costs": [], "se_costs": [],
            "nvs": [], "carbons": [], "rts": []
        })
        agg[day][sat]["costs"].append(r["cost"])
        agg[day][sat]["fe_costs"].append(r["fe_cost"])
        agg[day][sat]["se_costs"].append(r["se_cost"])
        agg[day][sat]["nvs"].append(r["nv"])
        agg[day][sat]["carbons"].append(r["carbon"])
        agg[day][sat]["rts"].append(r["elapsed"])

    # ── Write outputs ─────────────────────────────────────────────────────────
    _write_txt(agg)
    _write_full_csv(agg)
    _write_raw_csv(all_results)
    _write_summary(agg, wall_total)

    print(f"\n  Done. Wall time: {wall_total/3600:.2f} h")
    print(f"  TABLE13_INSGA3.TXT      → {TXT_FILE}")
    print(f"  TABLE13_FULL_INSGA3.CSV → {FULL_CSV}")
    print(f"  TABLE13_RAW_INSGA3.CSV  → {RAW_CSV}")
    print(f"  summary.txt             → {SUMMARY_FILE}")
    print(f"  routes/                 → {ROUTES_DIR}  ({N_DAYS*len(SAT_VALUES)*N_RUNS} JSON files)")


if __name__ == "__main__":
    main()

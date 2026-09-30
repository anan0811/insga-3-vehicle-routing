"""compute_pareto_metrics_4alg.py — Pareto quality metrics for 4 algorithms.

Algorithms:
  INSGA3   : 論文/INSGA-III/INSGA3_optsat_RAW.TXT
  NSGA-III : 論文/NSGA-III/deprecated/Instance_PURE_NSGA3_RAW.TXT
  NSGA-II  : 論文/NSGA-II/Instance_PURE_NSGA2_RAW.TXT
  MOVNS    : 論文/MOVNS/MOVNS_instance_RAW.TXT

Per-run raw format (all files):
  inst=XX run=XX cost=X NV=X carbon=X.XXXX elapsed=Xs

Metrics computed per instance, then averaged:
  GD   Generational Distance (lower = better)
  IGD  Inverted GD (lower = better)
  SD   Spread / Delta (lower = better)
  HV   Hypervolume 3D normalized, ref=(1.1,1.1,1.1) (higher = better)
  SP   Spacing — uniformity (lower = more uniform)
  SC   Set Coverage matrix (higher = better for attacker)

Reference Pareto front: empirical union of all 4 algorithms' non-dominated pts per instance.

Output: Pareto_metrics_4alg.TXT
"""
from __future__ import annotations
import math, re
from pathlib import Path

HERE     = Path(__file__).resolve().parent   # 論文/experiments/
ROOT     = HERE.parent                       # 論文/
OUTPUT   = HERE / "Pareto_metrics_4alg.TXT"

ALG_FILES = {
    "INSGA3":   ROOT / "INSGA-III" / "INSGA3_optsat_RAW.TXT",
    "NSGA-III": ROOT / "NSGA-III"  / "deprecated" / "Instance_PURE_NSGA3_RAW.TXT",
    "NSGA-II":  ROOT / "NSGA-II"   / "Instance_PURE_NSGA2_RAW.TXT",
    "MOVNS":    ROOT / "MOVNS"     / "MOVNS_instance_RAW.TXT",
}
ALG_NAMES = list(ALG_FILES.keys())
N_INST = 20

# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_raw(path: Path) -> dict[int, list[tuple]]:
    """Return {inst_id: [(cost, nv, carbon), ...]} from a RAW.TXT file."""
    pat = re.compile(
        r"inst=(\d+)\s+run=\d+\s+cost=([\d.]+)\s+NV=(\d+)\s+carbon=([\d.]+)"
    )
    data: dict[int, list] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.search(line)
        if m:
            inst = int(m.group(1))
            cost = float(m.group(2))
            nv   = int(m.group(3))
            co2  = float(m.group(4))
            data.setdefault(inst, []).append((cost, nv, co2))
    return data


# ── Math helpers ──────────────────────────────────────────────────────────────

def _avg(v): return sum(v) / len(v) if v else float("nan")

def _std(v):
    if len(v) < 2: return 0.0
    m = _avg(v)
    return math.sqrt(sum((x - m)**2 for x in v) / (len(v) - 1))

def _dominates(a, b):
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))

def _pareto_front(pts):
    """Return non-dominated subset of pts."""
    nd = []
    for p in pts:
        if not any(_dominates(q, p) for q in pts if q is not p):
            nd.append(p)
    return nd

def _normalize(pts_dict):
    """Min-max normalize all points in pts_dict to [0,1]."""
    all_pts = [p for pts in pts_dict.values() for p in pts]
    if not all_pts:
        return {k: [] for k in pts_dict}, [], []
    dim = len(all_pts[0])
    objs = list(zip(*all_pts))
    mn   = [min(o) for o in objs]
    mx   = [max(o) for o in objs]
    rng  = [mx[i] - mn[i] if mx[i] != mn[i] else 1.0 for i in range(dim)]
    norm = {
        alg: [tuple((v - mn[i]) / rng[i] for i, v in enumerate(p)) for p in pts]
        for alg, pts in pts_dict.items()
    }
    return norm, mn, mx

def _dist_point_to_set(p, S):
    return min(math.sqrt(sum((pi - si)**2 for pi, si in zip(p, s))) for s in S)

def _gd(approx, ref):
    if not approx or not ref: return float("nan")
    return math.sqrt(sum(_dist_point_to_set(p, ref)**2 for p in approx)) / len(approx)

def _igd(approx, ref):
    if not approx or not ref: return float("nan")
    return math.sqrt(sum(_dist_point_to_set(r, approx)**2 for r in ref)) / len(ref)

def _spacing(pts):
    if len(pts) < 2: return 0.0
    dists = [
        min(math.sqrt(sum((p[k]-q[k])**2 for k in range(len(p)))) for j,q in enumerate(pts) if j != i)
        for i, p in enumerate(pts)
    ]
    return _std(dists)

def _spread_delta(approx, ref):
    """Delta (spread) metric: measures extent + uniformity."""
    if len(approx) < 2 or not ref: return float("nan")
    # extreme points of ref front
    ref_min = [min(r[k] for r in ref) for k in range(len(ref[0]))]
    ref_max = [max(r[k] for r in ref) for k in range(len(ref[0]))]
    # find approx extremes
    d_extremes = []
    for k in range(len(approx[0])):
        val_min = min(p[k] for p in approx)
        val_max = max(p[k] for p in approx)
        d_extremes.append(abs(val_min - ref_min[k]))
        d_extremes.append(abs(val_max - ref_max[k]))
    df = sum(d_extremes)
    # consecutive distances
    dists = sorted([
        min(math.sqrt(sum((p[k]-q[k])**2 for k in range(len(p)))) for j,q in enumerate(approx) if j!=i)
        for i,p in enumerate(approx)
    ])
    d_bar = _avg(dists) if dists else 0.0
    numerator   = df + sum(abs(di - d_bar) for di in dists)
    denominator = df + len(dists) * d_bar
    return numerator / denominator if denominator > 0 else float("nan")

def _hypervolume_2d(pts_2d, ref=(1.1, 1.1)):
    """Exact 2D HV via staircase sweep (used as sub-routine for 3D HV)."""
    valid = [p for p in pts_2d if p[0] <= ref[0] and p[1] <= ref[1]]
    if not valid: return 0.0
    sorted_pts = sorted(valid, key=lambda x: x[0])
    hv, prev_y = 0.0, ref[1]
    for p in sorted_pts:
        if p[1] < prev_y:
            hv += (ref[0] - p[0]) * (prev_y - p[1])
            prev_y = p[1]
    return hv

def _hypervolume_3d(pts_3d, ref=(1.1, 1.1, 1.1)):
    """Exact 3D HV via sweep-line along f1 axis.

    Slices [f1_i, f1_{i+1}) are processed left-to-right; the (f2,f3)
    contribution of each slice is the 2D HV of all points active at that
    position (i.e., those with f1 <= current slice start).
    """
    valid = [p for p in pts_3d if p[0] < ref[0] and p[1] < ref[1] and p[2] < ref[2]]
    if not valid:
        return 0.0
    nd = _pareto_front(valid)
    if not nd:
        return 0.0
    nd_sorted = sorted(nd, key=lambda x: x[0])
    n = len(nd_sorted)
    hv = 0.0
    for i in range(n):
        width = (nd_sorted[i + 1][0] if i < n - 1 else ref[0]) - nd_sorted[i][0]
        if width <= 0:
            continue
        pts_2d = [(p[1], p[2]) for p in nd_sorted[: i + 1]]
        hv += width * _hypervolume_2d(pts_2d, (ref[1], ref[2]))
    return hv

def _set_coverage(A, B):
    """SC(A→B): fraction of B dominated by ≥1 point in A."""
    if not B: return float("nan")
    return sum(1 for b in B if any(_dominates(a, b) for a in A)) / len(B)


# ── Load data ─────────────────────────────────────────────────────────────────

def main():
    raw: dict[str, dict[int, list]] = {}
    missing = []
    for alg, path in ALG_FILES.items():
        if not path.exists():
            missing.append(f"{alg}: {path.name}")
            raw[alg] = {}
        else:
            raw[alg] = _parse_raw(path)
            n_runs = {inst: len(pts) for inst, pts in raw[alg].items()}
            print(f"  Loaded {alg}: {len(raw[alg])} instances, "
                  f"runs/inst = {set(n_runs.values())}")

    if missing:
        print("\n  WARNING — missing raw files:")
        for m in missing:
            print(f"    {m}")
        print("  Missing algorithms will show NaN.\n")

    # Per-instance metrics
    gd_by  = {a: [] for a in ALG_NAMES}
    igd_by = {a: [] for a in ALG_NAMES}
    sd_by  = {a: [] for a in ALG_NAMES}
    hv_by  = {a: [] for a in ALG_NAMES}
    sp_by  = {a: [] for a in ALG_NAMES}
    sc_mat = {a: {b: [] for b in ALG_NAMES if b != a} for a in ALG_NAMES}

    for inst in range(1, N_INST + 1):
        pts_3d_raw = {alg: raw[alg].get(inst, []) for alg in ALG_NAMES}

        # Empirical reference: non-dominated union of all algorithms (3D)
        all_pts = [p for pts in pts_3d_raw.values() for p in pts]
        ref_3d  = _pareto_front(all_pts) if all_pts else []

        # Normalize 3D for all metrics
        combined_pts = {alg: pts_3d_raw[alg] for alg in ALG_NAMES}
        norm3d, _, _ = _normalize(combined_pts)

        ref_3d_norm  = _pareto_front([p for pts in norm3d.values() for p in pts]) if any(norm3d.values()) else []

        for alg in ALG_NAMES:
            pts3 = norm3d.get(alg, [])

            gd_by[alg].append(_gd(pts3, ref_3d_norm))
            igd_by[alg].append(_igd(pts3, ref_3d_norm))
            sd_by[alg].append(_spread_delta(pts3, ref_3d_norm))
            hv_by[alg].append(_hypervolume_3d(pts3, (1.1, 1.1, 1.1)))
            sp_by[alg].append(_spacing(pts3))

        # SC uses raw 3D (unnormalized)
        for a in ALG_NAMES:
            for b in ALG_NAMES:
                if a == b: continue
                sc_mat[a][b].append(_set_coverage(pts_3d_raw[a], pts_3d_raw[b]))

    # ── Output ────────────────────────────────────────────────────────────────
    sep  = "=" * 72
    sep2 = "-" * 72
    lines = [sep, "Pareto Quality Metrics — 4 Algorithms", sep, ""]

    # Per-instance tables for each metric
    for metric_name, by_alg, better in [
        ("GD  (lower = better)", gd_by,  "↓"),
        ("IGD (lower = better)", igd_by, "↓"),
        ("SD  (lower = better)", sd_by,  "↓"),
        ("HV  (higher = better)", hv_by, "↑"),
        ("SP  (lower = better)",  sp_by, "↓"),
    ]:
        lines.append(f"  {metric_name}")
        lines.append(sep2)
        hdr = f"  {'Inst':>5}  " + "  ".join(f"{a:>12}" for a in ALG_NAMES)
        lines.append(hdr)
        lines.append(sep2)
        for inst in range(1, N_INST + 1):
            row = f"  {inst:>5}  "
            for alg in ALG_NAMES:
                v = by_alg[alg][inst - 1] if inst - 1 < len(by_alg[alg]) else float("nan")
                row += f"  {v:>12.6f}"
            lines.append(row)
        lines.append(sep2)
        avg_row = f"  {'Avg':>5}  "
        for alg in ALG_NAMES:
            vals = [v for v in by_alg[alg] if not math.isnan(v)]
            avg_row += f"  {_avg(vals):>12.6f}"
        lines.append(avg_row)
        lines.append("")

    # SC matrix
    lines += [sep, "SC(A→B): avg fraction of B dominated by A (higher = better for A)", sep2]
    hdr = f"  {'A\\B':>10}  " + "  ".join(f"{b:>10}" for b in ALG_NAMES)
    lines.append(hdr)
    lines.append(sep2)
    for a in ALG_NAMES:
        row = f"  {a:>10}  "
        for b in ALG_NAMES:
            if a == b:
                row += f"  {'—':>10}"
            else:
                vals = [v for v in sc_mat[a][b] if not math.isnan(v)]
                row += f"  {_avg(vals):>10.4f}"
        lines.append(row)
    lines.append("")

    # Summary table
    lines += [sep, "Summary (avg across 20 instances)", sep2]
    hdr = f"  {'Algorithm':>10}  {'Avg GD':>10}  {'Avg IGD':>10}  {'Avg SD':>10}  {'Avg HV':>10}  {'Avg SP':>10}"
    lines.append(hdr)
    lines.append(sep2)
    for alg in ALG_NAMES:
        def _m(by): return _avg([v for v in by[alg] if not math.isnan(v)])
        lines.append(
            f"  {alg:>10}  {_m(gd_by):>10.6f}  {_m(igd_by):>10.6f}"
            f"  {_m(sd_by):>10.6f}  {_m(hv_by):>10.6f}  {_m(sp_by):>10.6f}"
        )
    lines += [sep, ""]

    text = "\n".join(lines) + "\n"
    OUTPUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()

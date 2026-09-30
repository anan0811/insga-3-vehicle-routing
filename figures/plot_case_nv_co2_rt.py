import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv, pathlib

CSV_PATH = r"C:\Users\admin\Desktop\My-paper--main\論文\實驗\Case test\TABLE13_FULL_INSGA3.CSV"
OUT_DIR  = pathlib.Path(r"C:\Users\admin\Desktop\My-paper--main\論文\figures")

# ── load data ────────────────────────────────────────────────────────────────
data = {}  # data[(day, n_sat)] = {col: value}
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        d, s = int(row["day"]), int(row["n_sat"])
        data[(d, s)] = row

days = list(range(1, 31))

def series(col, sat):
    return [float(data[(d, sat)][col]) for d in days]

def dynamic_series(col):
    """value from the optimal satellite each day"""
    return [float(data[(d, int(data[(d,1)]["is_optimal"] == "1" and 1 or
                           data[(d,2)]["is_optimal"] == "1" and 2 or 3))][col])
            for d in days]

# find optimal satellite per day
opt_sat = []
for d in days:
    for s in [1, 2, 3]:
        if data[(d, s)]["is_optimal"] == "1":
            opt_sat.append(s)
            break

def dyn_vals(col):
    return [float(data[(d, opt_sat[i])][col]) for i, d in enumerate(days)]

# ── colour scheme (same as cost trend) ───────────────────────────────────────
c1, c2, c3, cd = "#4C8BCC", "#E07B39", "#3DA858", "#1A1A1A"
opt_colors = {1: c1, 2: c2, 3: c3}

# ── plot function ─────────────────────────────────────────────────────────────
def make_plot(col, title, ylabel, fname, fmt=None):
    s1 = series(col, 1)
    s2 = series(col, 2)
    s3 = series(col, 3)
    dy = dyn_vals(col)

    fig, ax = plt.subplots(figsize=(10, 5))

    lw = 1.3
    ax.plot(days, s1, color=c1, lw=lw, linestyle="--", alpha=0.75, zorder=2)
    ax.plot(days, s2, color=c2, lw=lw, linestyle="--", alpha=0.75, zorder=2)
    ax.plot(days, s3, color=c3, lw=lw, linestyle="--", alpha=0.75, zorder=2)
    ax.plot(days, dy, color=cd, lw=2.0, linestyle="-",  zorder=3)

    for d, opt, val in zip(days, opt_sat, dy):
        ax.scatter(d, val, color=opt_colors[opt], s=55, zorder=4,
                   edgecolors="white", linewidths=0.6)

    # totals / averages for legend
    tot1 = sum(s1); tot2 = sum(s2); tot3 = sum(s3); tot_d = sum(dy)
    if fmt == "avg":
        lbl = lambda v: f"{v/30:.2f} avg"
        tot_label = f"Dynamic optimal  (30-day avg: {tot_d/30:.2f})"
        p1_label  = f"Fixed Sat=1  (30-day avg: {tot1/30:.2f})"
        p2_label  = f"Fixed Sat=2  (30-day avg: {tot2/30:.2f})"
        p3_label  = f"Fixed Sat=3  (30-day avg: {tot3/30:.2f})"
    else:
        tot_label = f"Dynamic optimal  (30-day total: {tot_d:,.1f})"
        p1_label  = f"Fixed Sat=1  (30-day total: {tot1:,.1f})"
        p2_label  = f"Fixed Sat=2  (30-day total: {tot2:,.1f})"
        p3_label  = f"Fixed Sat=3  (30-day total: {tot3:,.1f})"

    patch1   = mpatches.Patch(color=c1, label=p1_label)
    patch2   = mpatches.Patch(color=c2, label=p2_label)
    patch3   = mpatches.Patch(color=c3, label=p3_label)
    dyn_line = plt.Line2D([0],[0], color=cd, lw=2, label=tot_label)
    dot1 = plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=c1,
                       markersize=7, label="Optimal = Sat 1")
    dot2 = plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=c2,
                       markersize=7, label="Optimal = Sat 2")
    dot3 = plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=c3,
                       markersize=7, label="Optimal = Sat 3")

    ax.legend(handles=[dyn_line, patch1, patch2, patch3, dot1, dot2, dot3],
              fontsize=8.5, loc="upper left", framealpha=0.9, ncol=1)

    ax.set_xlabel("Planning Day", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks(days)
    ax.set_xticklabels(days, fontsize=7.5)
    ax.tick_params(axis="y", labelsize=9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:,.0f}") if max(dy) > 50
        else plt.FuncFormatter(lambda v, _: f"{v:.1f}")
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_xlim(0.5, 30.5)

    plt.tight_layout()
    out = OUT_DIR / fname
    plt.savefig(out, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# ── generate three PDFs ───────────────────────────────────────────────────────
make_plot("nv_avg",
          "30-Day Vehicle Count Trend: Dynamic vs. Fixed Satellite Configurations",
          "Number of Vehicles (NV)",
          "case_nv_trend.pdf",
          fmt="avg")

make_plot("co2_avg",
          "30-Day Carbon Emission Trend: Dynamic vs. Fixed Satellite Configurations",
          "CO₂ Emissions (kg)",
          "case_co2_trend.pdf")

make_plot("rt_avg",
          "30-Day Runtime Trend: Dynamic vs. Fixed Satellite Configurations",
          "Runtime (s)",
          "case_rt_trend.pdf",
          fmt="avg")

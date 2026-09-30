import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

days = list(range(1, 31))

sat1 = [4352.4, 7245.8, 4704.0, 5054.9, 4937.0, 5995.4, 5030.5, 6995.6,
        6748.7, 6067.5, 8451.4, 8473.5, 8411.5, 5924.5, 7236.2, 8937.4,
        9047.0, 6864.4, 8658.2,10992.3,10224.5,10421.2,10859.6,13091.2,
        7793.4,11020.8,10860.4,11257.9,10319.4,11258.9]

sat2 = [5116.3, 5490.8, 4727.1, 4647.6, 5847.5, 5456.1, 6326.8, 8352.5,
        7906.2, 5358.8,10463.6,10803.7, 7693.6, 5730.8, 8749.8,10259.9,
       10811.8, 6638.3,10110.7,14168.2,13041.5,11311.9,12270.4,14072.8,
        6850.4, 9938.2,13789.6,12449.2,12769.5,12603.7]

sat3 = [4596.2, 6536.2, 4228.4, 5727.6, 5665.1, 6415.0, 4788.6, 6859.7,
        8252.9, 6315.7, 9624.0, 8207.0, 9084.1, 7597.7, 6808.3, 9017.7,
        9405.6, 6318.4, 8577.1,12544.5,14010.6, 8635.3,12178.2, 9399.1,
        6853.7,11266.9,13264.6, 8214.7,10776.4,11808.2]

optimal_sat = [1,2,3,2,1,2,3,3,1,2,1,3,2,2,3,1,1,3,3,1,1,3,1,3,2,2,1,3,1,1]
dynamic = [min(s1,s2,s3) for s1,s2,s3 in zip(sat1,sat2,sat3)]

fig, ax = plt.subplots(figsize=(10, 5))

c1 = "#4C8BCC"
c2 = "#E07B39"
c3 = "#3DA858"
cd = "#1A1A1A"

lw = 1.3
ax.plot(days, sat1, color=c1, lw=lw, linestyle="--", alpha=0.75, zorder=2)
ax.plot(days, sat2, color=c2, lw=lw, linestyle="--", alpha=0.75, zorder=2)
ax.plot(days, sat3, color=c3, lw=lw, linestyle="--", alpha=0.75, zorder=2)
ax.plot(days, dynamic, color=cd, lw=2.0, linestyle="-", zorder=3,
        label="Dynamic (optimal per day)")

# 最優點標記（顏色對應衛星數）
opt_colors = {1: c1, 2: c2, 3: c3}
for d, opt, dyn_cost in zip(days, optimal_sat, dynamic):
    ax.scatter(d, dyn_cost, color=opt_colors[opt], s=55, zorder=4,
               edgecolors="white", linewidths=0.6)

# 固定衛星政策水平總成本標注（右側）
total_dyn  = sum(dynamic)
total_s1   = sum(sat1)
total_s2   = sum(sat2)
total_s3   = sum(sat3)

ax.set_xlabel("Planning Day", fontsize=11)
ax.set_ylabel("Total Cost (\\$)", fontsize=11)
ax.set_title("30-Day Cost Trend: Dynamic vs. Fixed Satellite Configurations",
             fontsize=12, pad=10)
ax.set_xticks(days)
ax.set_xticklabels(days, fontsize=7.5)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.tick_params(axis="y", labelsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.set_xlim(0.5, 30.5)

# Legend
patch1 = mpatches.Patch(color=c1, label=f"Fixed Sat=1  (30-day total: \\${total_s1:,.1f})")
patch2 = mpatches.Patch(color=c2, label=f"Fixed Sat=2  (30-day total: \\${total_s2:,.1f})")
patch3 = mpatches.Patch(color=c3, label=f"Fixed Sat=3  (30-day total: \\${total_s3:,.1f})")
dyn_line = plt.Line2D([0],[0], color=cd, lw=2,
                       label=f"Dynamic optimal  (30-day total: \\${total_dyn:,.1f})")
dot1 = plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=c1,
                   markersize=7, label="Optimal = Sat 1")
dot2 = plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=c2,
                   markersize=7, label="Optimal = Sat 2")
dot3 = plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=c3,
                   markersize=7, label="Optimal = Sat 3")

ax.legend(handles=[dyn_line, patch1, patch2, patch3, dot1, dot2, dot3],
          fontsize=8.5, loc="upper left", framealpha=0.9, ncol=1)

plt.tight_layout()
out = r"C:\Users\admin\Desktop\My-paper--main\論文\figures\case_cost_trend.pdf"
plt.savefig(out, format="pdf", dpi=300, bbox_inches="tight")
print(f"Saved: {out}")

"""plot_toy_instance.py — Coordinate diagram for the illustrative numerical example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Node data: (label, x, y, demand)
depot     = [("$D$",  0, 0)]
satellites = [("$S_1$", 2, 2), ("$S_2$", 6, 3)]
customers  = [
    ("$C_1$", 1, 3, 5),
    ("$C_2$", 3, 1, 8),
    ("$C_3$", 2, 4, 4),
    ("$C_4$", 5, 4, 6),
    ("$C_5$", 7, 2, 7),
    ("$C_6$", 6, 5, 5),
]

fig, ax = plt.subplots(figsize=(4.5, 3.8))

# Depot
ax.scatter(0, 0, marker='s', s=120, color='#cc0000', zorder=5)
ax.annotate("$D$", (0, 0), textcoords="offset points", xytext=(-14, 5),
            fontsize=9, color='#cc0000', fontweight='bold')

# Satellites
for lbl, x, y in satellites:
    ax.scatter(x, y, marker='^', s=110, color='#1a6fbf', zorder=5)
    ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 5),
                fontsize=9, color='#1a6fbf', fontweight='bold')

# Customers
for lbl, x, y, d in customers:
    ax.scatter(x, y, marker='o', s=80, color='#333333', zorder=5)
    ax.annotate(f"{lbl} ({d})", (x, y), textcoords="offset points",
                xytext=(6, 4), fontsize=8, color='#333333')

# Axes
ax.set_xlim(-0.8, 8.2)
ax.set_ylim(-0.8, 6.2)
ax.set_xlabel("$x$", fontsize=9)
ax.set_ylabel("$y$", fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)

# Legend
leg_handles = [
    mpatches.Patch(color='#cc0000', label='Depot ($D$)'),
    mpatches.Patch(color='#1a6fbf', label='Satellite ($S_1$, $S_2$)'),
    mpatches.Patch(color='#333333', label='Customer (demand)'),
]
ax.legend(handles=leg_handles, fontsize=7.5, loc='lower right',
          framealpha=0.9, borderpad=0.5)

plt.tight_layout(pad=0.4)

out = r'C:\Users\admin\Desktop\My-paper--main\論文\figures\illustrative_instance.pdf'
fig.savefig(out, format='pdf', bbox_inches='tight')
print(f'Saved: {out}')
plt.close()

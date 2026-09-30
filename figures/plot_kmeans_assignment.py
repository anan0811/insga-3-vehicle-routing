"""plot_kmeans_assignment.py — K-Medoids cluster assignment diagram."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np

fig, ax = plt.subplots(figsize=(4.2, 3.6))

# ── Node data ─────────────────────────────────────────────────────────────────
depot      = (0, 0)
satellites = {'$S_1$': (2, 2), '$S_2$': (6, 3)}
cluster1   = {'$C_1$': (1, 3, 5), '$C_2$': (3, 1, 8), '$C_3$': (2, 4, 4)}
cluster2   = {'$C_4$': (5, 4, 6), '$C_5$': (7, 2, 7), '$C_6$': (6, 5, 5)}

C1_COLOR = '#1a6fbf'   # blue  — S1 cluster
C2_COLOR = '#e07b2a'   # orange — S2 cluster
SAT_EDGE = '#333333'

# ── Cluster background ellipses ───────────────────────────────────────────────
# S1 cluster: roughly centred around (2, 2.7)
e1 = Ellipse((2.0, 2.5), width=3.2, height=3.6, angle=10,
             facecolor=C1_COLOR, alpha=0.10, zorder=0)
ax.add_patch(e1)

# S2 cluster: roughly centred around (6, 3.7)
e2 = Ellipse((6.0, 3.7), width=3.2, height=3.6, angle=-5,
             facecolor=C2_COLOR, alpha=0.10, zorder=0)
ax.add_patch(e2)

# ── Lines: satellite → assigned customers ─────────────────────────────────────
for lbl, (x, y, _) in cluster1.items():
    ax.plot([satellites['$S_1$'][0], x], [satellites['$S_1$'][1], y],
            color=C1_COLOR, linewidth=0.8, linestyle='--', alpha=0.6, zorder=1)

for lbl, (x, y, _) in cluster2.items():
    ax.plot([satellites['$S_2$'][0], x], [satellites['$S_2$'][1], y],
            color=C2_COLOR, linewidth=0.8, linestyle='--', alpha=0.6, zorder=1)

# ── Depot ─────────────────────────────────────────────────────────────────────
ax.scatter(*depot, marker='s', s=110, color='#cc0000', zorder=5)
ax.annotate('$D$', depot, xytext=(-14, 5), textcoords='offset points',
            fontsize=9, color='#cc0000', fontweight='bold')

# ── Satellites ────────────────────────────────────────────────────────────────
for lbl, (x, y) in satellites.items():
    color = C1_COLOR if lbl == '$S_1$' else C2_COLOR
    ax.scatter(x, y, marker='^', s=130, color=color,
               edgecolors=SAT_EDGE, linewidths=0.6, zorder=5)
    ax.annotate(lbl, (x, y), xytext=(6, 5), textcoords='offset points',
                fontsize=9, color=color, fontweight='bold')

# ── Customers ─────────────────────────────────────────────────────────────────
for lbl, (x, y, d) in cluster1.items():
    ax.scatter(x, y, marker='o', s=75, color=C1_COLOR,
               edgecolors=SAT_EDGE, linewidths=0.5, zorder=5)
    ax.annotate(f'{lbl} ({d})', (x, y), xytext=(6, 4),
                textcoords='offset points', fontsize=8, color='#333333')

for lbl, (x, y, d) in cluster2.items():
    ax.scatter(x, y, marker='o', s=75, color=C2_COLOR,
               edgecolors=SAT_EDGE, linewidths=0.5, zorder=5)
    ax.annotate(f'{lbl} ({d})', (x, y), xytext=(6, 4),
                textcoords='offset points', fontsize=8, color='#333333')

# ── Demand labels ─────────────────────────────────────────────────────────────
ax.text(1.4, 1.1, 'Total demand: 17', fontsize=7.5, color=C1_COLOR,
        style='italic')
ax.text(4.8, 5.8, 'Total demand: 18', fontsize=7.5, color=C2_COLOR,
        style='italic')

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(-0.8, 8.5)
ax.set_ylim(-0.5, 6.5)
ax.set_xlabel('$x$', fontsize=9)
ax.set_ylabel('$y$', fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, linestyle='--', linewidth=0.35, alpha=0.5)

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color='#cc0000', label='Depot'),
    mpatches.Patch(color=C1_COLOR,  label='$S_1$ cluster'),
    mpatches.Patch(color=C2_COLOR,  label='$S_2$ cluster'),
]
ax.legend(handles=handles, fontsize=7.5, loc='lower right',
          framealpha=0.9, borderpad=0.5)

plt.tight_layout(pad=0.4)
out = r'C:\Users\admin\Desktop\My-paper--main\論文\figures\kmeans_assignment_draft.pdf'
fig.savefig(out, format='pdf', bbox_inches='tight')
print(f'Saved: {out}')
plt.close()

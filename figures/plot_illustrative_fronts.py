"""plot_illustrative_fronts.py — Pareto front assignment diagram for the illustrative population."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
solutions = ['$x_1$', '$x_2$', '$x_3$', '$x_4$', '$x_5$', '$x_6$']
f1 = [145, 158, 162, 170, 155, 175]   # cost
f2 = [2,   1,   2,   2,   3,   2  ]   # NV
f3 = [12.1, 11.5, 10.8, 11.9, 14.2, 13.0]  # CO2
fronts = ['I', 'I', 'I', 'II', 'II', 'III']

FRONT_COLORS = {'I': '#1a6fbf', 'II': '#e07b2a', 'III': '#888888'}
FRONT_MARKERS = {'I': 'o', 'II': 's', 'III': '^'}
FRONT_SIZES   = {'I': 90,  'II': 80,  'III': 75}

# ── Figure: Parallel Coordinates ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.set_xlim(-0.3, 2.3)
ax.set_ylim(0, 1.12)
ax.axis('off')

axes_x = [0, 1, 2]
axis_labels = ['$f_1$ (cost)', '$f_2$ (NV)', '$f_3$ (CO$_2$, kg)']

# Normalise each axis to [0, 1] (min→bottom, max→top)
data = np.array([f1, f2, f3], dtype=float)
axis_min = data.min(axis=1)
axis_max = data.max(axis=1)

def norm(val, ax_idx):
    return (val - axis_min[ax_idx]) / (axis_max[ax_idx] - axis_min[ax_idx])

# ── Draw vertical axes ────────────────────────────────────────────────────────
for xi in axes_x:
    ax.plot([xi, xi], [0, 1], color='#555555', linewidth=1.0, zorder=1)
    ax.text(xi, 1.05, axis_labels[xi], ha='center', va='bottom', fontsize=8.5,
            color='#333333')
    # tick labels: min at bottom, max at top
    ax.text(xi, -0.06, f'{axis_min[xi]:.0f}' if xi < 2 else f'{axis_min[xi]:.1f}',
            ha='center', va='top', fontsize=7, color='#666666')
    ax.text(xi, 1.03, f'{axis_max[xi]:.0f}' if xi < 2 else f'{axis_max[xi]:.1f}',
            ha='center', va='bottom', fontsize=7, color='#666666')

# ── Draw lines + dots per solution ────────────────────────────────────────────
for i, sol in enumerate(solutions):
    fnt = fronts[i]
    color = FRONT_COLORS[fnt]
    y_vals = [norm(f1[i], 0), norm(f2[i], 1), norm(f3[i], 2)]

    # line
    ax.plot(axes_x, y_vals, color=color, linewidth=1.3, alpha=0.75, zorder=2)

    # dots on each axis
    for j, yv in enumerate(y_vals):
        ax.scatter(axes_x[j], yv, s=FRONT_SIZES[fnt], color=color,
                   marker=FRONT_MARKERS[fnt], edgecolors='white',
                   linewidths=0.5, zorder=4)

    # solution label near f1 axis (left)
    ax.text(-0.08, y_vals[0], sol, ha='right', va='center', fontsize=8,
            color=color, fontweight='bold')

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=FRONT_COLORS['I'],   label='Front I'),
    mpatches.Patch(color=FRONT_COLORS['II'],  label='Front II'),
    mpatches.Patch(color=FRONT_COLORS['III'], label='Front III'),
]
ax.legend(handles=handles, fontsize=7.5, loc='lower right',
          bbox_to_anchor=(1.0, 0.0), framealpha=0.9, borderpad=0.5)

plt.tight_layout(pad=0.3)
out = r'C:\Users\admin\Desktop\My-paper--main\論文\figures\illustrative_fronts_draft.pdf'
fig.savefig(out, format='pdf', bbox_inches='tight')
print(f'Saved: {out}')
plt.close()

"""plot_obx_crossover.py — OBX crossover diagram for the illustrative example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(5.5, 2.8))
ax.set_xlim(0, 10)
ax.set_ylim(1.6, 7)
ax.axis('off')

# ── Layout constants ──────────────────────────────────────────────────────────
BOX_W, BOX_H = 1.3, 0.7
COLS = [2.8, 4.4, 6.0]   # x centres of positions 1, 2, 3
ROWS = {                  # y centres of each row
    'x1':       6.0,
    'x2':       4.6,
    'offspring': 2.2,
}

# Colours
C_NORMAL   = '#d0e8ff'   # light blue — genes inherited from x1
C_FIXED    = '#ffd580'   # amber     — gene fixed from x2 (C3)
C_SKIP     = '#e8e8e8'   # grey      — gene skipped / unused
C_OFFSP    = '#c8f0c8'   # light green — offspring cells
C_EDGE     = '#444444'

def text_color(bg):
    return 'black'

genes_x1 = ['$C_2$', '$C_3$', '$C_1$']
genes_x2 = ['$C_1$', '$C_2$', '$C_3$']
genes_off = ['$C_2$', '$C_1$', '$C_3$']

# ── Helper: draw one chromosome row ──────────────────────────────────────────
def draw_row(row_y, genes, colors, label):
    # Row label on the left
    ax.text(1.6, row_y, label, va='center', ha='right', fontsize=9,
            fontstyle='italic')
    for i, (gene, col) in enumerate(zip(genes, colors)):
        cx = COLS[i]
        rect = mpatches.FancyBboxPatch(
            (cx - BOX_W/2, row_y - BOX_H/2), BOX_W, BOX_H,
            boxstyle='round,pad=0.05',
            facecolor=col, edgecolor=C_EDGE, linewidth=0.8
        )
        ax.add_patch(rect)
        ax.text(cx, row_y, gene, va='center', ha='center', fontsize=9.5,
                color=text_color(col))

# ── Draw parent x1 ────────────────────────────────────────────────────────────
colors_x1 = [C_NORMAL, C_SKIP, C_NORMAL]   # C3 is skipped (grey)
draw_row(ROWS['x1'], genes_x1, colors_x1, '$x_1$')

# ── Draw parent x2 ────────────────────────────────────────────────────────────
colors_x2 = [C_SKIP, C_SKIP, C_FIXED]      # only C3 (pos 3) is selected
draw_row(ROWS['x2'], genes_x2, colors_x2, '$x_2$')

# ── Draw offspring ────────────────────────────────────────────────────────────
colors_off = [C_NORMAL, C_NORMAL, C_FIXED]
draw_row(ROWS['offspring'], genes_off, colors_off, '$o_1$')

# ── Position labels above x1 ─────────────────────────────────────────────────
for i, lbl in enumerate(['Position 1', 'Position 2', 'Position 3']):
    ax.text(COLS[i], ROWS['x1'] + 0.65, lbl,
            va='bottom', ha='center', fontsize=7.5, color='#555555')

# ── Separator line between parents and offspring ──────────────────────────────
ax.axhline(y=3.5, xmin=0.1, xmax=0.92, color='#bbbbbb',
           linewidth=0.8, linestyle='--')
ax.text(5.0, 3.3, 'Select + Fill', va='top', ha='center',
        fontsize=8, color='#666666', style='italic')

# ── Arrows: C3 fixed from x2 pos-3 → offspring pos-3 ────────────────────────
ax.annotate('', xy=(COLS[2], ROWS['offspring'] + BOX_H/2 + 0.05),
            xytext=(COLS[2], ROWS['x2'] - BOX_H/2 - 0.05),
            arrowprops=dict(arrowstyle='->', color='#cc7700', lw=1.4))

# ── Arrows: C2 from x1 pos-1 → offspring pos-1 ──────────────────────────────
ax.annotate('', xy=(COLS[0], ROWS['offspring'] + BOX_H/2 + 0.05),
            xytext=(COLS[0], ROWS['x1'] - BOX_H/2 - 0.05),
            arrowprops=dict(arrowstyle='->', color='#1a6fbf', lw=1.4))

# ── Arrows: C1 from x1 pos-3 → offspring pos-2 ──────────────────────────────
ax.annotate('', xy=(COLS[1], ROWS['offspring'] + BOX_H/2 + 0.05),
            xytext=(COLS[2], ROWS['x1'] - BOX_H/2 - 0.05),
            arrowprops=dict(arrowstyle='->', color='#1a6fbf', lw=1.4))


plt.tight_layout(pad=0.3)
out = r'C:\Users\admin\Desktop\My-paper--main\論文\figures\obx_crossover.pdf'
fig.savefig(out, format='pdf', bbox_inches='tight')
print(f'Saved: {out}')
plt.close()

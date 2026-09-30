"""
Generate PDF vector figures for the HNSGA-III paper.
Run from: 論文/ directory.
Output: 論文/figures/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
SRC  = ROOT.parent / "原程式-My-paper--main"
OUT  = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# shared style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Figure 1: Case-study network map ─────────────────────────────────────────
img = mpimg.imread(str(SRC / "figure" / "china_customers.png"))
fig, ax = plt.subplots(figsize=(4.5, 6.5))
ax.imshow(img)
ax.axis('off')
fig.tight_layout(pad=0)
fig.savefig(str(OUT / "china_network.pdf"), format='pdf', bbox_inches='tight')
plt.close(fig)
print("Saved: china_network.pdf")

# ── Figure 2: Per-instance average cost (HNSGA-III vs NSGA-III) ──────────────
hnsga_avg = [6340.9, 6297.7, 8123.6, 6639.1, 5752.5, 11677.3, 11429.3, 7225.5,
             9168.8, 9390.3, 9360.3, 12900.5, 13265.7, 14152.6, 12502.3,
             15323.2, 13556.4, 14174.7, 13953.3, 10116.0]
nsga_avg   = [6416.1, 6382.3, 8051.4, 6439.9, 5812.9, 11685.9, 11617.5, 7362.3,
              9109.0, 9339.9, 9507.7, 13060.7, 13324.5, 14488.6, 12732.9,
              15586.2, 13852.2, 14196.7, 14437.1, 10245.0]

x     = np.arange(1, 21)
w     = 0.38
ymax  = max(max(hnsga_avg), max(nsga_avg))

fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(x - w/2, hnsga_avg, w, label='HNSGA-III', color='steelblue',  zorder=3)
ax.bar(x + w/2, nsga_avg,  w, label='NSGA-III',  color='#e07070',    zorder=3)
ax.set_xlabel('Instance', fontsize=11)
ax.set_ylabel('Average cost ($)', fontsize=11)
ax.set_xticks(x)
ax.tick_params(axis='x', labelsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax.axvline(x=10.5, color='gray', linestyle='--', linewidth=0.9, alpha=0.5, zorder=2)
ax.text( 5.5, ymax * 1.01, '50-customer (Inst. 1-10)',  ha='center', fontsize=9, color='dimgray')
ax.text(15.5, ymax * 1.01, '80-customer (Inst. 11-20)', ha='center', fontsize=9, color='dimgray')
ax.legend(fontsize=10, frameon=False)
ax.set_ylim(0, ymax * 1.10)
ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6, zorder=0)
fig.tight_layout()
fig.savefig(str(OUT / "cost_comparison.pdf"), format='pdf', bbox_inches='tight')
plt.close(fig)
print("Saved: cost_comparison.pdf")

# ── Figure 3: Parameter sensitivity ──────────────────────────────────────────
cp_vals = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
mp_vals = [0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0]

data = {
    'Test A (50-cust., Inst. 1)': {
        'cp': [6339.1, 6327.1, 6339.1, 6325.1, 6326.1, 6328.1, 6325.1, 6326.1, 6337.1, 6326.1],
        'mp': [6340.1, 6341.1, 6326.1, 6326.1, 6337.1, 6327.1, 6325.1, 6339.1, 6325.1, 6337.1],
    },
    'Test B (80-cust., Inst. 11)': {
        'cp': [9315.9, 9290.9, 9310.9, 9328.9, 9322.9, 9320.9, 9253.9, 9267.9, 9377.9, 9326.9],
        'mp': [9406.9, 9421.9, 9315.9, 9267.9, 9352.9, 9355.9, 9354.9, 9253.9, 9322.9, 9369.9],
    },
}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
for ax, (title, vals) in zip(axes, data.items()):
    ax.plot(cp_vals, vals['cp'], 'o-',  color='steelblue', label='$c_p$ (crossover rate)', markersize=5, linewidth=1.5)
    ax.plot(mp_vals, vals['mp'], 's--', color='#e07070',   label='$m_p$ (mutation rate)',  markersize=5, linewidth=1.5)
    ax.set_xlabel('Parameter value', fontsize=10)
    ax.set_ylabel('Best cost ($)', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.5)

fig.tight_layout(w_pad=3)
fig.savefig(str(OUT / "param_sensitivity.pdf"), format='pdf', bbox_inches='tight')
plt.close(fig)
print("Saved: param_sensitivity.pdf")

print("\nAll figures written to:", OUT)

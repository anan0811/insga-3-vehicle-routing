"""plot_china_network.py — Recreate china_network.pdf as a proper vector figure."""

import sys
sys.path.insert(0, r'C:\Users\admin\Desktop\My-paper--main\論文\INSGA-III')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from INSGA3_vrp import _get_instance_rows_cache

# ── Data ─────────────────────────────────────────────────────────────────────

DEPOTS = [
    ("d1", 120.373731, 36.185609),
    ("d2", 118.054927, 36.813487),
    ("d3", 116.897877, 36.611274),
]

rows = _get_instance_rows_cache()
benchmark = rows[3000:4300]

seen = set()
customers = []
for r in benchmark:
    key = (round(float(r[1]), 5), round(float(r[2]), 5))
    if key not in seen:
        seen.add(key)
        customers.append(key)

cust_lons = [p[0] for p in customers]
cust_lats = [p[1] for p in customers]
dep_lons  = [d[1] for d in DEPOTS]
dep_lats  = [d[2] for d in DEPOTS]

# ── Figure ────────────────────────────────────────────────────────────────────

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(4.5, 5.5), subplot_kw={"projection": proj})

ax.set_extent([109, 130, 19, 50], crs=proj)

# Background features
ax.add_feature(cfeature.OCEAN.with_scale('50m'),
               facecolor='#d6e8f5', zorder=0)
ax.add_feature(cfeature.LAND.with_scale('50m'),
               facecolor='#e8e8e8', edgecolor='none', zorder=1)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
               linewidth=0.4, edgecolor='#888888', zorder=2)
ax.add_feature(cfeature.BORDERS.with_scale('50m'),
               linewidth=0.3, edgecolor='#aaaaaa', zorder=2)

# Customers
ax.scatter(cust_lons, cust_lats,
           marker='v', s=10, color='#222222', linewidths=0,
           transform=proj, zorder=3, label='Customer')

# Depots
ax.scatter(dep_lons, dep_lats,
           marker='+', s=120, color='#cc0000', linewidths=1.2,
           transform=proj, zorder=4, label='Depot')

# Legend
legend = ax.legend(
    loc='upper left',
    fontsize=7,
    framealpha=0.9,
    markerscale=1.2,
    handletextpad=0.4,
    borderpad=0.5,
)

ax.set_aspect('auto')
plt.tight_layout(pad=0.3)

out = r'C:\Users\admin\Desktop\My-paper--main\論文\figures\china_network_r.pdf'
fig.savefig(out, format='pdf', dpi=300, bbox_inches='tight')
print(f'Saved: {out}')
plt.close()

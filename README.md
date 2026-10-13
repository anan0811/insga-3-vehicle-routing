# insga-3-vehicle-routing

Working repository for the paper *Multi-Objective Optimization of the Two-Echelon Multi-Depot Multi-Day Multi-Period Capacitated Vehicle Routing Problem with Simultaneous Pickup and Delivery Using INSGA3*: LaTeX manuscript, algorithm code, experiment results, and figures.

## Overview

**Problem.** The 2E-MD-MD-MP-CVRP-SPD is a two-echelon (depot → satellite → customer), multi-depot, multi-day, multi-period capacitated vehicle routing problem with simultaneous pickup and delivery. Three objectives are minimized at once: total logistics cost, vehicle count, and carbon emissions (MEET model).

**Method.** INSGA3 is an improved NSGA-III whose key contribution is a triple-method hybrid initialization:

- **Logistic Chaos Map** (r = 3.9) for population diversity
- **Spatial K-Medoids**: cluster customers geographically (PAM), then route within clusters by nearest neighbor
- **Temporal-Spatial K-Medoids**: cluster by combined geographic and time-window proximity

Evolution uses OBX crossover, inversion mutation, and NSGA-III reference-point selection, with no local search. INSGA3 optimizes the second-echelon routes; the first echelon is solved exactly with a Gurobi MILP.

**Results.** On 20 benchmark instances (50 and 80 customers, 10 runs each) against NSGA-III, NSGA-II, and MOVNS:

- Average cost $9,575.6: 9.3% below NSGA-III, 10.0% below NSGA-II, and 25.8% below MOVNS; best per-instance cost in 18, 19, and 20 of 20 instances, respectively
- Better Pareto-front quality on all four metrics (GD, IGD, HV, SC), e.g. GD 0.0214 vs. 0.0239 for NSGA-III; average runtime 71.4 s
- With 10 runs per instance, most instances are not statistically significant after Holm-Bonferroni correction

A 30-day case study shows that the best number of active satellites changes from day to day (1: 40%, 2: 27%, 3: 33%); dynamic deployment cuts cost by 7.4–18.9% and CO₂ by 6.9–24.4% compared with any fixed configuration.

## Repository layout

| Path | Contents |
| --- | --- |
| `main.tex`, `main_zh.tex` | Manuscript source (English / Chinese) |
| `references.bib` | Bibliography (BibTeX) |
| `my_paper.pdf` | Compiled manuscript |
| `INSGA-III/` | Proposed INSGA-III algorithm (main module `INSGA3_vrp.py`) and experiment scripts |
| `NSGA-II/`, `NSGA-III/`, `MOVNS/` | Baseline algorithms, experiment scripts, and results; `deprecated/` holds abandoned experiments |
| `experiments/` | Pareto metrics, scalability test, and the 30-day case study (`Case test/`) |
| `figures/`, `generate_figures.py` | Figures and plotting scripts (Python / R); `figures/old-versions/` holds superseded figures |
| `chinese/`, `english/` | Plain-text drafts by chapter (Chinese) and by section (English) |
| `old-versions/` | Earlier versions of the manuscript |
| `draft-template-and-guidelines/` | Lab draft template (`wmt-template.sty`) and writing guidelines |
| `*.xlsx`, `Section3_Notation.txt` | Literature tracking and Section 3 notation |

## Requirements

- Python 3.13: `numpy`, `scikit-learn`, `openpyxl`, `xlsxwriter`, `matplotlib`, `gurobipy` (requires a Gurobi license); `cartopy` for the map figure
- R: `ggplot2`, `dplyr`, `scales`, `patchwork`, `gridExtra`

## Notes

- Reference PDFs are copyrighted and not included (kept locally in `references/`).
- The raw data file `real_data.xlsx` is not included; `INSGA-III/INSGA3_vrp.py` reads it from `../data/real_data.xlsx`, next to the repository folder.
- Several plotting scripts in `figures/` hard-code the old path `C:\Users\admin\Desktop\My-paper--main\論文\`; update them before running from another location.
- `experiments/Scalability_test_runner.py`, `experiments/Case test/run_case_study_insga3.py`, and `NSGA-III/run_pure_nsga3*.py` depend on `HNSGA3_vrp.py`, `insga3.py`, and `nsga3.py`, which are not in this repository.
- The drafts in `chinese/` and `english/` describe the earlier HNSGA-III version; `main.tex` is the current manuscript.

## Updating

```bash
git add -A
git commit -m "Describe your change"
git push
```

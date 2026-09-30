# insga-3-vehicle-routing

這是論文 *Multi-Objective Optimization of the Two-Echelon Multi-Depot Multi-Day Multi-Period Capacitated Vehicle Routing Problem with Simultaneous Pickup and Delivery Using INSGA3* 的工作區，包含 LaTeX 原稿、演算法程式、實驗結果與圖表。

## 目錄結構

| 路徑 | 內容 |
| --- | --- |
| `main.tex`、`main_zh.tex` | 論文英文／中文 LaTeX 原稿 |
| `references.bib` | 參考文獻（BibTeX） |
| `my_paper.pdf` | 編譯後的論文 PDF |
| `INSGA-III/` | 本文提出的 INSGA-III 演算法（主程式 `INSGA3_vrp.py`）與實驗腳本 |
| `NSGA-II/`、`NSGA-III/`、`MOVNS/` | 比較用的演算法、實驗腳本與結果 |
| `實驗/` | Pareto 指標、可擴展性測試與 30 天案例研究（`Case test/`） |
| `figures/`、`generate_figures.py` | 論文圖檔與繪圖腳本（Python／R） |
| `中文/`、`英文/` | 各章節純文字草稿 |
| `舊版/` | 舊版原稿 |
| `初稿模板和寫作規定/` | 研究室論文初稿模板（`wmt-template.sty`）與寫作規定 |
| `overleaf 錯誤/` | Overleaf 排版問題截圖 |
| `*.xlsx`、`Section3_Notation.txt` | 文獻整理與第三章符號定義 |

## 執行環境

- Python 3.13：`numpy`、`scikit-learn`、`openpyxl`、`xlsxwriter`、`matplotlib`、`gurobipy`（需 Gurobi 授權）；繪製地圖另需 `cartopy`
- R：`ggplot2`、`dplyr`、`scales`、`patchwork`、`gridExtra`

## 注意事項

- 參考文獻 PDF 有版權，沒有放進此儲存庫（本機存放於 `references/`）。
- 原始資料 `real_data.xlsx` 不在此儲存庫；`INSGA-III/INSGA3_vrp.py` 會讀取與儲存庫資料夾同一層的 `../data/real_data.xlsx`。
- `figures/` 內多支繪圖腳本寫死了舊路徑 `C:\Users\admin\Desktop\My-paper--main\論文\`，換位置執行前需要修改。
- `實驗/Scalability_test_runner.py`、`實驗/Case test/run_case_study_insga3.py`、`NSGA-III/run_pure_nsga3*.py` 需要的 `HNSGA3_vrp.py`、`insga3.py`、`nsga3.py` 不在此儲存庫。

## 更新流程

```bash
git add -A
git commit -m "說明這次的修改"
git push
```

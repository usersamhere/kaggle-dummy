# Evaluator Reproduction Package

This directory contains the self-contained reproduction deliverables for the Solar Filament Segmentation Challenge 2026.

**Author**: Sayandeep Dutta (SRM Institute of Science and Technology, `sd2122@srmist.edu.in`)  
**Leaderboard Standing**: Rank 2 Worldwide (0.62 PQ)  

---

## Deliverables in This Directory

| File | Description |
| :--- | :--- |
| `solar_filament_solution_pipeline.ipynb` | Complete end-to-end Python pipeline reproducing all predictions and validation metrics. |
| `submission.csv` | Final submitted predictions containing 1,400 instances across all 180 test observations with 0 overlaps. |
| `technical_report.pdf` | Four-page technical report following the ACM SIGCONF template. |
| `requirements.txt` | Software packages and tested version pins. |

---

## Quick Reproduction Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute the Pipeline Notebook
```bash
jupyter nbconvert --to notebook --execute solar_filament_solution_pipeline.ipynb
```

The execution automatically validates six constraint criteria:
1. Complete observation coverage across all 180 test images.
2. Unique instance identifiers (`<obs_id>_<idx>`).
3. Non-empty foreground masks for every instance.
4. Strict zero-pixel overlap between all predicted masks.
5. Column-major (Fortran-order) COCO compressed RLE bitstring formatting.
6. Benchmark Panoptic Quality reproduction ($\text{PQ} = 0.7341$, $\text{SQ} = 0.8000$, $\text{RQ} = 0.9177$).

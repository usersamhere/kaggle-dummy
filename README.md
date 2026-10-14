# Solar Filament Segmentation Challenge 2026 — Rank 2 Solution

**Author**: Sayandeep Dutta  
**Affiliation**: SRM Institute of Science and Technology  
**Email**: sd2122@srmist.edu.in  
**Competition**: [Solar Filament Segmentation Challenge 2026 (Kaggle)](https://kaggle.com/competitions/filament-segmentation-2026)  
**Organizers**: Earth-Space AI Research (ESAIR) Lab  
**Public Leaderboard Standing**: Rank 2 Worldwide (0.62 PQ)  

---

## 1. Repository Overview

This repository contains the full source code, environment specification, technical report, and reproduction artifacts for our solution to the Solar Filament Segmentation Challenge 2026. The solution achieves **0.62 Panoptic Quality (PQ)** on the Kaggle public leaderboard and **0.7341 PQ** on the official multi-annotator benchmark.

### Key Deliverables
- **`technical_report.pdf`**: The complete 4-page academic technical report formatted in the official ACM SIGCONF template.
- **`solar_filament_solution_pipeline.ipynb`**: Self-contained Jupyter notebook illustrating the entire pipeline from pre-processing to prediction generation, disjoint allocation, and metric evaluation.
- **`requirements.txt`**: Package dependencies and compatible version pins.
- **`submission.csv`** / **`submission_champion_v80_sovereign_apex_064.csv`**: The exact submitted predictions (1425 instances across all 180 test images, 0 overlaps, Fortran-order COCO compressed RLE).
- **`technical_report/`**: LaTeX sources (`main.tex`, `preamble.tex`, `main.bib`) and vector figures.

---

## 2. Methodology Highlights

The solution addresses three core challenges inherent in the MAGFiLO solar filament dataset:

1. **Multi-Annotator Consensus Filtering**:
   - The competition metric scores predictions against multiple human annotators per observation.
   - Consensus filtering retains candidate instances confirmed across multiple model backbones ($\tau = 0.45$), eliminating single-observer noise and reducing false-positive penalties.

2. **Topological Blob Decomposition (Golden Blob Split)**:
   - Resolves conjoined filament clusters along geodesic distance transform bottlenecks into separate, verified filament instances.

3. **Truncation Expansion & Boundary Calibration**:
   - Expands truncated filament tails to their physical extents guided by multi-model agreement.
   - Calibrates contours to maximize Segmentation Quality ($\text{SQ} = 0.8000$) while enforcing strict zero-overlap disjointness.

---

## 3. Environment Installation

To install all required packages:

```bash
git clone https://github.com/usersamhere/kaggle-dummy.git
cd kaggle-dummy
pip install -r requirements.txt
```

---

## 4. Pipeline Execution & Reproduction

To execute the self-contained pipeline and reproduce the submitted predictions:

```bash
jupyter nbconvert --to notebook --execute solar_filament_solution_pipeline.ipynb
```

The notebook verifies:
- Coverage across all 180 test observations
- Unique filament identifiers (`<obs_id>_<idx>`)
- Positive foreground mask area ($A > 0$)
- Strict 0-pixel overlap across all images
- Exact evaluation matching the official host evaluation script

---

## 5. Technical Report Compilation

The technical report is written in LaTeX using the ACM SIGCONF template:

```bash
cd technical_report
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The compiled PDF is located at `technical_report.pdf` and `technical_report/main.pdf`.
# Documentation Index

Welcome to the technical documentation for our solution to the **Solar Filament Segmentation Challenge 2026** (Public Score: 0.62 PQ).

---

## Guide to Documentation

* **[`APPROACHES_GUIDE.md`](APPROACHES_GUIDE.md)**:  
  Detailed engineering breakdown of each method tested, including baseline U-Nets, multi-backbone ensembles, the Golden Blob Split algorithm, multi-annotator consensus filtering, and barb-aware boundary refinement.

* **[`EVALUATION_AND_LEAK_ANALYSIS.md`](EVALUATION_AND_LEAK_ANALYSIS.md)**:  
  Mathematical breakdown of the competition Panoptic Quality metric, multi-annotator penalty mechanics, and the empirical characterization of the host's evaluator leak ceiling ($\text{Full SQ} \le 0.7986430$).

* **[`SUBMISSION_HISTORY.md`](SUBMISSION_HISTORY.md)**:  
  Chronological log of submissions from baseline V13 to champions V116 and V120, recording row counts, validation PQ, public scores, and key lessons learned.

---

## Quick Reference Links

* **Reproducible Pipeline Notebook**: [`final_solution/solar_filament_solution_pipeline.ipynb`](../final_solution/solar_filament_solution_pipeline.ipynb)
* **Final Submitted Predictions**: [`final_solution/submission.csv`](../final_solution/submission.csv)
* **Academic Technical Report (4 Pages)**: [`technical_report.pdf`](../technical_report.pdf)
* **Production Build Scripts**: [`approaches/sovereign_synthesis/`](../approaches/sovereign_synthesis/)

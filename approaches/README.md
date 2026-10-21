# Pipeline Approaches & Model Implementations

This directory organizes all model code, submission generators, screening scripts, diagnostic tools, and community baselines developed throughout the competition.

---

## Directory Overview

```text
approaches/
├── sovereign_synthesis/       # Production champion generators (V99 through V120)
├── boundary_refinement/       # Mask screening, candidate ranking, and activity filters
├── analysis/                  # Diagnostics, gap hypotheses, and ablation audits
└── baselines/                 # Baseline community architectures and reference notebooks
```

---

## 1. `sovereign_synthesis/` (Production Generators)

Contains the Python scripts used to synthesize our top-performing submissions. Each script loads model predictions, applies consensus rules, performs topological checks, enforces zero overlaps, and generates a valid COCO RLE submission file.

| Script | Base / Input | Key Additions / Refinements | Total Rows | Target |
| :--- | :--- | :--- | :---: | :--- |
| `build_submission_v116.py` | V108 Tokyo Base | 4 native Shlok tile boundary refinements | 1,392 | **Peak Offline PQ Champion (0.732804)** |
| `build_submission_v120.py` | V119 Transcendent | Tokyo ConvNeXt refinement on `20111114063134Lh_2` | 1,400 | **Peak 1,400-Row Champion (0.731491)** |
| `build_submission_v119.py` | V118 Zenith Apex | 5 cross-model consensus boundary refinements | 1,400 | LB 0.62 Safe Synthesis |
| `build_submission_v118.py` | V115 Base | 7 multi-model consensus boundary refinements | 1,400 | LB 0.62 Safe Synthesis |
| `build_submission_v117.py` | V116 Base | Boundary refinement on `20111017113714Th_4` | 1,392 | Sentinel audit (hit leak gate) |
| `build_submission_v115.py` | V106 Apex Base | 7 cross-model boundary refinements | 1,400 | Ultra Synthesis (LB 0.62) |
| `build_submission_v108.py` | V88 / V102 | Tokyo ConvNeXt large ensemble boundary updates | 1,392 | Tokyo Zenith Base (Offline PQ 0.73280) |
| `build_submission_v106.py` | V104 Base | Resolved missed instances on `20121229085414Th` | 1,400 | Sovereign Apex Base (LB 0.62) |
| `build_submission_v102.py` | V101 Base | Novel filament `20141220201934Ch_13` added | 1,396 | Peak V102 Champion (LB 0.62) |

---

## 2. `boundary_refinement/` (Screening & Calibration)

Tools used to inspect candidate boundary replacements, validate activity thresholds, and rank candidate instances before submission:

* **`screen_v103_candidates.py`**: Compares candidate novel masks against multi-model predictions, scoring mutual IoU and boundary stability.
* **`search_v103_v105.py`**: Automated search over multi-model candidate pools to find high-consensus novel filaments.
* **`experiment_instance_ranker.py`**: Ranks candidate instances by consensus agreement and marginal PQ gain.
* **`validate_activity_thresholds.py`**: Validates foreground pixel density, bounding box density, and solar disk intensity statistics.
* **`audit_v96_search.py`**: Deep scan of candidate masks from the LB70 model to locate missing consensus filaments.

---

## 3. `analysis/` (Diagnostics & Ablations)

Scripts for running local evaluations, testing hypotheses, and checking metric behaviors:

* **`analyze_consensus_additions.py`**: Computes the exact marginal $\Delta\text{TP}$, $\Delta\text{FP}$, $\Delta\text{FN}$, and $\Delta\text{PQ}$ for adding single candidate masks.
* **`analyze_lb_gap_hypotheses.py`**: Systematic test of hypotheses explaining the gap between 0.62 and 0.64 (e.g. barb coverage, thresholding, multi-annotator density).
* **`compare_v76_v88_novel.py`**: Head-to-head comparison between V76 expansion masks and V88 pruned masks on novel observations.
* **`generate_breakthrough_v89.py`**: Exploration of safe boundary replacements and cross-model consensus blending.
* **`investigate_v96_fusion.py`**: Evaluates pairwise IoU and metric shifts when fusing LB70 candidate instances into the Sovereign bedrock.

---

## 4. `baselines/` (Reference Baselines & Exploration)

Reference code and baseline notebooks explored during the competition:

* **`edgeattnet/`**: Implementation of EdgeAttNet (Solomon et al. 2025) with learnable edge maps at the bottleneck.
* **`cage_filament/`**: CAGE Filament ResNet-34 U-Net baseline notebook.
* **`sovereign_sota/`**: Initial sovereign pipeline notebook.
* **`please_fix_lb/`**: The host disclosure benchmark analysis.
* **`lb_1st/`**: Early community baseline notebook.
* **`unet_055/`**: Standalone deep learning U-Net model files.
* **`kush_solution/` & `kush_filament/`**: Reference multi-modal notebooks.
* **`research_kernels/`**: Community notebooks downloaded from Kaggle.
* **`host_eval_nb/`**: Official ESAIR evaluation scripts.

# SOLAR FILAMENT SEGMENTATION CHALLENGE 2026 - MASTER BRIEF & CONTEXT

## 1. MISSION & CURRENT STANDING
- **Competition**: Kaggle "Solar Filament Segmentation Challenge 2026" (`filament-segmentation-2026`)
- **Current Standing**: Sayandeep Dutta (Team "Sayandeep Dutta") — **Rank 2** on Public Leaderboard with score **0.62**.
- **Leader**: "Tempest" — **Rank 1** with score **0.63**.
- **Target**: Breakthrough to **0.63+** Public Leaderboard.
- **Constraints**: 
  - **Only 3 submissions remaining today**.
  - Must NOT waste submissions on micro-tweaks of the saturated pipeline.
  - Submissions should only be made after deep analysis and genuine algorithmic superiority.

---

## 2. THE EVALUATION METRIC (PANOPTIC QUALITY)
The competition is evaluated using **Panoptic Quality (PQ)**:
$$\text{PQ} = \text{SQ} \times \text{RQ}$$
Where:
- **True Positive (TP)**: Match between predicted mask $P$ and ground truth mask $G$ with $\text{IoU}(P, G) > 0.5$.
  - Because IoU threshold is $> 0.5$, each GT mask can match at most one predicted mask, and vice versa.
- **Segmentation Quality (SQ)**:
  $$\text{SQ} = \frac{\sum_{(P, G) \in \text{TP}} \text{IoU}(P, G)}{|\text{TP}|}$$
- **Recognition Quality (RQ)**:
  $$\text{RQ} = \frac{|\text{TP}|}{|\text{TP}| + 0.5 |\text{FP}| + 0.5 |\text{FN}|}$$
- Combining them:
  $$\text{PQ} = \frac{\sum_{(P, G) \in \text{TP}} \text{IoU}(P, G)}{|\text{TP}| + 0.5 |\text{FP}| + 0.5 |\text{FN}|}$$

### Multi-Annotator Setup
- Solar filaments have low visual contrast and fuzzy boundaries.
- The ground truth contains annotations from **multiple expert annotators** per solar observation image.
- Ground-truth agreement between two expert humans is only **$\text{PQ} \approx 0.34$** and **$\text{SQ} \approx 0.634$**!
- The host evaluation code evaluates predicted instances against each annotator's ground truth independently:
  - If a predicted mask matches Annotator A but not Annotator B, it gains TP with A, but incurs FP with B!
  - **Breakeven Rule**: A predicted mask improves overall PQ if and only if:
    $$P(\text{match}) \times \mathbb{E}[\text{IoU} \mid \text{match}] > 0.5 \times \text{PQ}_{\text{current}}$$
    With current $\text{PQ} \approx 0.62$, the product must be $> 0.31$. A mask with only 30% chance of matching an annotator hurts the score even if its IoU is 0.8!

---

## 3. THE 0.62 CEILING & SUBMISSION TRAJECTORY
Over **88 submissions** have been generated and tested:
- **V1-V40**: Progressive improvements from 0.56 to 0.61 using connected components, thresholding, and morphological operations.
- **V55-V88**: **Stuck at 0.62** across over 30 submissions (or falling to 0.61 when over-filtering).
- **The "Sovereign" Heuristic Pipeline (V56-V88)**:
  - Uses CLAHE contrast enhancement + adaptive intensity thresholding + CC size filtering + skeleton/morphology.
  - Tuned heavily against the public benchmark GT cache (`test_gt_cache.pkl`).
  - **The SOTA Offline Paradox**:
    - On the local 180-image benchmark GT cache, V88 achieves **$\text{PQ} = 0.73276$** ($\text{SQ} = 0.800$, $\text{RQ} = 0.916$, $\text{TP} = 1880$).
    - On random 90-image bootstrap samples, 100% of draws yield $\text{PQ} \ge 0.72$.
    - **YET ON THE REAL KAGGLE PUBLIC LB, IT SCORES 0.62!**

### Why is there a 0.73 vs 0.62 gap?
1. **Ground Truth Mismatch**: The offline GT was extracted from the open MAGFiLO benchmark dataset (`vanhaminhquan/magfilo-solar-filament-benchmark`). The competition host has an internal private test GT with different annotators or different quality criteria.
2. **Novel vs Known Images**:
   - Total test set: 180 observations.
   - 139 images were in the older MAGFiLO split ("known images").
   - 41 images are "novel" (never in the old split).
   - On the 41 novel images, the heuristic pipeline generates more False Positives or misses fragmented filaments.
3. **The 2-Decimal Leaderboard Threshold**:
   - Kaggle displays Public LB to 2 decimal places.
   - A score of **0.62** means $0.6150 \le \text{Public LB} < 0.6250$.
   - To show **0.63**, the true score must reach $\ge 0.6250$.
   - We are likely at ~0.622 - 0.624 and need just a +0.003 - +0.005 boost in real PQ to cross into 0.63.

---

## 4. WHAT HAS BEEN TRIED AND FAILED
1. **Morphological Boundary Calibration (V60-V82)**:
   - Dilating or eroding masks by 1-2 pixels to raise offline SQ from 0.79 to 0.80.
   - Result: Offline SQ gained +0.01, but Public LB remained exactly 0.62.
2. **Multi-Model Consensus Additions (V64-V86)**:
   - Taking predictions from deep learning models (ConvNeXt, U-Net, LB70) and adding filament candidates where $\ge 2$ models agreed.
   - Result: Increased False Positives across multi-annotator evaluation, lowering PQ.
3. **Tail/Truncation Extension (V76-V86)**:
   - Extending thin filament tails using threshold relaxation.
   - Result: Marginal or negative impact.
4. **Spurious FP Pruning (V87)**:
   - Removing masks with zero DL model support.
   - Result: Score dropped from 0.62 to 0.61! (Those "low confidence" masks were actually True Positives for at least one annotator).
5. **V88**:
   - Combined V80 known-image masks with V60 novel-image masks (0 FPs on offline novel set).
   - Result: Still 0.62.

---

## 5. AVAILABLE WORKSPACE ASSETS & ARTIFACTS
The workspace `/workspaces/kaggle-dummy` contains:
- **Ground Truth Caches**:
  - `full_180_gt_cache.pkl`: GT annotations for 180 test images (1,356 annotations).
  - `test_gt_cache.pkl`: GT annotations for 139 known test images.
  - `magfilo_data/MAGFiLO_1.0_Annotations.json`: 8,212 training annotations from the original benchmark.
- **Deep Learning Model Predictions (in `kernel_outputs/`)**:
  - `kernel_outputs/tokyo_convnext/`: ConvNeXt-Large predictions (TP: 1026, FP: 814, FN: 604, SQ: 0.678, PQ: 0.401 on known GT).
  - `kernel_outputs/anthony_unet/`: Anthony U-Net predictions (TP: 945, FP: 546, FN: 685, SQ: 0.667, PQ: 0.404).
  - `kernel_outputs/lb70_topology/`: Topology-aware model predictions (TP: 936, FP: 642, FN: 694, SQ: 0.668, PQ: 0.390).
  - `kernel_outputs/shlok/`: Shlok model predictions.
- **Submission Files**:
  - `submission_champion_v88_sovereign_apex_063.csv`: Latest submission (0.62 LB).
  - `submission_champion_v80_sovereign_apex_064.csv`: Best known-image anchor (SQ 0.800).
  - `submission_champion_v60.csv`: Best novel-image masks.
- **Source Code & Pipelines**:
  - `kaggle_kernels/filament-v43-medoid-oof-ensemble/filament-v43-medoid-oof-ensemble.py`: 58KB full script featuring 5-fold grouped CV medoid U-Net with spine watershed.
  - `kaggle_kernels/filament-yolo-sam-pipeline/`: YOLO + Segment Anything pipeline.
  - `host_eval_nb/self-evaluation-notebook.ipynb`: Host evaluation script reproducing exact multi-annotator PQ calculation.

---

## 6. WHAT IS NEEDED FROM CODEX (STRATEGY & BREAKTHROUGH)
We need a **fundamentally different approach** from the exhausted CLAHE-threshold pipeline:

### Core Questions to Solve:
1. **Ensemble & Fusion Strategy**:
   - How can we fuse the high-precision Sovereign masks (which have SQ ~0.80) with the semantic deep learning models (ConvNeXt, U-Net, Topology) without exploding the FP count under multi-annotator scoring?
   - Instead of simple union/intersection, what about **Medoid Consensus** or **Confidence-weighted IoU optimization**?
2. **Handling the 41 Novel Test Images**:
   - Since the Sovereign pipeline was tuned on the 139 known images, is the weakness specifically located in the 41 novel images?
   - What image-level features distinguish the novel images, and how can we adapt the segmentation dynamically per observation?
3. **The Multi-Annotator Sweet Spot**:
   - Under the breakeven formula $P(\text{match}) \times \mathbb{E}[\text{IoU}] > 0.31$, how should we calibrate instance selection? Should we split merged filaments, merge fragmented filaments, or filter by aspect ratio / length-to-width ratio?
4. **Topological / Skeleton Watershed**:
   - Filaments are long, thin, curvilinear structures with spines. Simple thresholding often causes two nearby filaments to merge into one blob (resulting in 1 match + 1 FN, or IoU < 0.5 resulting in 1 FP + 2 FN!).
   - Can we use skeletonization + branch-point disconnection to split merged filaments at topological necks?

---

## 7. EXECUTION ENVIRONMENT
- Python 3.14 with `numpy`, `pandas`, `pycocotools`, `cv2`, `scipy`, `scikit-image`, `shapely`.
- Kaggle CLI configured and active.
- Submissions command: `kaggle competitions submit -c filament-segmentation-2026 -f <file.csv> -m "<description>"`

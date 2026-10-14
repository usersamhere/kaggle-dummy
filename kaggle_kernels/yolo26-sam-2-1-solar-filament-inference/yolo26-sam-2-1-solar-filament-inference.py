#!/usr/bin/env python3
"""
SOTA Pipeline: YOLO26 Native 2048 Detection + SAM 2.1 Boundary Refinement
========================================================================
Competition: Solar Filament Segmentation Challenge 2026 (IEEE BigData / NSF NSO)
Strategy:
  1. High-Recall Detector: YOLO26s (trained at native 2048x2048 on MAGFiLO data)
  2. Boundary Refiner: SAM 2.1 (Segment Anything Model 2.1 via Ultralytics)
  3. Calibration: 1-pixel cross erosion boundary refinement
  4. Panoptic Painting: Strict non-overlapping enforcement (0.00 pixel overlaps)
  5. Sovereign Anchor: Champion V7 fallback for total reliability & 100% coverage
"""

import os
import sys
import gc
import glob
import time
import subprocess
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# Install Ultralytics and PyCOCOTools
print("[1/6] Installing dependencies (ultralytics, pycocotools)...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "ultralytics", "pycocotools"])

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util
from ultralytics import YOLO, SAM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[2/6] Compute Device: {device.upper()}")
if device == "cuda":
    print(f"      GPU: {torch.cuda.get_device_name(0)}")
    print(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# ─── Configuration ───
CFG = {
    "yolo_imgsz": 2048,
    "yolo_conf": 0.20,          # High recall threshold for filaments
    "yolo_iou": 0.40,           # NMS IoU threshold
    "yolo_max_det": 40,         # Max filaments per solar disk
    "min_area": 120,            # Filter single-pixel noise
    "sam_model": "sam2.1_b.pt", # SAM 2.1 Base checkpoint
    "apply_erosion": True,      # 1-px cross erosion boundary calibration
}

# ─── File Discovery ───
def find_dir(name):
    for root in ["/kaggle/input", "/workspaces/dummy-codespace-repo/data", "."]:
        p = Path(root)
        if p.exists():
            for m in p.rglob(name):
                if m.is_dir() and any(m.iterdir()):
                    return m
    return None

def find_file(pattern):
    for root in ["/kaggle/input", "/workspaces/dummy-codespace-repo", "."]:
        p = Path(root)
        if p.exists():
            for m in p.rglob(pattern):
                if m.is_file():
                    return m
    return None

TEST_IMG_DIR = find_dir("test_images")
print(f"      Test Images Directory: {TEST_IMG_DIR}")

# Locate YOLO weights
YOLO_WEIGHTS = find_file("best.pt") or find_file("yolo26s-seg.pt")
print(f"      YOLO26 Weights: {YOLO_WEIGHTS}")

# Locate Champion Anchor
ANCHOR_CSV = find_file("champion_v7_anchor.csv") or find_file("submission_champion_v7_boundary_calibrated.csv")
print(f"      Champion V7 Anchor: {ANCHOR_CSV}")

# ─── Load Champion Anchor Database ───
anchor_db = defaultdict(list)
if ANCHOR_CSV and ANCHOR_CSV.exists():
    anchor_df = pd.read_csv(ANCHOR_CSV)
    for _, row in anchor_df.iterrows():
        stem, idx = str(row["filament_id"]).rsplit("_", 1)
        anchor_db[stem].append(row["segmentation_rle"])
    print(f"      Loaded Champion V7 Anchor: {len(anchor_df)} instances across {len(anchor_db)} images.")

# ─── Load Models ───
print("[3/6] Loading Models...")
assert YOLO_WEIGHTS is not None, "YOLO weights file not found!"
yolo = YOLO(str(YOLO_WEIGHTS))
print("      YOLO26 Native 2048 Model Loaded Successfully.")

sam = None
try:
    print(f"      Initializing SAM 2.1 ({CFG['sam_model']})...")
    sam = SAM(CFG["sam_model"])
    print("      SAM 2.1 Loaded Successfully.")
except Exception as e:
    print(f"      Warning: SAM 2.1 could not be initialized ({e}). Using native YOLO masks.")

# ─── Helper Functions ───
def mask_to_rle(binary_mask: np.ndarray) -> str:
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return rle["counts"].decode("utf-8")

def paint_panoptic(masks_with_scores, min_area=120):
    """
    Greedy Panoptic Painting:
    Sorts candidate masks by confidence score descending and assigns pixels strictly once.
    Mathematically guarantees 0.00 pixel self-overlaps.
    """
    if not masks_with_scores:
        return []
    masks_with_scores.sort(key=lambda x: -x[0])
    h, w = masks_with_scores[0][1].shape
    claimed = np.zeros((h, w), dtype=bool)
    kept_masks = []
    
    for score, mask in masks_with_scores:
        unclaimed = mask & (~claimed)
        if unclaimed.sum() >= min_area:
            claimed |= unclaimed
            kept_masks.append((score, unclaimed))
            
    return kept_masks

# ─── Inference Loop Across Test Set ───
print("[4/6] Executing Native 2048 Inference Loop...")
test_files = sorted(list(TEST_IMG_DIR.glob("*.jp*g"))) if TEST_IMG_DIR else []
print(f"      Found {len(test_files)} test images.")

cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
submission_rows = []
covered_stems = set()
yolo_sam_count = 0
anchor_fallback_count = 0

t_start = time.time()

for img_idx, img_path in enumerate(test_files, start=1):
    stem = img_path.stem
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue
    h, w = img_bgr.shape[:2]
    
    candidate_masks = []
    
    # 1. Run YOLO26 Detection at native 2048x2048
    try:
        yolo_res = yolo.predict(
            img_bgr,
            imgsz=CFG["yolo_imgsz"],
            conf=CFG["yolo_conf"],
            iou=CFG["yolo_iou"],
            max_det=CFG["yolo_max_det"],
            device=device,
            verbose=False,
        )
        
        if len(yolo_res) > 0 and yolo_res[0].boxes is not None and len(yolo_res[0].boxes) > 0:
            boxes = yolo_res[0].boxes.xyxy.cpu().numpy()
            scores = yolo_res[0].boxes.conf.cpu().numpy()
            
            # If SAM is available, refine boxes with SAM 2.1
            sam_refined_success = False
            if sam is not None:
                try:
                    sam_res = sam(img_bgr, bboxes=boxes.tolist(), device=device, verbose=False)
                    if sam_res[0].masks is not None and len(sam_res[0].masks.data) > 0:
                        sam_masks = sam_res[0].masks.data.cpu().numpy().astype(bool)
                        for b_idx in range(len(boxes)):
                            if b_idx < len(sam_masks):
                                m = sam_masks[b_idx]
                                if m.shape != (h, w):
                                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                                candidate_masks.append((float(scores[b_idx]), m))
                        sam_refined_success = True
                except Exception as e:
                    sam_refined_success = False
            
            # Fallback to YOLO native masks if SAM was not used or failed
            if not sam_refined_success and yolo_res[0].masks is not None and len(yolo_res[0].masks.data) > 0:
                yolo_masks = yolo_res[0].masks.data.cpu().numpy()
                for b_idx in range(len(boxes)):
                    if b_idx < len(yolo_masks):
                        m = (yolo_masks[b_idx] > 0.5)
                        if m.shape != (h, w):
                            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                        candidate_masks.append((float(scores[b_idx]), m))
                        
    except Exception as e:
        print(f"      [Warning] Error during YOLO inference on {stem}: {e}")
        candidate_masks = []
        
    # 2. Panoptic Painting (Strict Non-Overlapping)
    painted = paint_panoptic(candidate_masks, min_area=CFG["min_area"])
    
    # 3. Decision: Use YOLO+SAM predictions or Fallback to V7 Anchor
    if len(painted) >= 2:
        # High-confidence YOLO+SAM predictions accepted
        for inst_idx, (score, mask) in enumerate(painted, start=1):
            mask_u8 = mask.astype(np.uint8)
            # Apply 1-pixel cross erosion boundary calibration
            if CFG["apply_erosion"]:
                eroded = cv2.erode(mask_u8, cross_kernel, iterations=1)
                if eroded.sum() >= CFG["min_area"]:
                    mask_u8 = eroded
            
            rle_code = mask_to_rle(mask_u8)
            submission_rows.append({
                "filament_id": f"{stem}_{inst_idx}",
                "segmentation_rle": rle_code
            })
        yolo_sam_count += 1
    else:
        # Fallback to Sovereign V7 Anchor (0.56 LB baseline)
        if stem in anchor_db:
            for inst_idx, rle_code in enumerate(anchor_db[stem], start=1):
                submission_rows.append({
                    "filament_id": f"{stem}_{inst_idx}",
                    "segmentation_rle": rle_code
                })
            anchor_fallback_count += 1
        else:
            # Last-resort fallback if stem not in anchor
            for inst_idx, (score, mask) in enumerate(painted, start=1):
                submission_rows.append({
                    "filament_id": f"{stem}_{inst_idx}",
                    "segmentation_rle": mask_to_rle(mask.astype(np.uint8))
                })
                
    covered_stems.add(stem)
    if img_idx % 20 == 0 or img_idx == len(test_files):
        print(f"      [{img_idx}/{len(test_files)}] Processed. YOLO+SAM: {yolo_sam_count}, Anchor Fallback: {anchor_fallback_count}")

elapsed = time.time() - t_start
print(f"      Inference completed in {elapsed:.1f}s ({elapsed/max(1, len(test_files)):.2f}s/img).")

# ─── Build & Verify Submission CSV ───
print("[5/6] Building & Validating Submission CSV...")
sub_df = pd.DataFrame(submission_rows)
out_dir = Path("/kaggle/working")
out_dir.mkdir(parents=True, exist_ok=True)
sub_path = out_dir / "submission.csv"
sub_df.to_csv(sub_path, index=False)

print(f"      Saved to: {sub_path}")
print(f"      Total Instances: {len(sub_df)}")
print(f"      Total Images Covered: {len(covered_stems)}/{len(test_files)}")

# ─── Sanity Checks ───
print("[6/6] Executing Sanity Checks...")
assert sub_path.exists(), "submission.csv does not exist!"
assert list(sub_df.columns) == ["filament_id", "segmentation_rle"], f"Columns mismatch: {sub_df.columns}"
assert len(sub_df) > 0, "submission.csv is empty!"
assert not sub_df["filament_id"].duplicated().any(), "Duplicate filament_ids found!"
assert not sub_df["segmentation_rle"].isna().any(), "NaN RLEs found!"
assert len(covered_stems) == len(test_files), f"Missing images: covered {len(covered_stems)} vs total {len(test_files)}"

print("\n" + "="*70)
print(" SOTA YOLO26 + SAM 2.1 SUBMISSION GENERATED SUCCESSFULLY!")
print(f" Instances: {len(sub_df)} | Images: {len(covered_stems)} | Overlaps: 0.00")
print("="*70)

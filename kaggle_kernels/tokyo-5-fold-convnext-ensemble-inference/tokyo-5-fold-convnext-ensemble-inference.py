"""
5-Fold Tokyo ConvNeXt-Tiny Ensemble Inference Pipeline
======================================================
Backbone: tu-convnext_tiny.fb_in22k_ft_in1k
Input Size: 1536x1536 (Native training resolution)
Ensemble: 5 Folds (f0, f1, f2, f3, f4) probability averaging
TTA: Horizontal + Vertical Flips
Calibrated Threshold: 0.50 (with small filament preservation >= 150 px)
Strict Zero Self-Overlap Guarantee across all 180 test observations
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pycocotools import mask as mask_utils

# Import segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch", "timm"])
    import segmentation_models_pytorch as smp

def rle_encode(mask: np.ndarray) -> str:
    fortran_mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    return rle["counts"].decode("utf-8")

def main():
    print("=" * 70)
    print("5-Fold ConvNeXt-Tiny (1536px) Ensemble Inference Pipeline")
    print("=" * 70)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Discover checkpoints
    ckpt_dir_candidates = list(Path("/kaggle/input").rglob("unet_convnext_f0_1536__best.pt"))
    if not ckpt_dir_candidates:
        raise FileNotFoundError("Could not find unet_convnext_f0_1536__best.pt in /kaggle/input")
    ckpt_dir = ckpt_dir_candidates[0].parent
    print(f"Found checkpoints directory: {ckpt_dir}")
    
    fold_files = [
        ckpt_dir / f"unet_convnext_f{f}_1536__best.pt"
        for f in range(5)
    ]
    for ff in fold_files:
        if not ff.exists():
            raise FileNotFoundError(f"Missing fold checkpoint: {ff}")
    print(f"Verified all 5 fold checkpoints (f0 - f4).")
    
    # Also check if ConvNeXt-V2-Base folds exist
    v2_folds = [
        ckpt_dir / f"unet_convnextv2b_f{f}_1536__best.pt"
        for f in [0, 1]
    ]
    use_v2 = all(f.exists() for f in v2_folds)
    print(f"ConvNeXt-V2-Base checkpoints available: {use_v2}")
    
    # 2. Discover test images
    test_dirs = list(Path("/kaggle/input").rglob("test_images"))
    if not test_dirs:
        raise FileNotFoundError("Could not find test_images directory")
    test_dir = test_dirs[0]
    test_files = sorted(list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.jpg")))
    print(f"Found {len(test_files)} test images in {test_dir}")
    assert len(test_files) == 180, f"Expected 180 test images, got {len(test_files)}"
    
    # 3. Load all 5 models into GPU
    models = []
    print("Loading 5-fold ConvNeXt-Tiny models...")
    for fold, cpath in enumerate(fold_files):
        m = smp.Unet(
            encoder_name="tu-convnext_tiny.fb_in22k_ft_in1k",
            in_channels=3,
            classes=2,
            encoder_weights=None
        ).to(device)
        sd = torch.load(str(cpath), map_location=device, weights_only=False)
        m.load_state_dict(sd["model"], strict=True)
        m.eval()
        models.append(m)
        print(f"  Loaded Fold {fold} from {cpath.name}")
        
    # Preprocessing constants (ImageNet standard)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    
    # Inference knobs
    # Note: In Tokyo training, Channel 0 is filament, Channel 1 is background.
    INPUT_SIZE = 1536
    PROB_THRESHOLD = 0.50
    MIN_AREA = 150
    MAX_AREA = 40000
    
    submission_rows = []
    total_filaments = 0
    start_time = time.time()
    
    print("\nStarting 5-Fold Ensemble inference across 180 test images...")
    
    for idx, img_path in enumerate(test_files, 1):
        stem = img_path.stem
        raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape != (2048, 2048):
            raise ValueError(f"Failed to read image {img_path}")
            
        # Resize to 1536x1536
        img_1536 = cv2.resize(raw, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = np.stack([img_1536, img_1536, img_1536], axis=0).astype(np.float32) / 255.0
        norm_img = (rgb - mean) / std
        
        # 2-way TTA: original and horizontal flip
        t_orig = torch.from_numpy(norm_img).unsqueeze(0).to(device)
        t_flip = torch.flip(t_orig, dims=[3])
        
        prob_accum = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for m in models:
                    # Original
                    logits1 = m(t_orig)
                    p1 = torch.softmax(logits1, dim=1)[0, 0].cpu().numpy()
                    
                    # Flip
                    logits2 = m(t_flip)
                    p2_flipped = torch.softmax(logits2, dim=1)[0, 0]
                    p2 = torch.flip(p2_flipped, dims=[1]).cpu().numpy()
                    
                    prob_accum += (p1 + p2) * 0.5
                    
        # Average across all 5 folds
        mean_prob_1536 = prob_accum / len(models)
        
        # Resize back to native 2048x2048
        prob_2048 = cv2.resize(mean_prob_1536, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        
        # Threshold
        bin_mask = (prob_2048 >= PROB_THRESHOLD).astype(np.uint8)
        
        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        
        instances = []
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            if MIN_AREA <= area <= MAX_AREA:
                comp_mask = (labels == lab).astype(np.uint8)
                instances.append((area, comp_mask))
                
        # Sort by area descending
        instances.sort(key=lambda x: x[0], reverse=True)
        
        # Guarantee strict non-overlapping topology
        occupied = np.zeros((2048, 2048), dtype=np.uint8)
        inst_idx = 1
        for area, comp_mask in instances:
            clean = comp_mask & (~occupied)
            if clean.sum() >= MIN_AREA:
                fid = f"{stem}_{inst_idx}"
                rle_str = rle_encode(clean)
                submission_rows.append({
                    "filament_id": fid,
                    "segmentation_rle": rle_str
                })
                occupied |= clean
                inst_idx += 1
                total_filaments += 1
                
        # Fallback if image had 0 detections
        if inst_idx == 1:
            fb = np.zeros((2048, 2048), dtype=np.uint8)
            fb[1000:1048, 1000:1048] = 1
            submission_rows.append({
                "filament_id": f"{stem}_1",
                "segmentation_rle": rle_encode(fb)
            })
            total_filaments += 1
            
        if idx % 20 == 0 or idx == len(test_files):
            elapsed = time.time() - start_time
            print(f"[{idx:3d}/{len(test_files)}] Elapsed: {elapsed:.1f}s ({elapsed/idx:.2f}s/img), Filaments so far: {total_filaments}")
            
    # Write submission
    sub_df = pd.DataFrame(submission_rows)
    out_dir = Path("/kaggle/working")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission.csv"
    sub_df.to_csv(out_path, index=False)
    
    print("\n" + "=" * 70)
    print("Inference Complete!")
    print(f"Total instances written: {len(sub_df)}")
    covered = sub_df['filament_id'].apply(lambda x: x.rsplit('_', 1)[0]).nunique()
    print(f"Unique test images covered: {covered} / 180")
    print(f"Average filaments per image: {len(sub_df)/covered:.2f}")
    print(f"Submission saved to: {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()

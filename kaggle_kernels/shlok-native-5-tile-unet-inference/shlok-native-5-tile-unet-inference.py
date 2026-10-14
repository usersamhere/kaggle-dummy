"""
Native Resolution 5-Tile U-Net Inference Pipeline
=================================================
Model: UNetBase32 (Exp 008, 0.66 Public LB architecture)
Resolution: 2048x2048 Native Resolution via 5-Tile Blending with Hann Window
TTA: 4-Way Flip Augmentation (identity, hflip, vflip, hvflip)
Postprocessing: Connected Components + Area Calibration (>= 150 px)
Strict Zero Self-Overlap Guarantee
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

# --- Model Definition ---
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.b(x)

class UNetBase32(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList([
            DoubleConv(1, 32), DoubleConv(32, 64), DoubleConv(64, 128),
            DoubleConv(128, 256), DoubleConv(256, 512)
        ])
        self.bot = DoubleConv(512, 1024)
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, 2, stride=2), nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ConvTranspose2d(64, 32, 2, stride=2)
        ])
        self.dec = nn.ModuleList([
            DoubleConv(1024, 512), DoubleConv(512, 256), DoubleConv(256, 128),
            DoubleConv(128, 64), DoubleConv(64, 32)
        ])
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        skips = []
        for stage in self.enc:
            x = stage(x); skips.append(x); x = nn.functional.max_pool2d(x, 2)
        x = self.bot(x)
        for i in range(5):
            x = self.up[i](x)
            x = torch.cat([skips[4 - i], x], dim=1)
            x = self.dec[i](x)
        return self.head(x)

def rle_encode(mask: np.ndarray) -> str:
    fortran_mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    return rle["counts"].decode("utf-8")

def predict_tile_tta(model, tile_tensor, device):
    """4-way TTA on a single 1024x1024 tile."""
    # tile_tensor: [1, 1, 1024, 1024]
    views = [
        tile_tensor,                                    # identity
        torch.flip(tile_tensor, dims=[3]),             # hflip
        torch.flip(tile_tensor, dims=[2]),             # vflip
        torch.flip(tile_tensor, dims=[2, 3])           # hvflip
    ]
    batch = torch.cat(views, dim=0).to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(batch)
            probs = torch.sigmoid(logits)
    
    # Invert TTA
    p0 = probs[0:1]
    p1 = torch.flip(probs[1:2], dims=[3])
    p2 = torch.flip(probs[2:3], dims=[2])
    p3 = torch.flip(probs[3:4], dims=[2, 3])
    
    mean_prob = (p0 + p1 + p2 + p3) / 4.0
    return mean_prob.squeeze().cpu().numpy()

def main():
    print("=" * 60)
    print("Native Resolution 5-Tile U-Net Inference Starting...")
    print("=" * 60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Locate weights
    ckpt_paths = list(Path("/kaggle/input").rglob("filament_unet.pt"))
    if not ckpt_paths:
        raise FileNotFoundError("Could not find filament_unet.pt under /kaggle/input")
    ckpt_path = ckpt_paths[0]
    print(f"Found checkpoint: {ckpt_path}")
    
    # 2. Locate test images
    test_dirs = list(Path("/kaggle/input").rglob("test_images"))
    if not test_dirs:
        raise FileNotFoundError("Could not find test_images directory")
    test_dir = test_dirs[0]
    test_files = sorted(list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.jpg")))
    print(f"Found {len(test_files)} test images in {test_dir}")
    assert len(test_files) == 180, f"Expected 180 test images, found {len(test_files)}"
    
    # 3. Load model
    model = UNetBase32().to(device)
    sd = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print("Model loaded successfully with strict=True.")
    
    # 4. 5-Tile setup
    tile_size = 1024
    tiles = [
        (0, 1024, 0, 1024),         # Top-Left
        (0, 1024, 1024, 2048),       # Top-Right
        (1024, 2048, 0, 1024),       # Bottom-Left
        (1024, 2048, 1024, 2048),     # Bottom-Right
        (512, 1536, 512, 1536),      # Center overlap
    ]
    hann_1d = np.hanning(tile_size)
    hann_2d = np.outer(hann_1d, hann_1d).astype(np.float32)
    hann_weight = np.clip(hann_2d * 2.0, 0.1, 1.0)
    
    # Threshold and area settings
    PROB_THRESHOLD = 0.50
    MIN_AREA = 150
    MAX_AREA = 40000
    
    submission_rows = []
    start_time = time.time()
    
    for idx, img_path in enumerate(test_files, 1):
        stem = img_path.stem
        raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape != (2048, 2048):
            raise ValueError(f"Failed to read image {img_path}")
        
        # Standardize normalization (0 to 1)
        norm_img = raw.astype(np.float32) / 255.0
        
        prob_accum = np.zeros((2048, 2048), dtype=np.float32)
        weight_accum = np.zeros((2048, 2048), dtype=np.float32)
        
        for (y0, y1, x0, x1) in tiles:
            tile_np = norm_img[y0:y1, x0:x1]
            tile_t = torch.from_numpy(tile_np).unsqueeze(0).unsqueeze(0)  # [1, 1, 1024, 1024]
            tile_prob = predict_tile_tta(model, tile_t, device)
            
            w = hann_weight if (y0, x0) == (512, 512) else np.ones((tile_size, tile_size), dtype=np.float32)
            prob_accum[y0:y1, x0:x1] += tile_prob * w
            weight_accum[y0:y1, x0:x1] += w
            
        full_prob = prob_accum / weight_accum
        
        # Thresholding
        bin_mask = (full_prob >= PROB_THRESHOLD).astype(np.uint8)
        
        # Connected Components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        
        # Extract and sort instances by area descending
        instances = []
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            if MIN_AREA <= area <= MAX_AREA:
                comp_mask = (labels == lab).astype(np.uint8)
                instances.append((area, comp_mask))
                
        instances.sort(key=lambda x: x[0], reverse=True)
        
        # Guarantee zero self-overlap
        accum_mask = np.zeros((2048, 2048), dtype=np.uint8)
        inst_idx = 1
        for area, comp_mask in instances:
            clean_mask = comp_mask & (~accum_mask)
            if clean_mask.sum() >= MIN_AREA:
                fid = f"{stem}_{inst_idx}"
                rle_str = rle_encode(clean_mask)
                submission_rows.append({
                    "filament_id": fid,
                    "segmentation_rle": rle_str
                })
                accum_mask |= clean_mask
                inst_idx += 1
                
        # If no instance found, supply disk center fallback
        if inst_idx == 1:
            fb_mask = np.zeros((2048, 2048), dtype=np.uint8)
            fb_mask[1000:1048, 1000:1048] = 1
            submission_rows.append({
                "filament_id": f"{stem}_1",
                "segmentation_rle": rle_encode(fb_mask)
            })
            
        if idx % 20 == 0 or idx == len(test_files):
            elapsed = time.time() - start_time
            print(f"[{idx:3d}/{len(test_files)}] Completed in {elapsed:.1f}s ({elapsed/idx:.2f}s/img)")
            
    # Write submission
    sub_df = pd.DataFrame(submission_rows)
    out_dir = Path("/kaggle/working")
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = out_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)
    
    print("=" * 60)
    print("Inference Complete!")
    print(f"Total instances generated: {len(sub_df)}")
    print(f"Unique images covered: {sub_df['filament_id'].apply(lambda x: x.rsplit('_', 1)[0]).nunique()} / 180")
    print(f"Output saved to: {sub_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()

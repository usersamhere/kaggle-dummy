#!/usr/bin/env python3
"""Five-fold, grouped, train-only medoid U-Net + spine watershed pipeline.

The script never reads labels from /kaggle/input/test or any local test-label
file. Training and threshold selection use official training annotations only.
It emits OOF fold metrics, per-fold checkpoints, and one test submission CSV.
"""
import os, sys, json, time, subprocess, warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

try:
    import segmentation_models_pytorch as smp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch", "albumentations", "scikit-learn", "pyarrow"])
    import segmentation_models_pytorch as smp

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from pycocotools import mask as mask_utils

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

def enhance_solar_contrast(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to reveal fine
    chromospheric fibrils and solar filament structures.
    """
    if len(img.shape) == 3:
        gray = img[:, :, 0]
    else:
        gray = img

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    return enhanced

def get_train_transforms(image_size: int = 1024) -> A.Compose:
    """
    Data augmentations respecting solar rotational and reflectional physics (D4 group).
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.04,
                scale_limit=0.08,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.Normalize(mean=[0.35], std=[0.25]),
            ToTensorV2(),
        ],
        additional_targets={"spine": "mask", "weight": "mask"},
    )

def get_valid_transforms(image_size: int = 1024) -> A.Compose:
    """
    Deterministic preprocessing for validation and test inference.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.35], std=[0.25]),
            ToTensorV2(),
        ],
        additional_targets={"spine": "mask", "weight": "mask"},
    )

def split_into_tiles(
    img: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 256,
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """
    Splits a (H, W) or (H, W, C) image into overlapping tiles.
    Returns (tiles, coords) where coords is (y1, y2, x1, x2).
    """
    H, W = img.shape[:2]
    stride = tile_size - overlap
    tiles = []
    coords = []

    y_starts = list(range(0, H - tile_size + 1, stride))
    if y_starts[-1] + tile_size < H:
        y_starts.append(H - tile_size)

    x_starts = list(range(0, W - tile_size + 1, stride))
    if x_starts[-1] + tile_size < W:
        x_starts.append(W - tile_size)

    for y in y_starts:
        for x in x_starts:
            y1, y2 = y, y + tile_size
            x1, x2 = x, x + tile_size
            tile = img[y1:y2, x1:x2]
            tiles.append(tile)
            coords.append((y1, y2, x1, x2))

    return tiles, coords

def merge_tiles(
    tiles: list[np.ndarray],
    coords: list[tuple[int, int, int, int]],
    out_shape: tuple[int, int] = (2048, 2048),
    tile_size: int = 1024,
) -> np.ndarray:
    """
    Seamlessly merges predicted tile probability maps using 2D Cosine window weighting.
    """
    H, W = out_shape
    output = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    # Generate 2D Cosine window
    win1d = np.sin(np.linspace(0, np.pi, tile_size))
    window2d = np.outer(win1d, win1d).astype(np.float32)
    window2d = np.clip(window2d, 1e-4, 1.0)

    for tile, (y1, y2, x1, x2) in zip(tiles, coords):
        output[y1:y2, x1:x2] += tile * window2d
        weight_map[y1:y2, x1:x2] += window2d

    weight_map = np.maximum(weight_map, 1e-6)
    return output / weight_map

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import cv2

def paint_panoptic(
    candidates: list[tuple[float, np.ndarray]],
    min_area: int = 150,
    H: int = 2048,
    W: int = 2048,
) -> list[tuple[float, np.ndarray, int]]:
    """
    Score-ordered panoptic painting to resolve pixel overlap.
    
    Args:
        candidates: List of (confidence_score: float, binary_mask: np.ndarray uint8 [H, W])
        min_area: Minimum pixel area to retain an instance after subtraction.
        H, W: Image height and width (default 2048).

    Returns:
        List of (score, disjoint_binary_mask, area)
    """
    # Sort candidates by score descending
    candidates = sorted(candidates, key=lambda x: -x[0])
    claimed = np.zeros((H, W), dtype=np.uint8)
    kept = []

    for score, binary in candidates:
        if binary.dtype != np.uint8:
            binary = binary.astype(np.uint8)

        # Subtract pixels already claimed by higher-scoring detections
        remaining = binary & (~claimed)
        area = int(remaining.sum())

        if area < min_area:
            continue

        # Optional: check morphological elongation (filaments are elongated, sunspots are compact)
        # We can compute connected components on remaining in case subtraction split it
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(remaining, connectivity=8)
        valid_components = []
        for l in range(1, num_labels):
            comp_area = stats[l, cv2.CC_STAT_AREA]
            if comp_area >= min_area:
                comp_mask = (labels == l).astype(np.uint8)
                valid_components.append((comp_area, comp_mask))

        if not valid_components:
            continue

        # Re-merge valid components or keep largest
        final_mask = np.zeros((H, W), dtype=np.uint8)
        for _, comp_mask in valid_components:
            final_mask |= comp_mask

        final_area = int(final_mask.sum())
        if final_area < min_area:
            continue

        claimed |= final_mask
        kept.append((score, final_mask, final_area))

    return kept

def encode_mask_to_rle(mask: np.ndarray) -> str:
    """
    Encodes a 2D binary uint8 mask into a COCO RLE string.
    CRITICAL: Mask must be in Fortran-contiguous memory layout.
    """
    if not mask.flags["F_CONTIGUOUS"]:
        mask = np.asfortranarray(mask, dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    rle = mask_util.encode(mask)
    return rle["counts"].decode("utf-8")

def decode_rle_to_mask(rle_str: str, H: int = 2048, W: int = 2048) -> np.ndarray:
    """
    Decodes a COCO RLE string back to a (H, W) uint8 binary mask.
    """
    rle_dict = {"size": [H, W], "counts": rle_str.encode("utf-8") if isinstance(rle_str, str) else rle_str}
    return mask_util.decode(rle_dict).astype(np.uint8)

def masks_to_submission_rows(
    image_id: str,
    kept_instances: list[tuple[float, np.ndarray, int]],
) -> list[dict]:
    """
    Converts kept instances into formatted submission rows.
    """
    rows = []
    for idx, (score, mask, area) in enumerate(kept_instances, start=1):
        rle_str = encode_mask_to_rle(mask)
        rows.append({
            "filament_id": f"{image_id}_{idx}",
            "segmentation_rle": rle_str,
        })
    return rows

def check_no_overlap(csv_path_or_df, H: int = 2048, W: int = 2048) -> bool:
    """
    Strict validation to guarantee that no pixels are shared by two instances
    in any image. Raises ValueError if overlap is found.
    """
    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df.copy()

    if df.empty:
        print("[CHECK] DataFrame is empty.")
        return True

    df["stem"] = df["filament_id"].str.rsplit("_", n=1).str[0]
    total_checked = 0
    multi_instance_images = 0

    for stem, group in df.groupby("stem"):
        total_checked += 1
        if len(group) < 2:
            continue
        multi_instance_images += 1

        rle_dicts = [
            {"size": [H, W], "counts": r.encode("utf-8") if isinstance(r, str) else r}
            for r in group["segmentation_rle"]
        ]
        masks = mask_util.decode(rle_dicts)
        acc = masks.sum(axis=2)

        max_overlap = int(acc.max())
        if max_overlap > 1:
            overlap_count = int((acc > 1).sum())
            raise ValueError(
                f"[FAIL] Image stem '{stem}' has {overlap_count} overlapping pixels (max count {max_overlap})!"
            )

    print(f"[PASS] Zero pixel overlaps found across {total_checked} images ({multi_instance_images} multi-instance).")
    return True

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np
import cv2

class FilamentSpineUNet(nn.Module):
    """
    Dual-Head UNet predicting:
      - Channel 0: Filament binary segmentation logits
      - Channel 1: Spine / skeleton heatmap logits
      - Channel 2: Internal distance transform regression
    """
    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 1,
        classes: int = 3,
    ):
        super().__init__()
        weights = "imagenet" if pretrained else None
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        return self.model(x)

class FilamentCompositeLoss(nn.Module):
    """
    Multi-task loss balancing instance boundary accuracy and spine topological localization:
      L = L_bce(mask) + L_dice(mask) + 2.0 * L_focal(spine) + 1.0 * L_smooth_l1(dist)
    """
    def __init__(self, mask_pos_weight: float = 3.0, spine_pos_weight: float = 6.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([mask_pos_weight]))
        self.spine_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([spine_pos_weight]))
        self.smooth_l1 = nn.SmoothL1Loss()

    def dice_loss(self, logits, targets, smooth=1.0):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(-2, -1))
        union = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(self, preds, targets):
        # preds: [B, 3, H, W]
        # targets: [B, 3, H, W] (mask, spine, dist)
        mask_pred = preds[:, 0:1]
        spine_pred = preds[:, 1:2]
        dist_pred = preds[:, 2:3]

        mask_target = targets[:, 0:1]
        spine_target = targets[:, 1:2]
        dist_target = targets[:, 2:3]

        # 1. Mask loss
        if self.bce.pos_weight.device != mask_pred.device:
            self.bce.pos_weight = self.bce.pos_weight.to(mask_pred.device)
        l_mask_bce = self.bce(mask_pred, mask_target)
        l_mask_dice = self.dice_loss(mask_pred, mask_target)
        loss_mask = l_mask_bce + l_mask_dice

        # 2. Spine loss
        if self.spine_bce.pos_weight.device != spine_pred.device:
            self.spine_bce.pos_weight = self.spine_bce.pos_weight.to(spine_pred.device)
        loss_spine = self.spine_bce(spine_pred, spine_target) + self.dice_loss(spine_pred, spine_target)

        # 3. Distance regression loss (sigmoid on dist_pred)
        dist_probs = torch.sigmoid(dist_pred)
        loss_dist = self.smooth_l1(dist_probs, dist_target)

        total_loss = loss_mask + 2.0 * loss_spine + 1.0 * loss_dist
        return total_loss, {
            "loss_mask": loss_mask.item(),
            "loss_spine": loss_spine.item(),
            "loss_dist": loss_dist.item(),
            "total_loss": total_loss.item(),
        }

def spine_watershed_postprocess(
    mask_prob: np.ndarray,
    spine_prob: np.ndarray,
    dist_prob: np.ndarray = None,
    mask_threshold: float = 0.40,
    spine_threshold: float = 0.35,
    min_area: int = 150,
) -> list[tuple[float, np.ndarray, int]]:
    """
    Extracts individual filament instances using Spine-Seeded Watershed disambiguation,
    followed by score-ordered panoptic painting.
    """
    H, W = mask_prob.shape[:2]
    binary_mask = (mask_prob > mask_threshold).astype(np.uint8)

    if binary_mask.sum() < min_area:
        return []

    # 1. Extract spine markers inside the binary mask
    spine_markers = (spine_prob > spine_threshold) & (binary_mask > 0)

    # 2. Label seed components
    num_seeds, seed_labels = cv2.connectedComponents(spine_markers.astype(np.uint8), connectivity=8)

    candidates = []

    if num_seeds <= 1:
        # Fallback to connected components of binary mask directly if no spine seeds detected
        num_comps, comp_labels = cv2.connectedComponents(binary_mask, connectivity=8)
        for c in range(1, num_comps):
            c_mask = (comp_labels == c).astype(np.uint8)
            area = int(c_mask.sum())
            if area >= min_area:
                score = float(mask_prob[c_mask > 0].mean())
                candidates.append((score, c_mask))
    else:
        # 3. Spine-Seeded Watershed segmentation to separate close/parallel filaments
        # Invert distance or use 1 - mask_prob as elevation surface
        elevation = (255 * (1.0 - mask_prob)).astype(np.uint8)
        elevation = cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR)

        # Watershed requires markers with background = 0, boundaries = -1
        markers = seed_labels.astype(np.int32)
        # Background pixels outside binary mask set to 1, seed markers offset by 1
        markers[binary_mask == 0] = 0
        markers[seed_labels > 0] += 1
        markers[binary_mask == 0] = 1

        cv2.watershed(elevation, markers)

        # Extract segmented instances (markers > 1)
        unique_markers = np.unique(markers)
        for m in unique_markers:
            if m <= 1:  # 0 is unknown, 1 is background, -1 is boundary
                continue
            inst_mask = ((markers == m) & (binary_mask > 0)).astype(np.uint8)
            area = int(inst_mask.sum())
            if area >= min_area:
                # Score combines mask confidence and spine presence
                mean_conf = float(mask_prob[inst_mask > 0].mean())
                max_spine = float(spine_prob[inst_mask > 0].max()) if (inst_mask > 0).any() else 0.0
                score = 0.7 * mean_conf + 0.3 * max_spine
                candidates.append((score, inst_mask))

        # Check for any truly separated unseeded mask regions (e.g. isolated filaments without a strong spine)
        assigned = np.zeros((H, W), dtype=np.uint8)
        for _, c_mask in candidates:
            assigned |= c_mask
        # Only consider unassigned pixels that are not adjacent (erode 2px) to existing assigned instances
        unassigned = binary_mask & (~assigned)
        unassigned = cv2.erode(unassigned, np.ones((3, 3), np.uint8))
        if unassigned.sum() >= min_area:
            num_unassigned, unassigned_labels = cv2.connectedComponents(unassigned, connectivity=8)
            for c in range(1, num_unassigned):
                c_mask = (unassigned_labels == c).astype(np.uint8)
                c_mask = cv2.dilate(c_mask, np.ones((3, 3), np.uint8)) & binary_mask
                area = int(c_mask.sum())
                if area >= min_area:
                    score = float(mask_prob[c_mask > 0].mean())
                    candidates.append((score, c_mask))

    # 4. Enforce strict non-overlapping constraint via Score-Ordered Panoptic Painting
    kept_instances = paint_panoptic(candidates, min_area=min_area, H=H, W=W)
    return kept_instances

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pycocotools.mask as mask_util
from collections import defaultdict

class SolarFilamentDataset(Dataset):
    """
    Dataset loader for GONG H-alpha solar filament segmentation.
    Produces:
      - Enhanced solar image (1 channel)
      - Binary filament mask (1 channel)
      - Spine / skeleton heatmap (1 channel)
      - Normalized distance transform map (1 channel)

    Args:
        records_df: DataFrame with columns [file_name, stem, fold]
        data_dir: Path to MAGFiLO_1.0_Kaggle_2026
        coco_json_path: Path to COCO annotations JSON
        image_size: Target resize resolution
        is_train: Whether to apply training augmentations
        single_view: If True, use only the medoid annotator's masks (not consensus OR).
                     Requires selected_image_ids to be provided.
        selected_image_ids: dict mapping filename -> selected annotator image_id (for single_view mode)
    """
    def __init__(
        self,
        records_df,
        data_dir: str = "/home/dead/filament-segmentation/data/MAGFiLO_1.0_Kaggle_2026",
        coco_json_path: str = None,
        image_size: int = 1024,
        is_train: bool = True,
        single_view: bool = False,
        selected_image_ids: dict = None,
    ):
        self.records_df = records_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "train/train_images")
        self.image_size = image_size
        self.is_train = is_train
        self.single_view = single_view
        self.selected_image_ids = selected_image_ids or {}

        if is_train:
            self.transforms = get_train_transforms(image_size)
        else:
            self.transforms = get_valid_transforms(image_size)

        if coco_json_path is None:
            coco_json_path = os.path.join(data_dir, "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")

        # Load COCO annotations grouped by physical filename and annotator image_id
        with open(coco_json_path, "r") as f:
            coco = json.load(f)

        img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
        self.img_id_to_anns = defaultdict(list)
        self.file_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            self.img_id_to_anns[ann["image_id"]].append(ann)
            fname = img_id_to_file[ann["image_id"]]
            self.file_to_anns[fname].append(ann)

        # Build image_id lookup: file_name -> annotator image_id
        self.file_to_image_ids = defaultdict(list)
        for img in coco["images"]:
            self.file_to_image_ids[img["file_name"]].append(img["id"])

    def __len__(self):
        return len(self.records_df)

    def __getitem__(self, idx):
        row = self.records_df.iloc[idx]
        fname = row["file_name"]
        img_path = os.path.join(self.image_dir, fname)

        # 1. Read grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # 2. Contrast enhancement
        enhanced = enhance_solar_contrast(img)

        # 3. Render ground-truth mask and spine maps at native 2048x2048
        H, W = img.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        spine = np.zeros((H, W), dtype=np.uint8)

        if self.single_view:
            # Use only the medoid annotator's annotations (single-view GT)
            selected_img_id = self.selected_image_ids.get(fname)
            if selected_img_id is not None:
                anns = self.img_id_to_anns.get(selected_img_id, [])
            else:
                anns = []
        else:
            # Consensus: OR all annotators' masks together
            anns = self.file_to_anns.get(fname, [])

        for ann in anns:
            # Render polygon mask
            try:
                rles = mask_util.frPyObjects(ann["segmentation"], H, W)
                poly_mask = mask_util.decode(mask_util.merge(rles))
                mask |= poly_mask.astype(np.uint8)
            except Exception:
                pass

            # Render spine polyline
            spine_pts = ann.get("spine", [])
            if spine_pts and len(spine_pts) >= 4:
                coords = np.array(spine_pts, dtype=np.float32).reshape(-1, 2)
                pts = np.round(coords).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(spine, [pts], isClosed=False, color=1, thickness=3)

        # Distance transform inside the filaments
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_d = dist.max()
        if max_d > 0:
            dist = dist / max_d

        # 4. Apply augmentations
        augmented = self.transforms(
            image=enhanced,
            mask=mask,
            spine=spine,
            weight=dist,
        )

        image_t = augmented["image"].float() # [1, H, W]
        mask_t = augmented["mask"].unsqueeze(0).float() # [1, H, W]
        spine_t = augmented["spine"].unsqueeze(0).float() # [1, H, W]
        dist_t = augmented["weight"].unsqueeze(0).float() # [1, H, W]

        # Target tensor concatenates [mask, spine, distance]
        target_t = torch.cat([mask_t, spine_t, dist_t], dim=0) # [3, H, W]

        return {
            "image": image_t,
            "target": target_t,
            "file_name": fname,
            "stem": row["stem"],
        }


def semantic_mask_for_image_id(image_id, annotations_by_image, images_by_id, h=512, w=512):
    info=images_by_id[image_id]; oh,ow=int(info["height"]),int(info["width"])
    canvas=np.zeros((h,w),dtype=bool)
    for ann in annotations_by_image.get(image_id,[]):
        if ann.get("iscrowd",0): continue
        seg=ann["segmentation"]
        if isinstance(seg,dict): continue
        polygons=seg if isinstance(seg,list) else [seg]
        for poly in polygons:
            pts=np.asarray(poly,dtype=np.float32).reshape(-1,2)
            if len(pts)<3: continue
            pts[:,0]=np.clip(pts[:,0]*(w/ow),0,w-1)
            pts[:,1]=np.clip(pts[:,1]*(h/oh),0,h-1)
            m=np.zeros((h,w),np.uint8); cv2.fillPoly(m,[pts.astype(np.int32)],1)
            canvas |= m>0
    return canvas

def select_medoid_annotators(image_ids_by_file, annotations_by_image, images_by_id):
    selected={}
    for filename,ids in image_ids_by_file.items():
        if len(ids)==1: selected[filename]=ids[0]; continue
        masks=[semantic_mask_for_image_id(i,annotations_by_image,images_by_id) for i in ids]
        scores=[]
        for i,a in enumerate(masks):
            vals=[]
            for j,b in enumerate(masks):
                if i==j: continue
                inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum()
                vals.append(inter/union if union else 0.)
            scores.append(np.mean(vals) if vals else 0.)
        selected[filename]=ids[int(np.argmax(scores))]
    return selected


import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

def enhance_solar_contrast(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to reveal fine
    chromospheric fibrils and solar filament structures.
    """
    if len(img.shape) == 3:
        gray = img[:, :, 0]
    else:
        gray = img

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    return enhanced

def get_train_transforms(image_size: int = 1024) -> A.Compose:
    """
    Data augmentations respecting solar rotational and reflectional physics (D4 group).
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.04,
                scale_limit=0.08,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.Normalize(mean=[0.35], std=[0.25]),
            ToTensorV2(),
        ],
        additional_targets={"spine": "mask", "weight": "mask"},
    )

def get_valid_transforms(image_size: int = 1024) -> A.Compose:
    """
    Deterministic preprocessing for validation and test inference.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.35], std=[0.25]),
            ToTensorV2(),
        ],
        additional_targets={"spine": "mask", "weight": "mask"},
    )

def split_into_tiles(
    img: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 256,
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """
    Splits a (H, W) or (H, W, C) image into overlapping tiles.
    Returns (tiles, coords) where coords is (y1, y2, x1, x2).
    """
    H, W = img.shape[:2]
    stride = tile_size - overlap
    tiles = []
    coords = []

    y_starts = list(range(0, H - tile_size + 1, stride))
    if y_starts[-1] + tile_size < H:
        y_starts.append(H - tile_size)

    x_starts = list(range(0, W - tile_size + 1, stride))
    if x_starts[-1] + tile_size < W:
        x_starts.append(W - tile_size)

    for y in y_starts:
        for x in x_starts:
            y1, y2 = y, y + tile_size
            x1, x2 = x, x + tile_size
            tile = img[y1:y2, x1:x2]
            tiles.append(tile)
            coords.append((y1, y2, x1, x2))

    return tiles, coords

def merge_tiles(
    tiles: list[np.ndarray],
    coords: list[tuple[int, int, int, int]],
    out_shape: tuple[int, int] = (2048, 2048),
    tile_size: int = 1024,
) -> np.ndarray:
    """
    Seamlessly merges predicted tile probability maps using 2D Cosine window weighting.
    """
    H, W = out_shape
    output = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    # Generate 2D Cosine window
    win1d = np.sin(np.linspace(0, np.pi, tile_size))
    window2d = np.outer(win1d, win1d).astype(np.float32)
    window2d = np.clip(window2d, 1e-4, 1.0)

    for tile, (y1, y2, x1, x2) in zip(tiles, coords):
        output[y1:y2, x1:x2] += tile * window2d
        weight_map[y1:y2, x1:x2] += window2d

    weight_map = np.maximum(weight_map, 1e-6)
    return output / weight_map

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import cv2

def paint_panoptic(
    candidates: list[tuple[float, np.ndarray]],
    min_area: int = 150,
    H: int = 2048,
    W: int = 2048,
) -> list[tuple[float, np.ndarray, int]]:
    """
    Score-ordered panoptic painting to resolve pixel overlap.
    
    Args:
        candidates: List of (confidence_score: float, binary_mask: np.ndarray uint8 [H, W])
        min_area: Minimum pixel area to retain an instance after subtraction.
        H, W: Image height and width (default 2048).

    Returns:
        List of (score, disjoint_binary_mask, area)
    """
    # Sort candidates by score descending
    candidates = sorted(candidates, key=lambda x: -x[0])
    claimed = np.zeros((H, W), dtype=np.uint8)
    kept = []

    for score, binary in candidates:
        if binary.dtype != np.uint8:
            binary = binary.astype(np.uint8)

        # Subtract pixels already claimed by higher-scoring detections
        remaining = binary & (~claimed)
        area = int(remaining.sum())

        if area < min_area:
            continue

        # Optional: check morphological elongation (filaments are elongated, sunspots are compact)
        # We can compute connected components on remaining in case subtraction split it
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(remaining, connectivity=8)
        valid_components = []
        for l in range(1, num_labels):
            comp_area = stats[l, cv2.CC_STAT_AREA]
            if comp_area >= min_area:
                comp_mask = (labels == l).astype(np.uint8)
                valid_components.append((comp_area, comp_mask))

        if not valid_components:
            continue

        # Re-merge valid components or keep largest
        final_mask = np.zeros((H, W), dtype=np.uint8)
        for _, comp_mask in valid_components:
            final_mask |= comp_mask

        final_area = int(final_mask.sum())
        if final_area < min_area:
            continue

        claimed |= final_mask
        kept.append((score, final_mask, final_area))

    return kept

def encode_mask_to_rle(mask: np.ndarray) -> str:
    """
    Encodes a 2D binary uint8 mask into a COCO RLE string.
    CRITICAL: Mask must be in Fortran-contiguous memory layout.
    """
    if not mask.flags["F_CONTIGUOUS"]:
        mask = np.asfortranarray(mask, dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    rle = mask_util.encode(mask)
    return rle["counts"].decode("utf-8")

def decode_rle_to_mask(rle_str: str, H: int = 2048, W: int = 2048) -> np.ndarray:
    """
    Decodes a COCO RLE string back to a (H, W) uint8 binary mask.
    """
    rle_dict = {"size": [H, W], "counts": rle_str.encode("utf-8") if isinstance(rle_str, str) else rle_str}
    return mask_util.decode(rle_dict).astype(np.uint8)

def masks_to_submission_rows(
    image_id: str,
    kept_instances: list[tuple[float, np.ndarray, int]],
) -> list[dict]:
    """
    Converts kept instances into formatted submission rows.
    """
    rows = []
    for idx, (score, mask, area) in enumerate(kept_instances, start=1):
        rle_str = encode_mask_to_rle(mask)
        rows.append({
            "filament_id": f"{image_id}_{idx}",
            "segmentation_rle": rle_str,
        })
    return rows

def check_no_overlap(csv_path_or_df, H: int = 2048, W: int = 2048) -> bool:
    """
    Strict validation to guarantee that no pixels are shared by two instances
    in any image. Raises ValueError if overlap is found.
    """
    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df.copy()

    if df.empty:
        print("[CHECK] DataFrame is empty.")
        return True

    df["stem"] = df["filament_id"].str.rsplit("_", n=1).str[0]
    total_checked = 0
    multi_instance_images = 0

    for stem, group in df.groupby("stem"):
        total_checked += 1
        if len(group) < 2:
            continue
        multi_instance_images += 1

        rle_dicts = [
            {"size": [H, W], "counts": r.encode("utf-8") if isinstance(r, str) else r}
            for r in group["segmentation_rle"]
        ]
        masks = mask_util.decode(rle_dicts)
        acc = masks.sum(axis=2)

        max_overlap = int(acc.max())
        if max_overlap > 1:
            overlap_count = int((acc > 1).sum())
            raise ValueError(
                f"[FAIL] Image stem '{stem}' has {overlap_count} overlapping pixels (max count {max_overlap})!"
            )

    print(f"[PASS] Zero pixel overlaps found across {total_checked} images ({multi_instance_images} multi-instance).")
    return True

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np
import cv2

class FilamentSpineUNet(nn.Module):
    """
    Dual-Head UNet predicting:
      - Channel 0: Filament binary segmentation logits
      - Channel 1: Spine / skeleton heatmap logits
      - Channel 2: Internal distance transform regression
    """
    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 1,
        classes: int = 3,
    ):
        super().__init__()
        weights = "imagenet" if pretrained else None
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        return self.model(x)

class FilamentCompositeLoss(nn.Module):
    """
    Multi-task loss balancing instance boundary accuracy and spine topological localization:
      L = L_bce(mask) + L_dice(mask) + 2.0 * L_focal(spine) + 1.0 * L_smooth_l1(dist)
    """
    def __init__(self, mask_pos_weight: float = 3.0, spine_pos_weight: float = 6.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([mask_pos_weight]))
        self.spine_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([spine_pos_weight]))
        self.smooth_l1 = nn.SmoothL1Loss()

    def dice_loss(self, logits, targets, smooth=1.0):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(-2, -1))
        union = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(self, preds, targets):
        # preds: [B, 3, H, W]
        # targets: [B, 3, H, W] (mask, spine, dist)
        mask_pred = preds[:, 0:1]
        spine_pred = preds[:, 1:2]
        dist_pred = preds[:, 2:3]

        mask_target = targets[:, 0:1]
        spine_target = targets[:, 1:2]
        dist_target = targets[:, 2:3]

        # 1. Mask loss
        if self.bce.pos_weight.device != mask_pred.device:
            self.bce.pos_weight = self.bce.pos_weight.to(mask_pred.device)
        l_mask_bce = self.bce(mask_pred, mask_target)
        l_mask_dice = self.dice_loss(mask_pred, mask_target)
        loss_mask = l_mask_bce + l_mask_dice

        # 2. Spine loss
        if self.spine_bce.pos_weight.device != spine_pred.device:
            self.spine_bce.pos_weight = self.spine_bce.pos_weight.to(spine_pred.device)
        loss_spine = self.spine_bce(spine_pred, spine_target) + self.dice_loss(spine_pred, spine_target)

        # 3. Distance regression loss (sigmoid on dist_pred)
        dist_probs = torch.sigmoid(dist_pred)
        loss_dist = self.smooth_l1(dist_probs, dist_target)

        total_loss = loss_mask + 2.0 * loss_spine + 1.0 * loss_dist
        return total_loss, {
            "loss_mask": loss_mask.item(),
            "loss_spine": loss_spine.item(),
            "loss_dist": loss_dist.item(),
            "total_loss": total_loss.item(),
        }

def spine_watershed_postprocess(
    mask_prob: np.ndarray,
    spine_prob: np.ndarray,
    dist_prob: np.ndarray = None,
    mask_threshold: float = 0.40,
    spine_threshold: float = 0.35,
    min_area: int = 150,
) -> list[tuple[float, np.ndarray, int]]:
    """
    Extracts individual filament instances using Spine-Seeded Watershed disambiguation,
    followed by score-ordered panoptic painting.
    """
    H, W = mask_prob.shape[:2]
    binary_mask = (mask_prob > mask_threshold).astype(np.uint8)

    if binary_mask.sum() < min_area:
        return []

    # 1. Extract spine markers inside the binary mask
    spine_markers = (spine_prob > spine_threshold) & (binary_mask > 0)

    # 2. Label seed components
    num_seeds, seed_labels = cv2.connectedComponents(spine_markers.astype(np.uint8), connectivity=8)

    candidates = []

    if num_seeds <= 1:
        # Fallback to connected components of binary mask directly if no spine seeds detected
        num_comps, comp_labels = cv2.connectedComponents(binary_mask, connectivity=8)
        for c in range(1, num_comps):
            c_mask = (comp_labels == c).astype(np.uint8)
            area = int(c_mask.sum())
            if area >= min_area:
                score = float(mask_prob[c_mask > 0].mean())
                candidates.append((score, c_mask))
    else:
        # 3. Spine-Seeded Watershed segmentation to separate close/parallel filaments
        # Invert distance or use 1 - mask_prob as elevation surface
        elevation = (255 * (1.0 - mask_prob)).astype(np.uint8)
        elevation = cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR)

        # Watershed requires markers with background = 0, boundaries = -1
        markers = seed_labels.astype(np.int32)
        # Background pixels outside binary mask set to 1, seed markers offset by 1
        markers[binary_mask == 0] = 0
        markers[seed_labels > 0] += 1
        markers[binary_mask == 0] = 1

        cv2.watershed(elevation, markers)

        # Extract segmented instances (markers > 1)
        unique_markers = np.unique(markers)
        for m in unique_markers:
            if m <= 1:  # 0 is unknown, 1 is background, -1 is boundary
                continue
            inst_mask = ((markers == m) & (binary_mask > 0)).astype(np.uint8)
            area = int(inst_mask.sum())
            if area >= min_area:
                # Score combines mask confidence and spine presence
                mean_conf = float(mask_prob[inst_mask > 0].mean())
                max_spine = float(spine_prob[inst_mask > 0].max()) if (inst_mask > 0).any() else 0.0
                score = 0.7 * mean_conf + 0.3 * max_spine
                candidates.append((score, inst_mask))

        # Check for any truly separated unseeded mask regions (e.g. isolated filaments without a strong spine)
        assigned = np.zeros((H, W), dtype=np.uint8)
        for _, c_mask in candidates:
            assigned |= c_mask
        # Only consider unassigned pixels that are not adjacent (erode 2px) to existing assigned instances
        unassigned = binary_mask & (~assigned)
        unassigned = cv2.erode(unassigned, np.ones((3, 3), np.uint8))
        if unassigned.sum() >= min_area:
            num_unassigned, unassigned_labels = cv2.connectedComponents(unassigned, connectivity=8)
            for c in range(1, num_unassigned):
                c_mask = (unassigned_labels == c).astype(np.uint8)
                c_mask = cv2.dilate(c_mask, np.ones((3, 3), np.uint8)) & binary_mask
                area = int(c_mask.sum())
                if area >= min_area:
                    score = float(mask_prob[c_mask > 0].mean())
                    candidates.append((score, c_mask))

    # 4. Enforce strict non-overlapping constraint via Score-Ordered Panoptic Painting
    kept_instances = paint_panoptic(candidates, min_area=min_area, H=H, W=W)
    return kept_instances

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pycocotools.mask as mask_util
from collections import defaultdict

class SolarFilamentDataset(Dataset):
    """
    Dataset loader for GONG H-alpha solar filament segmentation.
    Produces:
      - Enhanced solar image (1 channel)
      - Binary filament mask (1 channel)
      - Spine / skeleton heatmap (1 channel)
      - Normalized distance transform map (1 channel)

    Args:
        records_df: DataFrame with columns [file_name, stem, fold]
        data_dir: Path to MAGFiLO_1.0_Kaggle_2026
        coco_json_path: Path to COCO annotations JSON
        image_size: Target resize resolution
        is_train: Whether to apply training augmentations
        single_view: If True, use only the medoid annotator's masks (not consensus OR).
                     Requires selected_image_ids to be provided.
        selected_image_ids: dict mapping filename -> selected annotator image_id (for single_view mode)
    """
    def __init__(
        self,
        records_df,
        data_dir: str = "/home/dead/filament-segmentation/data/MAGFiLO_1.0_Kaggle_2026",
        coco_json_path: str = None,
        image_size: int = 1024,
        is_train: bool = True,
        single_view: bool = False,
        selected_image_ids: dict = None,
    ):
        self.records_df = records_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "train/train_images")
        self.image_size = image_size
        self.is_train = is_train
        self.single_view = single_view
        self.selected_image_ids = selected_image_ids or {}

        if is_train:
            self.transforms = get_train_transforms(image_size)
        else:
            self.transforms = get_valid_transforms(image_size)

        if coco_json_path is None:
            coco_json_path = os.path.join(data_dir, "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")

        # Load COCO annotations grouped by physical filename and annotator image_id
        with open(coco_json_path, "r") as f:
            coco = json.load(f)

        img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
        self.img_id_to_anns = defaultdict(list)
        self.file_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            self.img_id_to_anns[ann["image_id"]].append(ann)
            fname = img_id_to_file[ann["image_id"]]
            self.file_to_anns[fname].append(ann)

        # Build image_id lookup: file_name -> annotator image_id
        self.file_to_image_ids = defaultdict(list)
        for img in coco["images"]:
            self.file_to_image_ids[img["file_name"]].append(img["id"])

    def __len__(self):
        return len(self.records_df)

    def __getitem__(self, idx):
        row = self.records_df.iloc[idx]
        fname = row["file_name"]
        img_path = os.path.join(self.image_dir, fname)

        # 1. Read grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # 2. Contrast enhancement
        enhanced = enhance_solar_contrast(img)

        # 3. Render ground-truth mask and spine maps at native 2048x2048
        H, W = img.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        spine = np.zeros((H, W), dtype=np.uint8)

        if self.single_view:
            # Use only the medoid annotator's annotations (single-view GT)
            selected_img_id = self.selected_image_ids.get(fname)
            if selected_img_id is not None:
                anns = self.img_id_to_anns.get(selected_img_id, [])
            else:
                anns = []
        else:
            # Consensus: OR all annotators' masks together
            anns = self.file_to_anns.get(fname, [])

        for ann in anns:
            # Render polygon mask
            try:
                rles = mask_util.frPyObjects(ann["segmentation"], H, W)
                poly_mask = mask_util.decode(mask_util.merge(rles))
                mask |= poly_mask.astype(np.uint8)
            except Exception:
                pass

            # Render spine polyline
            spine_pts = ann.get("spine", [])
            if spine_pts and len(spine_pts) >= 4:
                coords = np.array(spine_pts, dtype=np.float32).reshape(-1, 2)
                pts = np.round(coords).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(spine, [pts], isClosed=False, color=1, thickness=3)

        # Distance transform inside the filaments
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_d = dist.max()
        if max_d > 0:
            dist = dist / max_d

        # 4. Apply augmentations
        augmented = self.transforms(
            image=enhanced,
            mask=mask,
            spine=spine,
            weight=dist,
        )

        image_t = augmented["image"].float() # [1, H, W]
        mask_t = augmented["mask"].unsqueeze(0).float() # [1, H, W]
        spine_t = augmented["spine"].unsqueeze(0).float() # [1, H, W]
        dist_t = augmented["weight"].unsqueeze(0).float() # [1, H, W]

        # Target tensor concatenates [mask, spine, distance]
        target_t = torch.cat([mask_t, spine_t, dist_t], dim=0) # [3, H, W]

        return {
            "image": image_t,
            "target": target_t,
            "file_name": fname,
            "stem": row["stem"],
        }


def semantic_mask_for_image_id(image_id, annotations_by_image, images_by_id, h=512, w=512):
    info=images_by_id[image_id]; oh,ow=int(info["height"]),int(info["width"])
    canvas=np.zeros((h,w),dtype=bool)
    for ann in annotations_by_image.get(image_id,[]):
        if ann.get("iscrowd",0): continue
        seg=ann["segmentation"]
        if isinstance(seg,dict): continue
        polygons=seg if isinstance(seg,list) else [seg]
        for poly in polygons:
            pts=np.asarray(poly,dtype=np.float32).reshape(-1,2)
            if len(pts)<3: continue
            pts[:,0]=np.clip(pts[:,0]*(w/ow),0,w-1)
            pts[:,1]=np.clip(pts[:,1]*(h/oh),0,h-1)
            m=np.zeros((h,w),np.uint8); cv2.fillPoly(m,[pts.astype(np.int32)],1)
            canvas |= m>0
    return canvas

def select_medoid_annotators(image_ids_by_file, annotations_by_image, images_by_id):
    selected={}
    for filename,ids in image_ids_by_file.items():
        if len(ids)==1: selected[filename]=ids[0]; continue
        masks=[semantic_mask_for_image_id(i,annotations_by_image,images_by_id) for i in ids]
        scores=[]
        for i,a in enumerate(masks):
            vals=[]
            for j,b in enumerate(masks):
                if i==j: continue
                inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum()
                vals.append(inter/union if union else 0.)
            scores.append(np.mean(vals) if vals else 0.)
        selected[filename]=ids[int(np.argmax(scores))]
    return selected


SEED = 42
FOLDS = 5
EPOCHS = 6
IMAGE_SIZE = 1536
BATCH_SIZE = 1
ACCUM_STEPS = 2
NUM_WORKERS = 4
MIN_AREAS = (80, 150, 250)
MASK_THRESHOLDS = (0.40, 0.46, 0.52, 0.58)
SPINE_THRESHOLDS = (0.30, 0.40, 0.50)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/kaggle/working/v43")
OUT.mkdir(parents=True, exist_ok=True)

def locate_data():
    for p in Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"):
        base = p.parent.parent
        if (base / "train" / "train_images").is_dir():
            return base, p
    raise FileNotFoundError("Competition training annotations/images not found under /kaggle/input")

def polygon_mask(ann, h, w):
    seg = ann["segmentation"]
    if isinstance(seg, dict):
        return mask_utils.decode(seg).astype(np.uint8)
    rles = mask_utils.frPyObjects(seg, h, w)
    return mask_utils.decode(mask_utils.merge(rles)).astype(np.uint8)

def make_folds(coco):
    images = {str(i["id"]): i for i in coco["images"]}
    anns_by_image = defaultdict(list)
    file_to_ids = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[str(a["image_id"])].append(a)
    for iid, info in images.items():
        file_to_ids[info["file_name"]].append(iid)
    for name in file_to_ids:
        file_to_ids[name].sort()
    selected = select_medoid_annotators(file_to_ids, anns_by_image, images)
    rows=[]
    for name, ids in sorted(file_to_ids.items()):
        nper=[len(anns_by_image.get(i, [])) for i in ids]
        rows.append({"file_name":name,"stem":Path(name).stem,"num_annotators":len(ids),
                     "mean_instances":float(np.mean(nper)),"max_instances":max(nper,default=0)})
    df=pd.DataFrame(rows)
    df["strata_bin"]=(pd.qcut(df.mean_instances,q=4,labels=False,duplicates="drop").astype(str)
                      + "_" + df.num_annotators.astype(str))
    df["fold"]=-1
    splitter=StratifiedGroupKFold(n_splits=FOLDS,shuffle=True,random_state=SEED)
    for fold,(_,vi) in enumerate(splitter.split(df,df.strata_bin,groups=df.stem)):
        df.loc[vi,"fold"]=fold
    return df, images, anns_by_image, file_to_ids, selected

def gt_views_by_stem(val_names, file_to_ids, images, anns_by_image):
    result={}
    scale=IMAGE_SIZE/2048.0
    for name in val_names:
        stem=Path(name).stem
        views=[]
        for iid in file_to_ids[name]:
            im=images[iid]; h=int(im["height"]); w=int(im["width"])
            view=[]
            for a in anns_by_image.get(iid,[]):
                if a.get("iscrowd",0): continue
                m=polygon_mask(a,h,w)
                m=cv2.resize(m,(IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_NEAREST)
                if int(m.sum())>=20:
                    view.append(mask_utils.encode(np.asfortranarray(m)))
            views.append(view)
        result[stem]=views
    return result

def greedy_match_ious(pred, gt):
    if not pred or not gt: return []
    ious=mask_utils.iou(pred,gt,[0]*len(gt))
    edges=[(float(ious[i,j]),i,j) for i in range(len(pred)) for j in range(len(gt)) if ious[i,j] > 0.5]
    edges.sort(reverse=True)
    usedp=set(); usedg=set(); matched=[]
    for v,i,j in edges:
        if i not in usedp and j not in usedg:
            usedp.add(i); usedg.add(j); matched.append(v)
    return matched

def score_predictions(pred_by_stem, gt_by_stem):
    tp=fp=fn=0; sum_iou=0.0
    for stem,views in gt_by_stem.items():
        pred=pred_by_stem.get(stem,[])
        for gt in views:
            matched=greedy_match_ious(pred,gt)
            t=len(matched); tp+=t; fp+=len(pred)-t; fn+=len(gt)-t; sum_iou+=sum(matched)
    den=tp+0.5*fp+0.5*fn
    return {"pq":sum_iou/den if den else 0.0,"sq":sum_iou/tp if tp else 0.0,
            "rq":tp/den if den else 0.0,"tp":tp,"fp":fp,"fn":fn,"sum_iou":sum_iou}

def predict_components(prob3, params):
    mask_p,spine_p,dist_p=prob3
    raw=spine_watershed_postprocess(mask_p,spine_p,dist_p,
        mask_threshold=params[0],spine_threshold=params[1],min_area=params[2])
    painted=paint_panoptic([(s,m.astype(np.uint8)) for s,m,_ in raw],min_area=params[2],H=IMAGE_SIZE,W=IMAGE_SIZE)
    return [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for _,m,_ in painted]

@torch.no_grad()
def infer_fold(model, loader, names):
    model.eval(); probs={}
    for batch in loader:
        x=batch["image"].to(DEVICE,non_blocking=True)
        # D4 test-time averaging; average logits after inverse transforms.
        outs=[]
        for k in range(4):
            for flip in (False,True):
                xx=torch.rot90(x,k,(-2,-1))
                if flip: xx=torch.flip(xx,(-1,))
                z=model(xx)
                if flip: z=torch.flip(z,(-1,))
                z=torch.rot90(z,-k,(-2,-1))
                outs.append(torch.sigmoid(z.float()))
        p=torch.stack(outs).mean(0)[0].cpu().numpy().astype(np.float16)
        probs[batch["stem"][0]]=p
    return probs

def train_one_fold(fold, frame, base, selected, all_views):
    train_df=frame[frame.fold!=fold].reset_index(drop=True)
    val_df=frame[frame.fold==fold].reset_index(drop=True)
    data_dir=str(base)
    train_ds=SolarFilamentDataset(train_df,data_dir=data_dir,image_size=IMAGE_SIZE,is_train=True,
                                  single_view=True,selected_image_ids=selected)
    val_ds=SolarFilamentDataset(val_df,data_dir=data_dir,image_size=IMAGE_SIZE,is_train=False,
                                single_view=True,selected_image_ids=selected)
    train_dl=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,
                        pin_memory=True,persistent_workers=True)
    val_dl=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=NUM_WORKERS,
                      pin_memory=True,persistent_workers=True)
    model=FilamentSpineUNet(backbone="resnet34",pretrained=True,in_channels=1,classes=3).to(DEVICE)
    criterion=FilamentCompositeLoss().to(DEVICE)
    optim=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=EPOCHS,eta_min=1e-6)
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE.type=="cuda"))
    print(f"FOLD {fold}: train={len(train_ds)} val={len(val_ds)} device={DEVICE}",flush=True)
    for ep in range(EPOCHS):
        model.train(); loss_sum=0.0; optim.zero_grad(set_to_none=True); t0=time.time()
        for step,b in enumerate(train_dl):
            x=b["image"].to(DEVICE,non_blocking=True); y=b["target"].to(DEVICE,non_blocking=True)
            with torch.autocast(device_type=DEVICE.type,dtype=torch.float16,enabled=(DEVICE.type=="cuda")):
                pred=model(x); loss,_=criterion(pred,y); loss=loss/ACCUM_STEPS
            scaler.scale(loss).backward()
            if (step+1)%ACCUM_STEPS==0 or step+1==len(train_dl):
                scaler.unscale_(optim); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
                scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
            loss_sum+=float(loss.detach())*ACCUM_STEPS
        sched.step()
        print(f"fold={fold} epoch={ep+1}/{EPOCHS} loss={loss_sum/len(train_dl):.5f} sec={time.time()-t0:.0f}",flush=True)
    ckpt=OUT/f"fold{fold}.pt"
    torch.save({"model_state":model.state_dict(),"fold":fold,"image_size":IMAGE_SIZE},ckpt)
    val_names=val_df.file_name.tolist()
    probs=infer_fold(model,val_dl,val_names)
    gt=gt_views_by_stem(val_names,all_views["file_to_ids"],all_views["images"],all_views["anns"])
    sweep=[]
    for mt in MASK_THRESHOLDS:
      for st in SPINE_THRESHOLDS:
       for ma in MIN_AREAS:
        pars=(mt,st,ma); pred={s:predict_components(p,pars) for s,p in probs.items()}
        met=score_predictions(pred,gt); sweep.append((met["pq"],pars,met))
    sweep.sort(key=lambda x:x[0],reverse=True)
    print(f"fold={fold} best_oof={sweep[0][0]:.5f} params={sweep[0][1]} metrics={sweep[0][2]}",flush=True)
    del model,train_dl,val_dl,train_ds,val_ds,probs,gt
    torch.cuda.empty_cache();
    return sweep[0][1],sweep[0][2]

def write_test_submission(frame,base,selected,fold_params):
    test_dir=base/"test"/"test_images"
    test_files=sorted(test_dir.glob("*.jp*g"))
    rows=[]
    models=[]
    for fold in range(FOLDS):
        model=FilamentSpineUNet(backbone="resnet34",pretrained=False,in_channels=1,classes=3).to(DEVICE)
        state=torch.load(OUT/f"fold{fold}.pt",map_location="cpu",weights_only=False)["model_state"]
        model.load_state_dict(state); model.eval(); models.append(model)
    from solar_transforms import enhance_solar_contrast, get_valid_transforms
    transform=get_valid_transforms(IMAGE_SIZE)
    params=tuple(np.median(np.array(fold_params,dtype=float),axis=0).tolist())
    params=(float(params[0]),float(params[1]),int(round(params[2])))
    print(f"TEST inference files={len(test_files)} params={params}",flush=True)
    for ix,path in enumerate(test_files,1):
        img=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        x=transform(image=enhance_solar_contrast(img))["image"].unsqueeze(0).to(DEVICE)
        outs=[]
        with torch.no_grad():
            for model in models:
                variants=[]
                for k in range(4):
                    for flip in (False,True):
                        xx=torch.rot90(x,k,(-2,-1))
                        if flip: xx=torch.flip(xx,(-1,))
                        z=model(xx)
                        if flip: z=torch.flip(z,(-1,))
                        z=torch.rot90(z,-k,(-2,-1))
                        variants.append(torch.sigmoid(z.float()))
                outs.append(torch.stack(variants).mean(0)[0])
        p=torch.stack(outs).mean(0).cpu().numpy()
        # Nearest-neighbor resize preserves instance labels at native resolution.
        candidates=predict_components(p,params)
        native=[]
        for r in candidates:
            m=mask_utils.decode(r)
            m=cv2.resize(m,(2048,2048),interpolation=cv2.INTER_NEAREST)
            if m.sum()>=100:
                native.append(m.astype(np.uint8))
        # Greedy allocation is repeated at native size to guarantee disjoint RLEs.
        claimed=np.zeros((2048,2048),np.uint8); inst=0
        for m in sorted(native,key=lambda a:int(a.sum()),reverse=True):
            clean=m & (claimed==0)
            if int(clean.sum())<100: continue
            claimed |= clean; inst+=1
            rows.append({"filament_id":f"{path.stem}_{inst}","segmentation_rle":encode_mask_to_rle(clean)})
        if ix%20==0 or ix==len(test_files): print(f"test {ix}/{len(test_files)} rows={len(rows)}",flush=True)
    # Guarantee a parseable row for every image; empty masks are valid but count as no detections.
    covered={r["filament_id"].rsplit("_",1)[0] for r in rows}
    for path in test_files:
        if path.stem not in covered:
            rows.append({"filament_id":f"{path.stem}_1","segmentation_rle":encode_mask_to_rle(np.zeros((2048,2048),np.uint8))})
    pd.DataFrame(rows,columns=["filament_id","segmentation_rle"]).to_csv(OUT/"submission_v43.csv",index=False)
    with open(OUT/"oof_summary.json","w") as f:
        json.dump({"fold_params":fold_params,"aggregate_oof":None},f,indent=2)
    print(f"WROTE {OUT/'submission_v43.csv'} rows={len(rows)}",flush=True)

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.benchmark=True
    base,ann_path=locate_data()
    print(f"training data={base} annotations={ann_path} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",flush=True)
    coco=json.load(open(ann_path))
    frame,images,anns,file_to_ids,selected=make_folds(coco)
    print(f"physical train images={len(frame)} selected medoid views={len(selected)}",flush=True)
    view={"images":images,"anns":anns,"file_to_ids":file_to_ids}
    params=[]; metrics=[]
    for fold in range(FOLDS):
        p,m=train_one_fold(fold,frame,base,selected,view); params.append(p); metrics.append(m)
    pq_num=sum(m["sum_iou"] for m in metrics)
    den=sum(m["tp"]+0.5*m["fp"]+0.5*m["fn"] for m in metrics)
    summary={"fold_metrics":metrics,"fold_params":params,"aggregate_oof_pq":pq_num/den if den else 0.0,
             "note":"OOF score is on grouped official training folds; no test labels used."}
    json.dump(summary,open(OUT/"oof_summary.json","w"),indent=2)
    print("AGGREGATE OOF",json.dumps(summary),flush=True)
    write_test_submission(frame,base,selected,params)

if __name__=="__main__": main()

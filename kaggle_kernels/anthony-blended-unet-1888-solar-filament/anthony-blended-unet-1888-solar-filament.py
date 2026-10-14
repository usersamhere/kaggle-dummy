"""
Solar Filament Segmentation Challenge 2026 -- single-pass U-Net baseline.

Binary segmentation (filament vs background) at NATIVE resolution:
  - COCO polygons (all 4 categories) merged into ONE foreground mask; the several
    annotator passes that share a file_name are unioned.
  - CENTRED CROP. Only the central 1888x1888 of each 2048^2 frame is modelled: the
    corners are black, the disk has radius ~1005px, and no annotated filament reaches
    past 903px from the centre, so the crop keeps margin while dividing evenly.
  - THE WHOLE CROP, TRAINING AND INFERENCE ALIKE. One 1888x1888 sample per frame, one
    forward pass per prediction. 1888 = 59*32, so it enters and leaves the network with
    no padding or resampling, and nothing is ever tiled: no seams to reconcile, no
    crossfade, and no gap between what the model is trained on and what it is asked.
    The price is a batch of ONE, which is why the optimizer accumulates -- see ACCUM.
  - Cache: 2048^2 uint8 memmaps for images and masks, decoded once. Every sample is
    the same centred crop of one frame, so nothing has to be sampled or balanced.
  - U-Net: any timm encoder (ImageNet, in_chans=1) + a decoder driven by the
    encoder's stage reductions.
  - Loss: BCE(pos_weight) + soft Dice over the whole tile. bf16 autocast. Two
    metric-shaped variants (clDice + instance-balanced pixel weights, and a
    per-instance soft Dice) both submitted worse and have been removed: the metric
    being instance-level does not mean a loss shaped like it trains better.
  - Validation reports what the submission is judged on: the DISTRIBUTIONS of
    per-instance Dice and IoU, per-image Dice, and the one-to-many / many-to-one
    relation counts between ground truth and prediction. It runs through the exact
    post-processing the submission uses, and both the kept weights and the decision
    threshold are chosen by it. The raw distributions from the best epoch land in
    checkpoints/val_distributions.csv, ready to plot for the report.
  - AdamW with per-part LRs (encoder / bottleneck / decoder / head) + one cosine curve.
  - No checkpoint is written. The best epoch's weights are held in RAM and, once
    training ends, predict the test set straight into submission.csv -- so what gets
    submitted is exactly the state validation picked, with no file in between.

Everything is hardcoded below. Just: python train.py
"""

import csv
import json
import math
import os
import random
import time
from collections import defaultdict

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
DATA = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
WORK_DIR = "/kaggle/working"
CACHE_DIR = "/kaggle/temp/cache"   # scratch, so the ~6 GB cache is not committed as output

TRAIN_DIR = os.path.join(DATA, "train", "train_images")
TEST_DIR = os.path.join(DATA, "test", "test_images")
ANN_JSON = os.path.join(DATA, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")  # only the val distributions land here
OUT_CSV = os.path.join(WORK_DIR, "submission.csv")
assert os.path.isdir(TRAIN_DIR), f"train images not found at {TRAIN_DIR}"

# --- geometry --------------------------------------------------------------- #
# The solar disk (radius ~1005px) sits centred in the 2048^2 frame, the corners are
# literally black, and the furthest annotated filament vertex is 903px from the centre
# (measured over every polygon in the train JSON), so only a centred CROP is modelled.
#
# Both sizes below sit on the encoder's stride-32 grid, which is the whole point:
# whatever is fed to the network enters and leaves untouched -- no reflect-pad inside
# UNet.forward, no resampling.
#
# CROP is what gets predicted, in a single pass. Half-width 944 clears the furthest
# annotated filament vertex (903px from the centre) by 41px, and stays inside the limb
# (disk radius ~1005) -- which matters, because the model is never trained on the limb
# or the black corners and should not be asked about them.
#
# There is only one size now. Training and inference both see the whole crop, so the
# encoder's attention sees the same 59x59 token map either way and there is no
# train/inference resolution gap left to reason about.
FULL = 2048                     # native image size / submission RLE size
CROP = 1888                     # the whole modelled region, train and predict; 59*32
OFFSET = (FULL - CROP) // 2     # 80
TTA = 8                         # dihedral transforms averaged at inference:
                                # 1 = off, 4 = the rot90s, 8 = those plus a flip

assert (FULL - CROP) % 2 == 0, "CROP must be centrable in FULL"
assert CROP % 32 == 0, "CROP must sit on the grid: it is fed to the network whole"

ENCODER = "efficientvit_b2.r288_in1k"
DECODER_CH = (192, 96, 48, 24, 12)
BOTTLENECK_CH = 192     # width inside the encoder->decoder bottleneck (0 = no bottleneck)
BOTTLENECK_DIL = (1, 2, 4)   # dilations of its 3x3 convs, widening context at /32

EPOCHS = 24
BATCH_SIZE = 1          # a 1888^2 sample is what fits; see ACCUM for the consequence
ACCUM = 4               # optimizer steps once every ACCUM batches, on the mean of
                        # their gradients, so the update is an effective batch of
                        # BATCH_SIZE * ACCUM. Note what this does NOT fix: BatchNorm
                        # normalises over the samples physically in the batch, so at
                        # BATCH_SIZE 1 every norm layer still standardises each frame
                        # by its own statistics, whatever the accumulation does. That
                        # is the real cost of training on the whole crop.
CROPS_PER_IMAGE = 1     # one sample per frame per epoch: the sample IS the whole crop,
                        # so there is no window left to draw and nothing to cover.
FG_POINTS = 128         # foreground pixels cached per image; unused now that samples
                        # are whole crops, kept so an existing cache still validates
# per-part learning rates: the encoder is ImageNet-pretrained and only needs gentle
# fine-tuning, the decoder is random-init, the head is a single 1x1 logit conv
ENCODER_LR = 1e-4
DECODER_LR = 5e-4
HEAD_LR = 5e-4
BOTTLENECK_LR = 5e-4     # random-init like the decoder, so it trains at the same rate
WEIGHT_DECAY = 2e-2
WARMUP_EPOCHS = 2
# Loss: BCE_W * BCE + DICE_W * soft Dice, per sample. A PQ-shaped loss (clDice for
# connectivity, instance-balanced pixel weights) was tried and scored WORSE on
# submission -- the metric being instance-level does not mean a loss shaped like it
# trains better -- so the two terms stay generic.
#
# POS_WEIGHT is the consequential number here. It was 10, on the reasoning that
# filaments are ~0.2% of a frame; but weighting positives by w drives the network's
# output to q = w*p / (1 - p + w*p) for a true posterior p, which systematically
# inflates every probability and drags the decision threshold with it (see
# EVAL_THRESHOLDS below). At 1.0 the network is calibrated: q = p, and 0.5 means 0.5.
# The Dice term already supplies the pressure against predicting empty, which is what
# POS_WEIGHT was really there for. Set it back to 10.0 to undo this in one number.
POS_WEIGHT = 1.0
BCE_W = 0.45
DICE_W = 0.55
# Smoothing added to both ends of the Dice ratio. Large values quietly forgive a tile
# that is nearly all background -- with eps=1 and a 300px filament in a 1M-pixel tile,
# the constant is a tenth of the numerator. Keep it only large enough to avoid 0/0.
DICE_EPS = 1e-6

# The cached mask is a plain 0/1 label. An older layout encoded a per-filament weight
# into the same array (small filaments weighted up, since the metric counts every
# instance once whatever its area); everything downstream only ever tested "> 0", so a
# cache written under that layout still reads correctly and needs no rebuild.
CACHE_VER = 2           # bump when the cache layout changes

# post-processing: validation and the submission run it, so val numbers are the
# submission's numbers. MIN_AREA / CLOSE_KERNEL are the values used when there is no
# validation set to search with; when there is one, evaluate() searches the whole
# EVAL_* grid below and main() submits with whatever won.
MIN_AREA = 150          # drop blobs smaller than this (0.3% of GT instances)
CLOSE_KERNEL = 5        # morphological closing to rejoin broken threads (0 = off)

# Where the optimum sits depends on POS_WEIGHT. Minimising BCE with positives weighted
# by w drives the output to q = w*p / (1 - p + w*p), so a cut at q = 0.5 is really a cut
# at the posterior p = 1/(w+1): at w = 10 that is 0.09, far too liberal, and the optimum
# gets dragged up towards q = w/(1+w) = 0.91. At w = 1 the network is calibrated and the
# optimum should sit near 0.5. The range is left wide on purpose -- it costs one extra
# labelling pass per threshold and removes the risk of the search hitting its own
# ceiling, which a range topping out at 0.7 could have been doing all along.
EVAL_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
EVAL_MIN_AREAS = (100, 150, 250, 400)
EVAL_CLOSE_KERNELS = (0, 5, 9)
# One cell per (threshold, min_area, close_kernel). Cost is one connected-components
# pass per (threshold, kernel), NOT per cell: min_area only ever drops whole components,
# so it is applied as a column mask over an already-computed overlap matrix.
EVAL_GRID = tuple((t, a, k) for t in EVAL_THRESHOLDS
                  for a in EVAL_MIN_AREAS
                  for k in EVAL_CLOSE_KERNELS)

# What the submission is judged on, so what validation reports and selects on:
#   - the DISTRIBUTION of Dice scores      (per instance, and per image)
#   - the DISTRIBUTION of IoU scores       (per instance)
#   - one-to-many / many-to-one relations  (fragmentation and over-merging)
# A prediction and a ground-truth instance count as "related" once one covers this
# fraction of the other, which is what turns raw overlap into a countable relation.
REL_OVERLAP = 0.1
PERCENTILES = (10, 25, 50, 75, 90)
NUM_WORKERS = 4
VAL_FRAC = 0.075
# VAL_FRAC = 0.0 trains on every image and skips validation entirely: no inference
# per epoch, no epoch selection, no threshold search. The last epoch is what gets
# submitted, at NO_VAL_THRESHOLD -- with no held-out set the decision threshold cannot
# be discovered, so it has to be stated. Use this only once the geometry and the
# threshold are settled and the remaining win is the extra VAL_FRAC of training data.
NO_VAL_THRESHOLD = 0.5
SEED = 42

# Per-image intensity normalization. The six GONG sites and eleven years of data
# differ in exposure, flat-fielding and seeing, so a fixed /255 leaves the encoder to
# absorb an instrument offset that has nothing to do with filaments. Each frame is
# centred on the median and scaled by the 16-84 spread of the pixels inside a centred
# NORM_RADIUS disk -- well inside the limb, since the black corners would drag the
# median down and limb darkening would inflate the spread.
NORM_RADIUS = 900       # px; every annotated filament is within 903px of the centre

DEVICE = "cuda"

# float16, not bfloat16: bf16 needs compute capability 8.0+ and Kaggle's T4 (7.5) and
# P100 (6.0) do not have it. float16 has a narrow exponent, so training runs under a
# GradScaler -- correct on every Kaggle GPU, including the ones that could have done
# bf16, at the cost of a little numerical headroom.
AMP_DTYPE = torch.float16

_yy, _xx = np.ogrid[:FULL, :FULL]
DISK_MASK = ((_yy - FULL // 2) ** 2 + (_xx - FULL // 2) ** 2) <= NORM_RADIUS ** 2


def norm_stats(img):
    """(centre, scale) for one full frame: median and 16-84 spread inside the disk.

    Exact for uint8 via a 256-bin histogram, which is a single O(n) pass -- np.percentile
    would sort ~2.5M values per call, and this runs once per image on every val pass.
    """
    hist = np.bincount(img[DISK_MASK], minlength=256).astype(np.float64)
    cdf = np.cumsum(hist) / max(hist.sum(), 1.0)
    lo, mid, hi = np.searchsorted(cdf, (0.16, 0.50, 0.84))
    return float(mid), float(max(hi - lo, 1.0))


def build_norm_cache(img_path, n):
    """Per-image (centre, scale) for every cached train image, measured once.

    Kept out of build_cache so an existing image/mask cache does not have to be rebuilt
    to gain it. Training reads these; validation and the submission recompute the same
    numbers from the frame in hand, so the two paths agree by construction.
    """
    path = os.path.join(CACHE_DIR, f"norm_{FULL}_v{CACHE_VER}.npy")
    if os.path.exists(path):
        return np.load(path)
    images = np.load(img_path, mmap_mode="r")
    out = np.zeros((n, 2), dtype=np.float32)
    for i in tqdm(range(n), desc="[cache] intensity stats", unit="img"):
        out[i] = norm_stats(np.asarray(images[i]))
    np.save(path, out)
    return out

# --------------------------------------------------------------------------- #
# cache building
# --------------------------------------------------------------------------- #
def build_cache():
    """Decode every train JPEG + rasterize its polygons once, at native resolution.

    The FULL frame is cached, not the crop: augmentation runs on the whole frame and
    the crop is cut out afterwards, so the rotation draws the crop's corners from real
    pixels beyond it. All cached coordinates are in full-frame space.

    Returns (img_path, msk_path, fg_path, cnt_path, names). The two big arrays are
    ~2.6 GB each; they are written as .npy memmaps and read lazily afterwards.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    img_path = os.path.join(CACHE_DIR, f"images_{FULL}_v{CACHE_VER}.npy")
    msk_path = os.path.join(CACHE_DIR, f"masks_{FULL}_v{CACHE_VER}.npy")
    fg_path = os.path.join(CACHE_DIR, f"fgpts_{FULL}_v{CACHE_VER}.npy")
    cnt_path = os.path.join(CACHE_DIR, f"fgcnt_{FULL}_v{CACHE_VER}.npy")
    names_path = os.path.join(CACHE_DIR, f"names_{FULL}_v{CACHE_VER}.json")

    if all(os.path.exists(p) for p in (img_path, msk_path, fg_path, cnt_path, names_path)):
        names = json.load(open(names_path))
        print(f"[cache] reusing {len(names)} cached images at {FULL}px")
        return img_path, msk_path, fg_path, cnt_path, names

    ann = json.load(open(ANN_JSON))

    # image_id -> file_name  (several image_ids map to the same file: annotator passes)
    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    id2hw = {im["id"]: (im["height"], im["width"]) for im in ann["images"]}

    # file_name -> list of polygons
    file2polys = defaultdict(list)
    for a in ann["annotations"]:
        file2polys[id2file[a["image_id"]]].append((a["segmentation"], id2hw[a["image_id"]]))

    names = sorted(f for f in file2polys if os.path.exists(os.path.join(TRAIN_DIR, f)))
    n = len(names)
    print(f"[cache] building full-frame cache for {n} images "
          f"(~{2 * n * FULL * FULL / 1e9:.1f} GB, one-off)...")

    images = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8, shape=(n, FULL, FULL))
    masks = np.lib.format.open_memmap(msk_path, mode="w+", dtype=np.uint8, shape=(n, FULL, FULL))
    fg = np.zeros((n, FG_POINTS, 2), dtype=np.int16)
    cnt = np.zeros(n, dtype=np.int32)
    rng = np.random.RandomState(SEED)
    sl = slice(OFFSET, OFFSET + CROP)
    kept_px = dropped_px = 0

    t0 = time.time()
    for i, fn in enumerate(names):
        img = cv2.imread(os.path.join(TRAIN_DIR, fn), cv2.IMREAD_GRAYSCALE)
        if img.shape != (FULL, FULL):
            img = cv2.resize(img, (FULL, FULL), interpolation=cv2.INTER_AREA)
        images[i] = img

        merged = np.zeros((FULL, FULL), dtype=np.uint8)
        for seg, (ah, aw) in file2polys[fn]:
            rles = coco_mask.frPyObjects(seg, ah, aw)
            m = coco_mask.decode(coco_mask.merge(rles))
            if m.shape != (FULL, FULL):
                m = cv2.resize(m, (FULL, FULL), interpolation=cv2.INTER_NEAREST)
            np.maximum(merged, m, out=merged)
        masks[i] = merged

        # the crop is only safe if it throws away no ground truth -- keep score
        kept = int(merged[sl, sl].sum())
        kept_px += kept
        dropped_px += int(merged.sum()) - kept

        # anchor points come from inside the crop: tiles never start outside it
        ys, xs = np.nonzero(merged[sl, sl])
        ys, xs = ys + OFFSET, xs + OFFSET
        if len(ys):
            pick = rng.choice(len(ys), size=min(FG_POINTS, len(ys)), replace=False)
            fg[i, :len(pick)] = np.stack([ys[pick], xs[pick]], axis=1)
            cnt[i] = len(pick)

        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}  ({time.time() - t0:.0f}s)")

    images.flush()
    masks.flush()
    np.save(fg_path, fg)
    np.save(cnt_path, cnt)
    json.dump(names, open(names_path, "w"))
    print(f"[cache] done in {time.time() - t0:.0f}s")
    print(f"[cache] filament pixels kept {kept_px:,} / dropped by the crop {dropped_px:,} "
          f"({100 * dropped_px / max(kept_px + dropped_px, 1):.4f}%)")
    return img_path, msk_path, fg_path, cnt_path, names

# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def normalize(batch_u8, centre, scale):
    """uint8 (B, H, W) numpy -> (B, 1, H, W) tensor on DEVICE, per-image normalized."""
    x = torch.from_numpy(batch_u8).to(DEVICE).float().sub_(centre).div_(scale)
    return x.unsqueeze(1).to(memory_format=torch.channels_last)


@torch.no_grad()
def predict_frame(model, img):
    """FULL x FULL probability map for one frame, in a single forward pass.

    img is a uint8 FULL x FULL array. The centred CROP is fed to the network whole --
    it is a multiple of 32, so it enters and leaves untouched -- and written back into
    a full-frame canvas at the same coordinates.

    Pixels outside the crop stay 0. They are the limb and the black sky beyond it: no
    training tile ever contained them and they hold no ground truth. The canvas is
    full-frame only because that is the size the submission RLE expects.
    """
    def forward(x):
        """Mean sigmoid over the first TTA elements of the dihedral group.

        Training applies flips and RandomRotate90(p=1.0), so the model is already close
        to D4-equivariant and the transforms mostly disagree on noise. Averaging cancels
        that, which matters here specifically because the noise it cancels is what
        breaks a faint filament into two components and costs one-to-many.

        Each transform is undone on the OUTPUT before averaging, so this averages
        predictions in image space, never images.
        """
        if TTA <= 1:
            return torch.sigmoid(model(x))
        outs = []
        for flip in range(2 if TTA >= 8 else 1):
            xf = torch.flip(x, dims=[-1]) if flip else x
            for k in range(4 if TTA >= 4 else 1):
                xr = torch.rot90(xf, k, dims=(-2, -1))
                y = torch.sigmoid(model(xr.contiguous(memory_format=torch.channels_last)))
                y = torch.rot90(y, -k, dims=(-2, -1))
                outs.append(torch.flip(y, dims=[-1]) if flip else y)
        return torch.stack(outs).mean(0)

    centre, scale = norm_stats(img)
    sl = slice(OFFSET, OFFSET + CROP)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        p = forward(normalize(np.ascontiguousarray(img[sl, sl])[None],
                              centre, scale)).float()
    out = torch.zeros((FULL, FULL), device=DEVICE, dtype=torch.float32)
    out[sl, sl] = p[0, 0]
    return out

# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
# Augmentation strength lives here because the sampling margin is derived from it --
# keep them together so the two can never drift apart.
AUG_ROTATE = 25          # degrees
AUG_SCALE = (0.9, 1.1)
AUG_TRANSLATE = 0.05     # fraction of the window

TRAIN_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),
    A.Affine(scale=AUG_SCALE, translate_percent=(-AUG_TRANSLATE, AUG_TRANSLATE),
             rotate=(-AUG_ROTATE, AUG_ROTATE), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
])

# The augmentation runs on the WHOLE 2048 frame and the crop is cut out afterwards,
# which is why no margin has to be reserved: a rotation pulls the crop's corners from
# further out, and further out is genuinely there. Beyond the frame edge albumentations
# fills with 0, and that is correct rather than merely tolerable -- the frame edge sits
# at radius 1024 from the centre while the solar disk ends at ~1005, so everything the
# fill replaces was black sky already.


class FilamentCrops(Dataset):
    """One augmented CROP per frame, straight from the full-res cache.

    There is no sampling left to do: the sample is the entire modelled region, so the
    foreground-anchored crop selection this class used to need is gone with it. The
    frame is augmented whole and the centred crop cut out of the result.
    """

    def __init__(self, img_path, msk_path, norm, indices):
        self.img_path, self.msk_path = img_path, msk_path
        self.norm = norm
        self.indices = indices
        self.images = None
        self.masks = None

    def __len__(self):
        return len(self.indices) * CROPS_PER_IMAGE

    def __getitem__(self, i):
        if self.images is None:  # open memmaps lazily (per worker)
            self.images = np.load(self.img_path, mmap_mode="r")
            self.masks = np.load(self.msk_path, mmap_mode="r")
        j = int(self.indices[i // CROPS_PER_IMAGE])

        # augment the WHOLE frame, then cut the crop out of the result: the rotation
        # then draws the crop's corners from real pixels further out instead of from
        # a reserved margin that would have to be carried around
        out = TRAIN_AUG(image=np.asarray(self.images[j]),
                        mask=np.asarray(self.masks[j]))
        cut = slice(OFFSET, OFFSET + CROP)
        img, msk = out["image"][cut, cut], out["mask"][cut, cut]

        label = (np.ascontiguousarray(msk) > 0).astype(np.float32)
        # the frame's own centre and scale, measured before augmentation
        centre, scale = self.norm[j]
        x_t = torch.from_numpy(np.ascontiguousarray(img)).float().sub_(centre).div_(scale)
        return x_t.unsqueeze(0), torch.from_numpy(label).unsqueeze(0)

# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Bottleneck(nn.Module):
    """Residual squeeze-transform-expand block sitting between encoder and decoder.

    The encoder hands over its deepest stage (/32) and the decoder starts upsampling
    from it immediately, so nothing ever mixes information across that map at full
    width. This block does: 1x1 down to BOTTLENECK_CH, a stack of 3x3 convs with
    growing dilation (context out to several hundred input pixels at /32, which is the
    scale a filament spine spans), 1x1 back up, added residually.

    The final BatchNorm is zero-initialised, so at step 0 the block is exactly the
    identity and training starts from the same place the plain U-Net would.
    """

    def __init__(self, ch, mid, dilations):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(mid, mid, 3, padding=d, dilation=d, bias=False),
                          nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
            for d in dilations])
        self.expand = nn.Sequential(
            nn.Conv2d(mid, ch, 1, bias=False), nn.BatchNorm2d(ch))
        nn.init.zeros_(self.expand[1].weight)   # start as identity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.expand(self.convs(self.reduce(x))))


class UNet(nn.Module):
    """timm features_only encoder + U-Net decoder that upsamples back to /1.

    The decoder is driven by the encoder's stage reductions, so it works whether
    the encoder exposes 5 stages starting at /2 (resnet34: 2 4 8 16 32) or 4
    stages starting at /4 (efficientvit_b1: 4 8 16 32). Levels without a matching
    encoder stage simply get no skip connection.
    """

    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            ENCODER, pretrained=True, features_only=True, in_chans=1
        )
        enc_ch = self.encoder.feature_info.channels()
        self.enc_red = self.encoder.feature_info.reduction()
        red2ch = dict(zip(self.enc_red, enc_ch))

        deepest = max(self.enc_red)
        n_up = int(math.log2(deepest))          # /32 -> 5 upsampling steps to /1
        cin = enc_ch[-1]
        blocks, skip_reds = [], []
        for i in range(n_up):
            red = deepest // (2 ** (i + 1))     # resolution level after this upsample
            skip_ch = red2ch.get(red, 0)
            cout = DECODER_CH[min(i, len(DECODER_CH) - 1)]
            blocks.append(ConvBlock(cin + skip_ch, cout))
            skip_reds.append(red if skip_ch else None)
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.skip_reds = skip_reds
        self.bottleneck = (Bottleneck(enc_ch[-1], BOTTLENECK_CH, BOTTLENECK_DIL)
                           if BOTTLENECK_CH else nn.Identity())
        self.head = nn.Conv2d(cin, 1, 1)

    def forward(self, x):
        # The decoder doubles resolution 5x, so the encoder stages only line up with
        # their skips when the input is a multiple of 32. The geometry is chosen
        # so the one size ever fed here is a multiple of 32 by construction: CROP
        # (1888 = 59*32), which is what the assert up top is for. This pad is a no-op
        # at the configured geometry and only fires if CROP is retuned off the grid, in
        # which case it adds <32px of reflected border the crop below takes back off.
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % 32, (-w) % 32
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        feats = self.encoder(x)
        red2feat = dict(zip(self.enc_red, feats))
        out = self.bottleneck(feats[-1])
        for block, red in zip(self.blocks, self.skip_reds):
            out = F.interpolate(out, scale_factor=2, mode="nearest")
            if red is not None:
                out = torch.cat([out, red2feat[red]], dim=1)
            out = block(out)
        out = self.head(out)
        return out[..., :h, :w] if (pad_h or pad_w) else out


def build_param_groups(model):
    """One optimizer group per model part, so each part trains at its own LR.

    Within every part, params with ndim <= 1 (norm weights, biases) are split off
    into a weight_decay=0 group -- the usual pairing once groups are built by hand.
    """
    parts = (("encoder", model.encoder, ENCODER_LR),
             ("bottleneck", model.bottleneck, BOTTLENECK_LR),
             ("decoder", model.blocks, DECODER_LR),
             ("head", model.head, HEAD_LR))

    groups, covered = [], 0
    for part, module, lr in parts:
        trainable = [p for p in module.parameters() if p.requires_grad]
        decay = [p for p in trainable if p.ndim > 1]
        no_decay = [p for p in trainable if p.ndim <= 1]
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": WEIGHT_DECAY, "part": part})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "part": part})
        n_part = sum(p.numel() for p in trainable)
        covered += n_part
        print(f"[optim] {part:8s} lr={lr:.1e}  {n_part / 1e6:6.2f}M params  "
              f"({len(decay)} decayed / {len(no_decay)} not)")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert covered == total, f"param groups cover {covered} of {total} trainable params"
    return groups

# --------------------------------------------------------------------------- #
# loss / metric
# --------------------------------------------------------------------------- #
def criterion(logits, target, pos_weight):
    """BCE + soft Dice over the whole tile.

    Every pixel of the training tile is supervised. Nothing is masked out: the tile is
    a random window drawn wholly inside the crop, so every pixel in it is a pixel the
    network will also be asked about at inference, when the crop goes through whole.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

    # .float() is load-bearing under float16 autocast: p.sum(1) now runs over CROP^2 =
    # 3,564,544 probabilities and float16 tops out at 65,504, so a frame more than
    # ~1.8% foreground would overflow the sum to inf and take the whole loss with it.
    # The margin shrank with the sample size -- this line is not optional.
    prob = torch.sigmoid(logits.float())
    p, t = prob.flatten(1), target.flatten(1)
    # per SAMPLE, then averaged -- not pooled over the batch. Pooling would let one
    # tile packed with filament carry the gradient for a tile that has almost none.
    inter = (p * t).sum(1)
    dice = (1.0 - (2.0 * inter + DICE_EPS)
            / (p.sum(1) + t.sum(1) + DICE_EPS)).mean()

    return BCE_W * bce + DICE_W * dice

# --------------------------------------------------------------------------- #
# instances + Panoptic Quality (the ranking metric)
# --------------------------------------------------------------------------- #
def instance_labels(binary, min_area=MIN_AREA, close_kernel=CLOSE_KERNEL):
    """Binary mask -> (int32 label map, count), 0 = background, labels are 1..count.

    Shared by validation and the submission so val scores exactly what gets sent.
    """
    if close_kernel:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = [lab for lab in range(1, n) if stats[lab, cv2.CC_STAT_AREA] >= min_area]
    lut = np.zeros(max(n, 1), dtype=np.int32)
    lut[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    return lut[labels], len(keep)


def overlap_matrices(gt_lab, n_gt, pr_lab, n_pr):
    """Pairwise intersection / IoU / Dice between ground-truth and predicted instances.

    Returns (inter, iou, dice, gt_area, pr_area), the three matrices shaped
    (n_gt, n_pr). One bincount over the label pairs does all of it, so the cost is a
    single pass over the image rather than a loop over instance pairs.
    """
    gt_area = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)[1:].astype(np.float64)
    pr_area = np.bincount(pr_lab.ravel(), minlength=n_pr + 1)[1:].astype(np.float64)
    if n_gt == 0 or n_pr == 0:
        empty = np.zeros((n_gt, n_pr), dtype=np.float64)
        return empty, empty.copy(), empty.copy(), gt_area, pr_area

    pair = gt_lab.astype(np.int64) * (n_pr + 1) + pr_lab
    inter = np.bincount(pair.ravel(), minlength=(n_gt + 1) * (n_pr + 1))
    inter = inter.reshape(n_gt + 1, n_pr + 1)[1:, 1:].astype(np.float64)

    total = gt_area[:, None] + pr_area[None, :]
    iou = np.where(total - inter > 0, inter / np.maximum(total - inter, 1e-9), 0.0)
    dice = np.where(total > 0, 2.0 * inter / np.maximum(total, 1e-9), 0.0)
    return inter, iou, dice, gt_area, pr_area


def relation_counts(inter, gt_area, pr_area):
    """(one-to-many, many-to-one) counts for one image.

    one-to-many: a ground-truth filament covered by two or more predictions, i.e. the
    prediction fragmented it. many-to-one: one prediction covering two or more
    ground-truth filaments, i.e. it merged them. Both are judged against REL_OVERLAP
    rather than a match threshold -- a break only has to be a break, not a miss.
    """
    if inter.size == 0:
        return 0, 0
    gt_cover = inter / np.maximum(gt_area[:, None], 1e-9)
    pr_cover = inter / np.maximum(pr_area[None, :], 1e-9)
    return (int(((gt_cover >= REL_OVERLAP).sum(axis=1) >= 2).sum()),
            int(((pr_cover >= REL_OVERLAP).sum(axis=0) >= 2).sum()))


def summarize(values, prefix):
    """Mean, percentiles and tail fractions of one distribution, as flat metrics."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {f"{prefix}_mean": 0.0, **{f"{prefix}_p{p:02d}": 0.0 for p in PERCENTILES},
                f"{prefix}_ge50": 0.0, f"{prefix}_ge75": 0.0}
    out = {f"{prefix}_mean": float(values.mean())}
    for p in PERCENTILES:
        out[f"{prefix}_p{p:02d}"] = float(np.percentile(values, p))
    out[f"{prefix}_ge50"] = float((values >= 0.5).mean())
    out[f"{prefix}_ge75"] = float((values >= 0.75).mean())
    return out


def write_distributions(dist, path):
    """Dump the raw per-instance / per-image scores so the report can plot them."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["population", "value"])
        for name, values in dist.items():
            for v in np.asarray(values).ravel():
                writer.writerow([name, f"{float(v):.6f}"])


@torch.no_grad()
def evaluate(model, images, masks, val_idx):
    """Score the val set on the judged criteria, at each decision threshold.

    Runs the full inference path the submission uses -- whole-crop prediction, closing,
    area filtering, connected components -- so these are the submission's numbers, and
    it does so for every cell of EVAL_GRID rather than for the threshold alone. Blob
    area and closing move missed_rate and spurious_rate as hard as the threshold does,
    and all three enter the score as zeros, so leaving two of them hand-set was leaving
    score on the table.

    The grid is cheap because min_area only ever DROPS whole components: the image is
    labelled once per (threshold, kernel) with no area filter, and each min_area is then
    a column mask over that one overlap matrix. Cost is |thresholds| x |kernels| passes,
    not |grid|.

    For every cell it collects, over the whole val set:

      * per-instance Dice and IoU against the best-overlapping counterpart, with missed
        ground truth entering as 0 and spurious predictions entering as 0
      * per-image pixel Dice
      * one-to-many (fragmentation) and many-to-one (over-merging) relation counts

    Returns {(threshold, min_area, close_kernel): {metric: value}}; the "_dist" entry
    carries the raw arrays so the distributions can be written out, not just summaries.
    """
    model.eval()
    acc = {key: {"dice_gt": [], "iou_gt": [], "dice_pred": [], "dice_img": [],
                 "one_to_many": 0, "many_to_one": 0, "n_gt": 0, "n_pred": 0,
                 "missed": 0, "spurious": 0} for key in EVAL_GRID}

    # the bar reports the running per-image Dice at one middle cell -- not the selection
    # score (that needs the whole set), just something moving that tracks it
    probe = EVAL_GRID[len(EVAL_GRID) // 2]
    bar = tqdm(val_idx, desc="  val", leave=False, dynamic_ncols=True, unit="img")
    for j in bar:
        prob = predict_frame(model, np.asarray(images[j]))
        gt_bin = (np.asarray(masks[j]) > 0).astype(np.uint8)
        # ground truth is never area-filtered: a filament too small to survive
        # post-processing still has to count as a miss
        gt_lab, n_gt = instance_labels(gt_bin, min_area=0, close_kernel=0)
        gt_total = float(gt_bin.sum())

        for t in EVAL_THRESHOLDS:
            pred = (prob > t).to(torch.uint8).cpu().numpy()   # one transfer per threshold
            for k in EVAL_CLOSE_KERNELS:
                # label with NO area filter, so every min_area below is a column mask
                pr_lab, n_pr = instance_labels(pred, min_area=0, close_kernel=k)
                inter, iou, dice, gt_area, pr_area = overlap_matrices(
                    gt_lab, n_gt, pr_lab, n_pr)

                for area in EVAL_MIN_AREAS:
                    sel = pr_area >= area
                    n_sel = int(sel.sum())
                    it, iu, di = inter[:, sel], iou[:, sel], dice[:, sel]

                    a = acc[(t, area, k)]
                    a["n_gt"] += n_gt
                    a["n_pred"] += n_sel
                    if n_gt:
                        best_g = di.max(axis=1) if n_sel else np.zeros(n_gt)
                        a["dice_gt"].append(best_g)
                        a["iou_gt"].append(iu.max(axis=1) if n_sel else np.zeros(n_gt))
                        a["missed"] += int((best_g == 0).sum())
                    if n_sel:
                        best_p = di.max(axis=0) if n_gt else np.zeros(n_sel)
                        a["dice_pred"].append(best_p)
                        a["spurious"] += int((best_p == 0).sum())

                    o2m, m2o = relation_counts(it, gt_area, pr_area[sel])
                    a["one_to_many"] += o2m
                    a["many_to_one"] += m2o

                    # per-image pixel Dice straight off the overlap matrix: the kept
                    # predictions' total intersection with any GT over the two masses.
                    # No filtered mask ever has to be materialised.
                    pred_px = float(pr_area[sel].sum())
                    inter_px = float(it.sum())
                    a["dice_img"].append(2 * inter_px / max(pred_px + gt_total, 1.0))

        bar.set_postfix_str(f"dice {np.mean(acc[probe]['dice_img']):.4f}", refresh=False)

    out = {}
    for key, a in acc.items():
        dice_gt = np.concatenate(a["dice_gt"]) if a["dice_gt"] else np.zeros(0)
        iou_gt = np.concatenate(a["iou_gt"]) if a["iou_gt"] else np.zeros(0)
        dice_pred = np.concatenate(a["dice_pred"]) if a["dice_pred"] else np.zeros(0)
        img = np.asarray(a["dice_img"], dtype=np.float64)

        m = {}
        m.update(summarize(dice_gt, "dice_inst"))     # distribution of Dice scores
        m.update(summarize(iou_gt, "iou_inst"))       # distribution of IoU scores
        m.update(summarize(dice_pred, "dice_pred"))   # same, seen from the prediction side
        m.update(summarize(img, "dice_img"))
        m["one_to_many"] = a["one_to_many"]
        m["many_to_one"] = a["many_to_one"]
        m["one_to_many_rate"] = a["one_to_many"] / max(a["n_gt"], 1)
        m["many_to_one_rate"] = a["many_to_one"] / max(a["n_pred"], 1)
        m["missed_rate"] = a["missed"] / max(a["n_gt"], 1)
        m["spurious_rate"] = a["spurious"] / max(a["n_pred"], 1)
        m["n_gt"] = a["n_gt"]
        m["n_pred"] = a["n_pred"]
        # Selection needs one number out of these distributions: the centre of the
        # instance-Dice distribution from both directions. Missed filaments enter the
        # ground-truth side as 0 and spurious ones the prediction side as 0, so neither
        # under- nor over-detection can win.
        m["score"] = 0.5 * (m["dice_inst_mean"] + m["dice_pred_mean"])
        m["_dist"] = {"dice_inst": dice_gt, "iou_inst": iou_gt,
                      "dice_pred": dice_pred, "dice_img": img}
        out[key] = m
    return out

# --------------------------------------------------------------------------- #
# submission
# --------------------------------------------------------------------------- #
class TestImages(Dataset):
    """Whole FULL^2 greyscale test images; cropping and tiling happen GPU-side.

    The full frame is loaded, not the crop: the submission RLE is FULL x FULL, so
    predict_frame paints the crop onto a full-frame canvas.
    """

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img.shape != (FULL, FULL):
            img = cv2.resize(img, (FULL, FULL), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img), os.path.splitext(os.path.basename(path))[0]


def encode_rle(mask):
    """COCO compressed RLE counts string for a full-res binary mask."""
    rle = coco_mask.encode(np.asfortranarray(mask))
    return rle["counts"].decode("ascii")


@torch.no_grad()
def write_submission(model, threshold, min_area, close_kernel):
    """Predict the test set and write one CSV row per filament:

        filament_id,segmentation_rle
        20150125172714Mh_1,<COCO RLE counts>

      - one row per filament, not per image; scoring matches predicted instances to
        ground-truth ones by overlap, so fragmenting one filament into pieces
        (one-to-many) or merging two into one (many-to-one) both count against you
      - segmentation_rle is the COCO *compressed* RLE counts string only (no size, no
        quotes). RLE is column-major and the size is fixed at FULL x FULL
      - images with no detected filament simply contribute no rows

    This is the same path evaluate() scored, so the (threshold, min_area, close_kernel)
    triple -- picked on validation -- is applied at exactly the operating point it was
    chosen at.
    """
    model.eval()
    files = sorted(os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
                   if f.lower().endswith((".jpeg", ".jpg", ".png")))
    print(f"[sub] {len(files)} test images in {TEST_DIR}")

    # one image at a time: the whole crop is a single forward pass in predict_frame
    loader = DataLoader(TestImages(files), batch_size=1, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=False)

    rows, areas, empty = [], [], 0
    t0 = time.time()
    for img_t, stems in tqdm(loader, desc="  test", leave=False, dynamic_ncols=True,
                             unit="img"):
        # predict_frame returns the whole FULL^2 canvas: the band outside the centred
        # crop stays 0, which is correct -- it is the limb and black sky beyond it
        prob = predict_frame(model, img_t[0].numpy())
        labels, n = instance_labels((prob > threshold).to(torch.uint8).cpu().numpy(),
                                    min_area=min_area, close_kernel=close_kernel)
        if not n:
            empty += 1
        for lab in range(1, n + 1):
            inst = (labels == lab).astype(np.uint8)
            rows.append((f"{stems[0]}_{lab}", encode_rle(inst)))
            areas.append(int(inst.sum()))
    elapsed = time.time() - t0

    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)  # RLE chars are ASCII 48..111, so nothing ever gets quoted
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)

    # ---- sanity checks -------------------------------------------------- #
    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate filament_id in submission"
    if rows:
        probe = coco_mask.decode({"counts": rows[0][1].encode("ascii"), "size": [FULL, FULL]})
        assert probe.sum() == areas[0], "RLE round-trip mismatch"
        assert not any("," in r[1] or '"' in r[1] for r in rows), "RLE contains a CSV-unsafe char"

    print(f"[sub] {len(rows)} filaments over {len(files)} images "
          f"({len(rows) / max(len(files), 1):.1f} per image), "
          f"{empty} images with no detection")
    if areas:
        print(f"[sub] instance area px: min {min(areas)}  "
              f"median {int(np.median(areas))}  max {max(areas)}")
    print(f"[sub] inference {elapsed:.1f}s "
          f"({elapsed / max(len(files), 1) * 1000:.0f} ms/image)")
    print(f"[sub] wrote {OUT_CSV}")


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    os.makedirs(CKPT_DIR, exist_ok=True)

    img_path, msk_path, fg_path, cnt_path, names = build_cache()

    idx = np.arange(len(names))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    n_val = int(len(idx) * VAL_FRAC)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    validating = len(val_idx) > 0
    if validating:
        print(f"[data] train={len(train_idx)} images x {CROPS_PER_IMAGE} crops   "
              f"val={len(val_idx)} images, predicted whole")
    else:
        print(f"[data] train={len(train_idx)} images x {CROPS_PER_IMAGE} crops   "
              f"val=0 (VAL_FRAC=0) -- validation skipped, the LAST epoch is submitted "
              f"at thr {NO_VAL_THRESHOLD:.1f}")
    print(f"[data] {CROP}px centred crop, whole, for both training and prediction "
          f"({CROP} = {CROP // 32}*32, so nothing is padded)")
    print(f"[opt]  batch {BATCH_SIZE} x accum {ACCUM} = {BATCH_SIZE * ACCUM} "
          f"frames per optimizer step")

    norm = build_norm_cache(img_path, len(names))
    train_ds = FilamentCrops(img_path, msk_path, norm, train_idx)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                          pin_memory=True, drop_last=True, persistent_workers=NUM_WORKERS > 0)
    # validation predicts whole images, so it reads the cache directly
    val_images = np.load(img_path, mmap_mode="r")
    val_masks = np.load(msk_path, mmap_mode="r")

    model = UNet().to(DEVICE).to(memory_format=torch.channels_last)
    # every group carries its own lr / weight_decay; the cosine lambda scales them all
    opt = torch.optim.AdamW(build_param_groups(model), lr=DECODER_LR, weight_decay=WEIGHT_DECAY)
    pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)

    # the schedule advances per OPTIMIZER step, not per batch, or accumulation would
    # silently compress the whole cosine curve into 1/ACCUM of the run
    batches_per_epoch = max(len(train_dl), 1)
    steps_per_epoch = max(batches_per_epoch // ACCUM, 1)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    pos_rate = float(val_masks[::20].mean())  # subsampled: the full array is ~3 GB
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {ENCODER}  {n_params / 1e6:.1f}M params   fg pixel rate {pos_rate:.5f}")

    scaler = torch.amp.GradScaler("cuda")
    best, best_state, best_key = 0.0, None, EVAL_GRID[0]
    for epoch in range(EPOCHS):
        model.train()
        t0, running = time.time(), 0.0
        bar = tqdm(train_dl, desc=f"  epoch {epoch + 1:3d}/{EPOCHS}", leave=False,
                   dynamic_ncols=True, unit="step")
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(bar, 1):
            x = x.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                loss = criterion(model(x), y, pos_weight)
            # divide before backward so the accumulated gradient is the MEAN over the
            # ACCUM batches rather than their sum -- otherwise the effective learning
            # rate would scale with ACCUM
            scaler.scale(loss / ACCUM).backward()

            if step % ACCUM == 0 or step == batches_per_epoch:
                # unscale before clipping, or the clip norm would be measured against
                # scaled gradients and effectively do nothing
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

            running += loss.item()
            # running mean, not the last batch's value -- a single frame's loss swings
            # hard depending on how much filament it happens to hold
            bar.set_postfix_str(f"loss {running / step:.4f}", refresh=False)

        train_time = time.time() - t0
        train_loss = running / batches_per_epoch
        part_lrs = {f"lr_{g['part']}": g["lr"] for g in opt.param_groups}
        lr_txt = " ".join(f"{k[3:7]} {v:.1e}" for k, v in part_lrs.items())

        if not validating:
            # nothing to select on, so nothing to snapshot either: training just runs
            # to the end and the weights in hand at that point are the submission
            print(f"epoch {epoch + 1:3d}/{EPOCHS}  loss {train_loss:.4f}  {lr_txt}  "
                  f"{train_time:5.1f}s")
            continue

        results = evaluate(model, val_images, val_masks, val_idx)
        epoch_key = max(results, key=lambda key: results[key]["score"])
        r = results[epoch_key]
        eval_time = time.time() - t0 - train_time
        print(f"epoch {epoch + 1:3d}/{EPOCHS}  loss {train_loss:.4f}  {lr_txt}  "
              f"{train_time:5.1f}s+{eval_time:4.1f}s")
        print(f"          @thr {epoch_key[0]:.1f} area {epoch_key[1]} close "
              f"{epoch_key[2]}  inst dice {r['dice_inst_mean']:.3f} "
              f"(p25 {r['dice_inst_p25']:.3f} p50 {r['dice_inst_p50']:.3f} "
              f"p75 {r['dice_inst_p75']:.3f}, >=.5 {100 * r['dice_inst_ge50']:.0f}%)  "
              f"iou {r['iou_inst_mean']:.3f}  img dice {r['dice_img_mean']:.3f}")
        print(f"          1->many {r['one_to_many']} ({100 * r['one_to_many_rate']:.1f}% of gt)"
              f"  many->1 {r['many_to_one']} ({100 * r['many_to_one_rate']:.1f}% of pred)"
              f"  missed {100 * r['missed_rate']:.1f}%"
              f"  spurious {100 * r['spurious_rate']:.1f}%  score {r['score']:.4f}")

        if r["score"] > best:
            best, best_key = r["score"], epoch_key
            # nothing is written to disk: the winning weights are snapshotted to CPU
            # RAM and stay there until they predict the test set below
            best_state = {k: v.detach().to("cpu", copy=True)
                          for k, v in model.state_dict().items()}
            # keep the winning distributions themselves, not just their summaries
            dist_path = os.path.join(CKPT_DIR, "val_distributions.csv")
            write_distributions(r["_dist"], dist_path)
            print(f"  -> kept (score {best:.4f} @ thr {epoch_key[0]:.1f} "
                  f"area {epoch_key[1]} close {epoch_key[2]})")

    if validating:
        if best_state is None:
            raise SystemExit("no epoch scored above 0 -- nothing to submit")
        print(f"[done] best val score = {best:.4f} @ thr {best_key[0]:.1f} "
              f"area {best_key[1]} close {best_key[2]}, epoch weights restored")
        model.load_state_dict(best_state)
    else:
        # nothing was searched, so the hand-set post-processing is what ships
        best_key = (NO_VAL_THRESHOLD, MIN_AREA, CLOSE_KERNEL)
        print(f"[done] no validation set -- submitting the final epoch's weights "
              f"at thr {best_key[0]:.1f} area {best_key[1]} close {best_key[2]}")
    write_submission(model, *best_key)


if __name__ == "__main__":
    main()
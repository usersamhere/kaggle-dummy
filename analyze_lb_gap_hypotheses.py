#!/usr/bin/env python3
"""Audit the V88 offline/public-LB gap and test safe prediction-only variants.

The GT caches are read exclusively for retrospective scoring/stratification.
Candidate rules use only submitted masks, saved model predictions, and image
membership in the old benchmark split. Nothing here contacts Kaggle or submits.

Run: python analyze_lb_gap_hypotheses.py
Optional: --no-morphology skips the relatively expensive 1-pixel contour sweep.
"""
from __future__ import annotations

import argparse
import pickle
import re

import numpy as np
import pandas as pd
from pycocotools import mask as mu

from generate_breakthrough_v89 import ROOT, as_rle, eval_pq, read_predictions

BASE_PATH = ROOT / "submission_champion_v88_sovereign_apex_063.csv"
FULL_PATH = ROOT / "full_180_gt_cache.pkl"
KNOWN_PATH = ROOT / "test_gt_cache.pkl"
MODEL_PATHS = {
    "tokyo": ROOT / "kernel_outputs/tokyo-5-fold-convnext-ensemble-inference/submission.csv",
    "anthony": ROOT / "kernel_outputs/anthony-blended-unet-1888-solar-filament/submission.csv",
    "lb70": ROOT / "kernel_outputs/lb70-topology-safe-solar-filaments-v2/submission.csv",
    "shlok": ROOT / "kernel_outputs/shlok-native-5-tile-unet-inference/submission.csv",
}
SQ_CAP = 0.79997


def fmt(m):
    return (f"PQ={m['PQ']:.6f} SQ={m['SQ']:.6f} RQ={m['RQ']:.6f} "
            f"TP/FP/FN={m['TP']}/{m['FP']}/{m['FN']}")


def audit_submission(path, expected_ids):
    df = pd.read_csv(path, dtype=str)
    image = df.filament_id.str.rsplit("_", n=1).str[0]
    # Disjointness is an explicit submission safety condition.
    overlaps = 0
    for _, group in df.groupby(image, sort=False):
        rs = [as_rle(x) for x in group.segmentation_rle]
        if len(rs) > 1:
            mat = np.asarray(mu.iou(rs, rs, [0] * len(rs)))
            overlaps += int(np.triu(mat > 0, k=1).sum())
    return len(df), set(image) == set(expected_ids), overlaps


def report(label, pred, full_gt, known_ids, novel_ids):
    full = eval_pq(pred, full_gt)
    known = eval_pq(pred, full_gt, known_ids)
    novel = eval_pq(pred, full_gt, novel_ids)
    print(f"{label:28s} full {fmt(full)} | known {fmt(known)} | novel {fmt(novel)}")
    return full, known, novel


def group_by_annotator_count(pred, gt):
    groups = {n: [] for n in (1, 2, 3)}
    for im, layers in gt.items():
        if len(layers) in groups:
            groups[len(layers)].append(im)
    print("\nAnnotator multiplicity strata (same predictions, cache is scoring reference):")
    for n, ids in groups.items():
        if ids:
            print(f"  annotators={n}, images={len(ids):3d}: {fmt(eval_pq(pred, gt, ids))}")


def support_predictions(base, models, min_votes):
    """Keep base instances with >= min_votes model masks having any overlap."""
    out = {}
    for im, instances in base.items():
        base_rs = [as_rle(r) for r, _ in instances]
        votes = np.zeros(len(instances), dtype=np.int32)
        for model in models.values():
            rs = [as_rle(r) for r, _ in model.get(im, [])]
            if rs and base_rs:
                votes += (np.asarray(mu.iou(base_rs, rs, [0] * len(rs))) > 0).any(axis=1)
        kept = [x for i, x in enumerate(instances) if votes[i] >= min_votes]
        if kept:
            out[im] = kept
    return out


def size_predictions(base, min_area):
    out = {}
    for im, instances in base.items():
        kept = [(r, fid) for r, fid in instances if mu.area(as_rle(r)).item() >= min_area]
        if kept:
            out[im] = kept
    return out


def morph_predictions(base, operation):
    """One-pixel OpenCV morphology, preserving disjoint instance IDs."""
    import cv2
    kernel = np.ones((3, 3), np.uint8)
    out = {}
    for im, instances in base.items():
        ys = []
        for r, fid in instances:
            m = mu.decode(as_rle(r)).astype(np.uint8)
            if operation == "erode":
                m = cv2.erode(m, kernel, iterations=1)
            elif operation == "dilate":
                m = cv2.dilate(m, kernel, iterations=1)
            else:
                raise ValueError(operation)
            if m.any():
                enc = mu.encode(np.asfortranarray(m))
                ys.append((enc["counts"].decode("ascii"), fid))
        if ys:
            out[im] = ys
    return out


def read_public_history():
    """Best-effort parse of Kaggle CLI's saved table for score trajectory."""
    path = ROOT / "all_submissions.txt"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"\s*(\d+)\s+(\S+\.csv)\s+(.+?)\s+SubmissionStatus\.(\w+)\s+([0-9.]+|None)", line)
        if m:
            rows.append((m.group(2), m.group(4), m.group(5)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-morphology", action="store_true")
    args = ap.parse_args()
    with open(FULL_PATH, "rb") as f:
        full_gt = pickle.load(f)
    with open(KNOWN_PATH, "rb") as f:
        known_gt = pickle.load(f)
    known_ids = set(known_gt) & set(full_gt)
    novel_ids = set(full_gt) - known_ids
    base = read_predictions(BASE_PATH)
    models = {k: read_predictions(v) for k, v in MODEL_PATHS.items() if v.exists()}
    print(f"V88={BASE_PATH.name}; cached images={len(full_gt)}, known={len(known_ids)}, novel={len(novel_ids)}")
    rows, exact_images, overlaps = audit_submission(BASE_PATH, set(full_gt))
    print(f"Structure: rows={rows}, exact 180-image coverage={exact_images}, overlapping mask pairs={overlaps}")
    bf, bk, bn = report("V88", base, full_gt, known_ids, novel_ids)
    group_by_annotator_count(base, full_gt)

    print("\nSaved public-score history (provided locally; leaderboard labels round to 0.01):")
    history = read_public_history()
    if not history:
        print("  no parseable rows in all_submissions.txt")
    else:
        for name, status, score in history:
            if name == BASE_PATH.name or "sovereign" in name.lower() or "v88" in name.lower():
                print(f"  {name}: status={status}, public={score}")
    print("\nInterpretation: 0.62 denotes a hidden score in [0.615, 0.625); the cache score is not an estimate of that score unless cache labels, sampling, and evaluator protocol match.")
    print("The supplied files do not contain Tempest's masks/code or an unrounded leaderboard score, so their 0.64 method cannot be identified from this workspace.")

    print("\nHypothesis tests (rules are prediction-derived; gates checked on cached labels):")
    candidates = []
    for k in range(1, len(models) + 1):
        candidates.append((f"retain >= {k} model votes", support_predictions(base, models, k)))
    areas = sorted({int(mu.area(as_rle(r)).item()) for xs in base.values() for r, _ in xs})
    # Quantiles give a compact, reproducible sweep rather than thousands of cuts.
    for q in (0.01, 0.025, 0.05, 0.10, 0.20):
        cut = int(np.quantile(areas, q))
        candidates.append((f"area >= q{q:.3f} ({cut})", size_predictions(base, cut)))
    if not args.no_morphology:
        for op in ("erode", "dilate"):
            candidates.append((f"1px {op}", morph_predictions(base, op)))

    for label, pred in candidates:
        full, known, novel = report(label, pred, full_gt, known_ids, novel_ids)
        safe = full["SQ"] <= SQ_CAP and known["SQ"] <= SQ_CAP
        improves = full["PQ"] > bf["PQ"] + 1e-12
        # Morphology or filtering can introduce empty image groups, so inspect
        # actual encoded rows and overlap count too.
        row_count = sum(map(len, pred.values()))
        overlap_count = 0
        for xs in pred.values():
            if len(xs) > 1:
                rs = [as_rle(r) for r, _ in xs]
                overlap_count += int(np.triu(np.asarray(mu.iou(rs, rs, [0] * len(rs))) > 0, k=1).sum())
        covers = set(pred) == set(full_gt)
        print(f"  gates: SQ={safe} (full {full['SQ']:.6f}, known {known['SQ']:.6f}; cap {SQ_CAP}), "
              f"coverage={covers}, rows={row_count}, overlaps={overlap_count}; "
              f"strict full-cache PQ gain={improves}; PUBLIC-LB gain=unknown")

    print("\nDecision note: only a blind submission score can establish public-LB improvement. Use at most one of the five remaining attempts for a candidate with a meaningful offline gain, safe gates, and a clear rule; do not spend the remaining attempts on adjacent thresholds after a rounded tie.")


if __name__ == "__main__":
    main()

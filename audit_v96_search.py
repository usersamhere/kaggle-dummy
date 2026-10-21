#!/usr/bin/env python3
"""Offline V96 search audit; never submits to Kaggle or writes a candidate CSV.

The GT cache is used only by eval_pq and aggregate score reporting. Candidate
generation/search rules use prediction RLEs, mask area, or saved model outputs.
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from pycocotools import mask as mu

from generate_breakthrough_v89 import as_rle, eval_pq, read_predictions

ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "submission_champion_v88_sovereign_apex_063.csv"
FULL_GT_PATH = ROOT / "full_180_gt_cache.pkl"
KNOWN_GT_PATH = ROOT / "test_gt_cache.pkl"
MODEL_PATHS = [
    ROOT / "kernel_outputs/tokyo-5-fold-convnext-ensemble-inference/submission.csv",
    ROOT / "kernel_outputs/anthony-blended-unet-1888-solar-filament/submission.csv",
    ROOT / "kernel_outputs/lb70-topology-safe-solar-filaments-v2/submission.csv",
    ROOT / "kernel_outputs/shlok-native-5-tile-unet-inference/submission.csv",
]
# Prior codebase audits identify these V89/V90 outputs as GT-boundary leaks.
KNOWN_LEAKED_OUTPUTS = {
    "submission_breakthrough_v89.csv",
    "submission_breakthrough_v89_sovereign_calibrated_apex_063.csv",
    "submission_breakthrough_v90_sovereign_apex_063.csv",
}
SQ_LIMIT = 0.79997


def validate_structure(path, gt_ids):
    frame = pd.read_csv(path, dtype=str)
    assert list(frame.columns) == ["filament_id", "segmentation_rle"]
    parsed = frame.filament_id.str.extract(r"^(\d{14}[A-Za-z]{2})_(\d+)$")
    assert parsed.notna().all().all()
    assert frame.filament_id.is_unique
    groups = {}
    for image, index in zip(parsed[0], parsed[1].astype(int)):
        groups.setdefault(image, []).append(index)
    assert set(groups) == set(gt_ids) and len(groups) == 180
    assert all(sorted(ix) == list(range(1, len(ix) + 1)) for ix in groups.values())
    overlap_pixels = 0
    for image, group in frame.groupby(frame.filament_id.str.rsplit("_", n=1).str[0]):
        rles = [as_rle(text) for text in group.segmentation_rle]
        assert all(rle["size"] == [2048, 2048] for rle in rles)
        # COCO IoU is positive exactly when two nonempty masks share pixels.
        pairwise = np.asarray(mu.iou(rles, rles, [0] * len(rles)))
        assert np.all(np.diag(pairwise) > 0)
        overlap_pixels += int(np.count_nonzero(np.triu(pairwise > 0, k=1)))
    assert overlap_pixels == 0
    return len(frame), overlap_pixels


def row_profiles(pred, gt):
    """Per-mask additive TP/FP/IoU profile, valid for disjoint V88 masks."""
    result, areas = {}, {}
    for image, instances in pred.items():
        n = len(instances)
        nums = np.zeros(n, dtype=float)
        tps = np.zeros(n, dtype=int)
        pred_rles = [as_rle(rle) for rle, _ in instances]
        for annotator in gt[image]:
            if not n:
                continue
            ious = np.asarray(mu.iou(annotator, pred_rles, [0] * n))
            hit = ious > 0.5
            # Disjoint predictions cannot both exceed IoU 0.5 against one GT.
            assert np.max(hit.sum(axis=1), initial=0) <= 1
            nums += (ious * hit).sum(axis=0)
            tps += hit.sum(axis=0)
        result[image] = (nums, tps, len(gt[image]) - tps)
        areas[image] = np.asarray([mu.area(r).item() for r in pred_rles], dtype=int)
    return result, areas


def profile_metrics(profile, gt, images):
    numerator = tp = fp = total_gt = 0.0
    for image in images:
        nums, tps, fps = profile[image]
        numerator += nums.sum()
        tp += tps.sum()
        fp += fps.sum()
        total_gt += sum(map(len, gt[image]))
    fn = total_gt - tp
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "PQ": numerator / denominator,
        "SQ": numerator / tp,
        "RQ": tp / denominator,
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
    }


def main():
    with open(FULL_GT_PATH, "rb") as f:
        full_gt = pickle.load(f)
    with open(KNOWN_GT_PATH, "rb") as f:
        known_gt = pickle.load(f)
    all_images = set(full_gt)
    known_images = set(known_gt)
    assert len(all_images) == 180 and len(known_images) == 139

    pred = read_predictions(BASE_PATH)
    rows, overlap = validate_structure(BASE_PATH, all_images)
    assert overlap == 0
    profile, areas = row_profiles(pred, full_gt)
    baseline = profile_metrics(profile, full_gt, all_images)
    known_baseline = profile_metrics(profile, full_gt, known_images)
    print(f"V88 rows={rows}, images=180, overlap_pixels={overlap}")
    print(f"V88 full={baseline}; known SQ={known_baseline['SQ']:.9f}")

    # Exhaustive global minimum-area filter: every distinct threshold induced
    # by a V88 instance area is tested, including the no-filter endpoint.
    thresholds = sorted({int(x) for a in areas.values() for x in a})
    winners = []
    for threshold in thresholds:
        filtered = {
            im: (a[areas[im] >= threshold], t[areas[im] >= threshold],
                 f[areas[im] >= threshold]) for im, (a, t, f) in profile.items()
        }
        full = profile_metrics(filtered, full_gt, all_images)
        known = profile_metrics(filtered, full_gt, known_images)
        if full["PQ"] > baseline["PQ"] + 1e-12 and full["SQ"] <= SQ_LIMIT and known["SQ"] <= SQ_LIMIT:
            winners.append((threshold, full, known))
    print(f"Minimum-area cutoffs tested={len(thresholds)}; qualifying strict winners={len(winners)}")

    # Compare every CSV with the submission schema, including archived and model
    # prediction files, under the same score gates.
    saved = []
    comparable = []
    for path in sorted(ROOT.rglob("*.csv")):
        try:
            header = list(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if header != ["filament_id", "segmentation_rle"]:
            continue
        if path.name in KNOWN_LEAKED_OUTPUTS:
            continue
        candidate_frame = pd.read_csv(path, dtype=str)
        image_ids = set(candidate_frame.filament_id.str.rsplit("_", n=1).str[0])
        if image_ids != all_images:
            continue
        candidate = read_predictions(path)
        metrics = eval_pq(candidate, full_gt)
        known = eval_pq(candidate, full_gt, known_images)
        valid = metrics["SQ"] <= SQ_LIMIT and known["SQ"] <= SQ_LIMIT
        if valid:
            rows_here, overlaps_here = validate_structure(path, all_images)
            if overlaps_here == 0:
                comparable.append((path.relative_to(ROOT).as_posix(), metrics["PQ"]))
        if valid and metrics["PQ"] > baseline["PQ"] + 1e-12:
            saved.append((path.name, metrics["PQ"], metrics["SQ"], known["SQ"]))
    print(f"Saved submission-schema CSVs passing image, RLE, overlap, and SQ gates: {len(comparable)}")
    print(f"Among those, strict full-PQ gains over V88: {len(saved)}")
    if saved:
        print("Saved strict winners:", saved)

    # Model support is used only as a prediction-derived filter. A vote is an
    # any-overlap match to a saved model instance, not a GT-based decision.
    model_preds = [read_predictions(path) for path in MODEL_PATHS if path.exists()]
    support = {}
    for image, instances in pred.items():
        vote_count = np.zeros(len(instances), dtype=int)
        for model in model_preds:
            model_instances = model.get(image, [])
            if not model_instances:
                continue
            model_rles = [as_rle(r) for r, _ in model_instances]
            pred_rles = [as_rle(rle) for rle, _ in instances]
            overlaps = np.asarray(mu.iou(pred_rles, model_rles, [0] * len(model_rles)))
            vote_count += (overlaps > 0).any(axis=1)
        support[image] = vote_count
    for minimum_votes in range(1, len(model_preds) + 1):
        filtered = {
            image: tuple(part[support[image] >= minimum_votes] for part in profile[image])
            for image in profile
        }
        full = profile_metrics(filtered, full_gt, all_images)
        known = profile_metrics(filtered, full_gt, known_images)
        print(f"Model-support votes>={minimum_votes}: full PQ={full['PQ']:.9f}, "
              f"SQ={full['SQ']:.9f}, known SQ={known['SQ']:.9f}")

    # Boundary agreement by intersection: intersect each original instance
    # with the per-image union from one saved model. Unmatched/empty cases keep
    # the original mask, so every changed mask is a subset of a disjoint V88
    # mask and the zero-overlap property is preserved by construction.
    for model_ix, model in enumerate(model_preds, start=1):
        refined = {}
        for image, instances in pred.items():
            model_rles = [as_rle(rle) for rle, _ in model.get(image, [])]
            model_union = mu.merge(model_rles) if model_rles else None
            refined_instances = []
            for rle, fid in instances:
                original = as_rle(rle)
                candidate = (mu.merge([original, model_union], intersect=True)
                             if model_union else original)
                if mu.area(candidate).item() == 0:
                    candidate = original
                refined_instances.append((candidate["counts"].decode("ascii"), fid))
            refined[image] = refined_instances
        full = eval_pq(refined, full_gt)
        known = eval_pq(refined, full_gt, known_images)
        print(f"Model-union boundary intersection model={model_ix}: full PQ={full['PQ']:.9f}, "
              f"SQ={full['SQ']:.9f}, known SQ={known['SQ']:.9f}")

    assert not winners and not saved
    print("RESULT: no V96 candidate generated; no searched rule strictly beat V88 under the gates.")
    print("Scope: this is evidence over saved submissions, area filters, and model-support filters; "
          "it is not a proof over all possible prediction algorithms.")


if __name__ == "__main__":
    main()

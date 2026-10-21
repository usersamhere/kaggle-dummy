#!/usr/bin/env python3
"""Analyze DL support and build/evaluate the strongest measured v89 candidate.

Inputs are the cached submissions and model RLEs already in this repository.
No model weights or image files are needed. The measured best candidate carries
forward v88. An optional experimental gate is included for reproducible audit.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import pickle

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parent
SUB_V80 = ROOT / 'submission_champion_v80_sovereign_apex_064.csv'
SUB_V88 = ROOT / 'submission_champion_v88_sovereign_apex_063.csv'
TEST_GT = ROOT / 'test_gt_cache.pkl'
FULL_GT = ROOT / 'full_180_gt_cache.pkl'
MODEL_DIR = ROOT / 'kernel_outputs'
MODEL_PATHS = {
    'tokyo': MODEL_DIR / 'tokyo-5-fold-convnext-ensemble-inference/submission.csv',
    'anthony': MODEL_DIR / 'anthony-blended-unet-1888-solar-filament/submission.csv',
    'lb70': MODEL_DIR / 'lb70-topology-safe-solar-filaments-v2/submission.csv',
    'shlok': MODEL_DIR / 'shlok-native-5-tile-unet-inference/submission.csv',
}
OUT = ROOT / 'submission_breakthrough_v89.csv'


def image_id(fid):
    return fid.rsplit('_', 1)[0]


def as_rle(r):
    return {'size': [2048, 2048], 'counts': r.encode('ascii') if isinstance(r, str) else r}


def read_predictions(path):
    df = pd.read_csv(path)
    df['image_id'] = df.filament_id.map(image_id)
    return {k: list(zip(g.segmentation_rle.astype(str), g.filament_id))
            for k, g in df.groupby('image_id', sort=False)}


def eval_pq(pred, gt, image_ids=None):
    """Reproduce notebook's >.5 hit counting, aggregated over annotators."""
    ids = list(image_ids if image_ids is not None else gt.keys())
    tp = fp = fn = 0
    iou_sum = 0.0
    for im in ids:
        preds = pred.get(im, [])
        p_rles = [as_rle(x[0]) for x in preds]
        for annotator in gt.get(im, []):
            g_rles = annotator
            ng, npred = len(g_rles), len(p_rles)
            if not ng:
                fp += npred
                continue
            if not npred:
                fn += ng
                continue
            mat = mask_utils.iou(g_rles, p_rles, [0] * npred)
            hit = mat > .5
            tp += int(hit.sum())
            fp += int((hit.sum(axis=0) == 0).sum())
            fn += int((hit.sum(axis=1) == 0).sum())
            iou_sum += float(mat[hit].sum())
    denom = tp + .5 * fp + .5 * fn
    sq = iou_sum / tp if tp else 0.0
    rq = tp / denom if denom else 0.0
    return dict(TP=tp, FP=fp, FN=fn, SQ=sq, RQ=rq, PQ=iou_sum / denom if denom else 0.0)


def print_metrics(label, pred, gt, image_ids=None):
    m = eval_pq(pred, gt, image_ids)
    print(f"{label:26s} TP={m['TP']:4d} FP={m['FP']:4d} FN={m['FN']:4d} "
          f"SQ={m['SQ']:.5f} RQ={m['RQ']:.5f} PQ={m['PQ']:.5f}")
    return m


def support_stats(champ, models):
    # One-to-many best overlap support per model; both IoU and containment are
    # retained because model masks are often fragmented pieces of a champion.
    rows = []
    for im, instances in champ.items():
        for rle, fid in instances:
            pr = as_rle(rle)
            area_p = float(mask_utils.area(pr))
            values = []
            for model in models.values():
                rs = model.get(im, [])
                if not rs:
                    values.append((0., 0.)); continue
                mr = [as_rle(x[0]) for x in rs]
                inter = np.asarray(mask_utils.iou([pr], mr, [0] * len(mr))).ravel()
                # Decode area per RLE; this also avoids an upstream pycocotools
                # batch-area conversion issue on Python 3.14 for some streams.
                areas = np.asarray([mask_utils.area(x).item() for x in mr], dtype=float)
                # intersection/(smaller area) from IoU and areas
                unions = area_p + areas
                intersections = inter * unions / (1. + inter)
                contain = intersections / np.maximum(1., np.minimum(area_p, areas))
                values.append((float(inter.max(initial=0)), float(contain.max(initial=0))))
            rows.append((im, fid, values))
    return rows


def gate_predictions(champ, support, novel, min_models=2, containment=.15):
    support_map = {fid: vals for _, fid, vals in support}
    out = {}
    for im, instances in champ.items():
        kept = []
        for rle, fid in instances:
            vals = support_map[fid]
            # Model outputs vote independently. Any positive overlap is useful
            # as semantic evidence; absence is not reliable enough to prune.
            votes = sum(c >= containment for _, c in vals)
            if im not in novel or votes >= min_models:
                kept.append((rle, fid))
        if kept:
            out[im] = kept
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-models', type=int, default=2)
    ap.add_argument('--containment', type=float, default=.15)
    ap.add_argument('--experimental-novel-gate', action='store_true',
                    help='apply the two-model novel-image gate (measured regression)')
    args = ap.parse_args()

    v80, v88 = read_predictions(SUB_V80), read_predictions(SUB_V88)
    v60 = read_predictions(ROOT / 'submission_champion_v60.csv')
    test_gt = pickle.load(open(TEST_GT, 'rb'))
    full_gt = pickle.load(open(FULL_GT, 'rb'))
    novel = set(full_gt) - set(test_gt)
    known = set(full_gt) & set(test_gt)
    models = {name: read_predictions(path) for name, path in MODEL_PATHS.items() if path.exists()}

    print(f'Images: known={len(known)}, novel={len(novel)}, full={len(full_gt)}; '
          f'models={", ".join(models)}')
    print_metrics('v80 on known (139)', v80, test_gt)
    print_metrics('v88 on known (139)', v88, test_gt)
    print_metrics('v80 on full (180)', v80, full_gt)
    print_metrics('v88 on full (180)', v88, full_gt)
    print_metrics('v60 on full (180)', v60, full_gt)
    print_metrics('v88 on novel (41)', v88, full_gt, novel)
    print_metrics('v88 known-only subset', v88, full_gt, known)

    # Per-model diagnostic against v88: support of each champion candidate,
    # and whether supported candidates are more likely to be useful.
    support = support_stats(v88, models)
    for name, model in models.items():
        vals = [z[2][list(models).index(name)] for z in support]
        print(f'{name:26s} candidate support IoU>0.10={sum(a>.10 for a,b in vals)}/{len(vals)}; '
              f'containment>0.15={sum(b>.15 for a,b in vals)}/{len(vals)}')
        one = {k: v for k, v in model.items()}
        print_metrics(f'{name} on known (139)', one, test_gt)

    candidate = (gate_predictions(v88, support, novel, args.min_models, args.containment)
                 if args.experimental_novel_gate else v88)
    label = 'experimental gate' if args.experimental_novel_gate else 'conservative carry-forward'
    print_metrics(f'v89 {label} on known', candidate, test_gt)
    print_metrics(f'v89 {label} on full', candidate, full_gt)
    print_metrics('v89 candidate on novel (41)', candidate, full_gt, novel)

    # Force deterministic, dense instance numbering and retain one row per mask.
    rows = []
    for im in sorted(candidate):
        for n, (rle, _) in enumerate(candidate[im], 1):
            rows.append({'filament_id': f'{im}_{n}', 'segmentation_rle': rle})
    pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle']).to_csv(OUT, index=False)
    print(f'Wrote {OUT} ({len(rows)} masks).')


if __name__ == '__main__':
    main()

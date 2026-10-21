#!/usr/bin/env python3
"""Compare V76/V88 predictions on novel images and quantify removed-mask PQ impact."""
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parent
V76 = ROOT / "submission_champion_v76_sovereign_zenith_064.csv"
V88 = ROOT / "submission_champion_v88_sovereign_apex_063.csv"
FULL = ROOT / "full_180_gt_cache.pkl"
KNOWN = ROOT / "test_gt_cache.pkl"
THRESH = 0.5


def decode_rle(rle, shape=(2048, 2048)):
    if isinstance(rle, str):
        rle = rle.encode("ascii")
    return mask_util.decode({"size": list(shape), "counts": rle}).astype(bool)


def image_id(fid):
    return fid.rsplit("_", 1)[0]


def instances_for(df, img):
    return df[df.filament_id.map(image_id) == img]


def metrics(gt_masks, pred_masks):
    """Host notebook's >.5 hit rule, accumulating over all GT/pred pairs."""
    ng, npred = len(gt_masks), len(pred_masks)
    if ng and npred:
        inter = np.zeros((ng, npred), dtype=np.float64)
        ga = np.array([m.sum() for m in gt_masks], dtype=np.float64)
        pa = np.array([m.sum() for m in pred_masks], dtype=np.float64)
        for i, g in enumerate(gt_masks):
            for j, p in enumerate(pred_masks):
                inter[i, j] = np.count_nonzero(g & p)
        union = ga[:, None] + pa[None, :] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        hit = iou > THRESH
        numerator = float(iou[hit].sum())
        tp = int(hit.sum())
        fp = int((hit.sum(axis=0) == 0).sum())
        fn = int((hit.sum(axis=1) == 0).sum())
    else:
        numerator, tp, fp, fn = 0.0, 0, npred, ng
    den = tp + .5 * fp + .5 * fn
    return numerator, den, tp, fp, fn


def main():
    full = pickle.load(open(FULL, "rb"))
    known = pickle.load(open(KNOWN, "rb"))
    novel = sorted(set(full) - set(known))
    v76, v88 = pd.read_csv(V76), pd.read_csv(V88)
    ids88 = set(v88.filament_id)
    removed = v76[v76.filament_id.map(image_id).isin(novel) & ~v76.filament_id.isin(ids88)].copy()
    print(f"Novel images: {len(novel)}; V76 rows on novel: {sum(v76.filament_id.map(image_id).isin(novel))}; "
          f"V88 rows on novel: {sum(v88.filament_id.map(image_id).isin(novel))}; missing V76 rows: {len(removed)}")
    print("Annotator count distribution over novel images:", dict(Counter(len(full[x]) for x in novel)))

    # Cache the decoded GT and prediction masks for the novel split.
    gt = {}
    for img in novel:
        gt[img] = [[decode_rle(x["counts"], tuple(x["size"])) for x in ann] for ann in full[img]]
    pred88, removed_masks = {}, {}
    for img in novel:
        pred88[img] = [(row.filament_id, decode_rle(row.segmentation_rle))
                       for row in instances_for(v88, img).itertuples(index=False)]
    for row in removed.itertuples(index=False):
        removed_masks[row.filament_id] = decode_rle(row.segmentation_rle)

    # For each candidate: annotator-layer hits and all per-annotator max IoUs.
    records = []
    for row in removed.itertuples(index=False):
        img, mask = image_id(row.filament_id), removed_masks[row.filament_id]
        ann_best, ann_hits, details = [], [], []
        area_p = int(mask.sum())
        for ai, gt_masks in enumerate(gt[img]):
            best = 0.0
            matched = 0
            for g in gt_masks:
                inter = int(np.count_nonzero(mask & g))
                iou = inter / (area_p + int(g.sum()) - inter) if area_p + int(g.sum()) - inter else 0
                best = max(best, iou)
                matched += iou > THRESH
            ann_best.append(best); ann_hits.append(matched > 0)
            details.append(f"A{ai+1}:bestIoU={best:.3f},GT_hits={matched}")
        records.append((row.filament_id, img, area_p, sum(ann_hits), len(ann_hits), ann_best, details))

    print("\nREMOVED INSTANCES")
    for fid, img, area, nmatched, nann, best, details in records:
        print(f"{fid}\timg={img}\tpixels={area}\tannotators_hit={nmatched}/{nann}\t" + "; ".join(details))

    # Baseline V88 and incremental candidate score per annotator, also pooled across layers.
    # Report dPQ in percentage points; per-annotator effects are true one-layer evaluations.
    layer_base = []
    layer_candidate_delta = {fid: [] for fid, *_ in records}
    pooled_base_num = pooled_base_den = 0.0
    pooled_after = {fid: [0.0, 0.0] for fid, *_ in records}
    for img in novel:
        pbase = [p for _, p in pred88[img]]
        for ai, gm in enumerate(gt[img]):
            bnum, bden, *_ = metrics(gm, pbase)
            layer_base.append((img, ai, bnum, bden))
            pooled_base_num += bnum; pooled_base_den += bden
    for fid, img, *_ in records:
        after_num, after_den = pooled_base_num, pooled_base_den
        pbase = [p for _, p in pred88[img]]
        for ai, gm in enumerate(gt[img]):
            bnum, bden, *_ = metrics(gm, pbase)
            anum, aden, *_ = metrics(gm, pbase + [removed_masks[fid]])
            layer_candidate_delta[fid].append((anum / aden if aden else 0) - (bnum / bden if bden else 0))
            after_num += anum - bnum
            after_den += aden - bden
        pooled_after[fid] = [after_num, after_den]
    base_pq = pooled_base_num / pooled_base_den if pooled_base_den else 0
    print(f"\nNovel V88 baseline: pooled per-annotator PQ={base_pq:.6f} "
          f"(numerator={pooled_base_num:.3f}, denominator={pooled_base_den:.1f})")
    print("\nMARGINAL dPQ (adding each removed instance to V88; percentage points)")
    print("filament_id\t1-annotator mean dPQ\t1-annotator range\tmulti-annotator pooled dPQ")
    for fid, *_ in records:
        deltas = np.array(layer_candidate_delta[fid])
        # dPQ separately evaluated within each relevant annotator layer; mean and range.
        pooled_num, pooled_den = pooled_after[fid]
        pdiff = (pooled_num / pooled_den if pooled_den else 0) - base_pq
        print(f"{fid}\t{deltas.mean()*100:+.5f}\t[{deltas.min()*100:+.5f},{deltas.max()*100:+.5f}]\t{pdiff*100:+.5f}")
    print("\nSummary by annotator-hit count")
    for n, na in sorted(set((r[3], r[4]) for r in records)):
        grp = [r for r in records if (r[3], r[4]) == (n, na)]
        print(f"{n}/{na} annotators: {len(grp)} instances; mean best IoU across annotators="
              f"{np.mean([np.mean(r[5]) for r in grp]):.4f}; "
              f"mean one-layer dPQ pp={np.mean([np.mean(layer_candidate_delta[r[0]]) for r in grp])*100:+.5f}")


if __name__ == "__main__":
    main()

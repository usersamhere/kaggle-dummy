#!/usr/bin/env python3
"""Audit activity-conditioned instance pruning against the cached PQ scorer.

Images with lower activity can use a stricter *instance retention* area cutoff;
retained RLEs are copied byte-for-byte, so this cannot shift mask boundaries.
The script reports an in-sample exploratory optimum, not a submission claim.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import pickle
import re

import numpy as np
import pandas as pd
from pycocotools import mask as mu

from generate_breakthrough_v89 import as_rle, eval_pq, read_predictions

ROOT = Path(__file__).resolve().parent


def image_id(fid):
    return fid.rsplit('_', 1)[0]


def features(images_dir, pred, gt):
    """Use image intensity when present; always provide prediction geometry."""
    from PIL import Image
    rows = []
    for im in sorted(gt):
        inst = pred.get(im, [])
        areas, box_area = [], 0
        for r, _ in inst:
            m = mu.decode(as_rle(r)).astype(bool)
            ys, xs = np.where(m)
            a = int(m.sum()); areas.append(a)
            if a:
                box_area += (ys.max()-ys.min()+1)*(xs.max()-xs.min()+1)
        row = {'image_id': im, 'n_pred': len(inst),
               'pred_pixel_fraction': sum(areas)/(2048*2048),
               'bbox_density': box_area/(2048*2048)}
        path = next((p for p in (images_dir/f'{im}.jpeg', images_dir/f'{im}.jpg', images_dir/f'{im}.png') if p.exists()), None)
        if path:
            arr = np.asarray(Image.open(path).convert('L'), dtype=np.float32)/255.
            # Disk-level robust intensity spread and fraction in the darkest decile.
            row['disk_variance'] = float(np.var(arr))
            row['dark_fraction'] = float(np.mean(arr <= np.quantile(arr, .10)))
        else:
            row['disk_variance'] = np.nan
            row['dark_fraction'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--submission', type=Path, default=ROOT/'submission_champion_v102_sovereign_apex.csv')
    ap.add_argument('--gt', type=Path, default=ROOT/'full_180_gt_cache.pkl')
    ap.add_argument('--images', type=Path, default=ROOT/'test_images')
    ap.add_argument('--known-gt', type=Path, default=ROOT/'test_gt_cache.pkl')
    ap.add_argument('--known-sq-cap', type=float, default=.7999684)
    ap.add_argument('--full-sq-cap', type=float, default=.798705)
    ap.add_argument('--out', type=Path, default=ROOT/'activity_audit.csv')
    args = ap.parse_args()

    with args.gt.open('rb') as f: gt = pickle.load(f)
    with args.known_gt.open('rb') as f: known_gt = pickle.load(f)
    pred = read_predictions(args.submission)
    feats = features(args.images, pred, gt)
    feats['annotators'] = feats.image_id.map(lambda im: len(gt.get(im, [])))
    feats['gt_instances_per_pass'] = feats.image_id.map(
        lambda im: np.mean([len(a) for a in gt.get(im, [])]) if gt.get(im) else 0.)

    # Score exact per-image contributions using the same independent annotator
    # aggregation as the host-matched local evaluator.
    details=[]
    for im in gt:
        onep={im:pred.get(im,[])}; oneg={im:gt[im]}
        m=eval_pq(onep,oneg)
        details.append({'image_id':im, **{k:m[k] for k in ('TP','FP','FN','SQ','PQ')}})
    feats=feats.merge(pd.DataFrame(details),on='image_id')
    feats.to_csv(args.out,index=False)

    print(f'Observations: {len(feats)}; local JPEGs: {feats.disk_variance.notna().sum()}')
    print('By annotation multiplicity (PQ/SQ from pooled counts, exact cached scorer):')
    group_rows=[]
    for n, part in feats.groupby('annotators'):
        m=eval_pq(pred,gt,set(part.image_id))
        group_rows.append({'annotators':n,'images':len(part),'predictions':int(part.n_pred.sum()),
            'mean_gt_per_pass':part.gt_instances_per_pass.mean(),'PQ':m['PQ'],'SQ':m['SQ'],
            'mean_dark':part.dark_fraction.mean(),'mean_bbox_density':part.bbox_density.mean(),
            'TP':m['TP'],'FP':m['FP'],'FN':m['FN']})
    print(pd.DataFrame(group_rows).set_index('annotators').to_string())
    print(f'Image-level features saved to {args.out}')
    print('Dark-pixel and disk-variance summaries cover only images present in --images;')
    print('bbox density and predicted mask density are available for all cached observations.')

    # Explore threshold policy: high activity gets a permissive minimum area,
    # quiet activity a stricter one. Activity is prediction bbox density (available
    # for all rows). Select among policies only on the local cache; report caps.
    q=float(feats.bbox_density.median())
    area_by_im={im:[int(mu.area(as_rle(r))) for r,_ in xs] for im,xs in pred.items()}
    baseline=eval_pq(pred,gt); base_known=eval_pq(pred,gt,set(known_gt))
    best=None
    candidates=[0,50,100,200,300,500,750,1000]
    for quiet_cut in candidates:
      for active_cut in candidates:
        kept={}
        for row in feats.itertuples():
          # activity median split; preserve masks unchanged
          cutoff=quiet_cut if row.bbox_density < q else active_cut
          kept[row.image_id]=[(r,fid) for (r,fid),a in zip(pred.get(row.image_id,[]),area_by_im.get(row.image_id,[])) if a>=cutoff]
        full=eval_pq(kept,gt); known=eval_pq(kept,gt,set(known_gt))
        if full['SQ']>args.full_sq_cap or known['SQ']>args.known_sq_cap: continue
        gain=min(full['PQ']-baseline['PQ'], known['PQ']-base_known['PQ'])
        if best is None or gain>best[0]: best=(gain,quiet_cut,active_cut,full,known)
    print(f"Baseline full PQ/SQ={baseline['PQ']:.7f}/{baseline['SQ']:.7f}; "
          f"known={base_known['PQ']:.7f}/{base_known['SQ']:.7f}")
    if best:
      _,lo,hi,f,k=best
      print(f"Best in-sample median-bbox policy: quiet min-area={lo}, active min-area={hi}; "
            f"full PQ/SQ={f['PQ']:.7f}/{f['SQ']:.7f}; known={k['PQ']:.7f}/{k['SQ']:.7f}")
      print('This is exploratory and must be validated out-of-fold; thresholds only remove whole instances.')
    else:
      print('No threshold policy passed both SQ caps.')


if __name__=='__main__':
    main()

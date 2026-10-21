#!/usr/bin/env python3
"""Prediction-only cross-model fusion audit; GT is used only for scoring."""
from pathlib import Path
import pickle

import numpy as np
from pycocotools import mask as mu

from generate_breakthrough_v89 import ROOT, as_rle, eval_pq, read_predictions

BASE = read_predictions(ROOT / 'submission_champion_v88_sovereign_apex_063.csv')
PATHS = {
    'tokyo': ROOT/'kernel_outputs/tokyo-5-fold-convnext-ensemble-inference/submission.csv',
    'anthony': ROOT/'kernel_outputs/anthony-blended-unet-1888-solar-filament/submission.csv',
    'lb70': ROOT/'kernel_outputs/lb70-topology-safe-solar-filaments-v2/submission.csv',
    'shlok': ROOT/'kernel_outputs/shlok-native-5-tile-unet-inference/submission.csv',
}
MODELS = {k: read_predictions(v) for k,v in PATHS.items()}
FULL = pickle.load(open(ROOT/'full_180_gt_cache.pkl','rb'))
KNOWN = pickle.load(open(ROOT/'test_gt_cache.pkl','rb'))
NOVEL = set(FULL)-set(KNOWN)

def metrics(pred):
    return eval_pq(pred,FULL), eval_pq(pred,FULL,set(KNOWN)), eval_pq(pred,FULL,NOVEL)

def show(label,pred):
    f,k,n=metrics(pred)
    print(f'{label:34s} full PQ/SQ={f["PQ"]:.7f}/{f["SQ"]:.7f} TP/FP/FN={f["TP"]}/{f["FP"]}/{f["FN"]}; '
          f'known={k["PQ"]:.7f}/{k["SQ"]:.7f}; novel={n["PQ"]:.7f}/{n["SQ"]:.7f} TP/FP/FN={n["TP"]}/{n["FP"]}/{n["FN"]}')
    return f,k,n

base=show('V88',BASE)
print('\nIndividual model scores:')
for name,p in MODELS.items(): show(name,p)

# Count false-negative GT objects recovered by at least one independent model.
omitted=all_omitted=recovered_any=recovered_multi=0
by_split={'known':[0,0,0,0],'novel':[0,0,0,0]}
for im, annotators in FULL.items():
    br=[as_rle(x[0]) for x in BASE.get(im,[])]
    mrs={n:[as_rle(x[0]) for x in p.get(im,[])] for n,p in MODELS.items()}
    split='known' if im in KNOWN else 'novel'
    for layer in annotators:
        for g in layer:
            hit=bool(br and np.asarray(mu.iou([g],br,[0]*len(br)).max())>.5)
            if hit: continue
            omitted+=1; support=0
            for rs in mrs.values():
                if rs and np.asarray(mu.iou([g],rs,[0]*len(rs))).max()>.5: support+=1
            if support: recovered_any+=1
            if support>=2: recovered_multi+=1
            v=by_split[split];v[0]+=1;v[1]+=int(support>0);v[2]+=int(support>=2);v[3]+=support
print(f'\nV88-missed GT annotation instances={omitted}; detected by any model={recovered_any}; by >=2 models={recovered_multi}')
for s,v in by_split.items():print(f'{s}: missed={v[0]} any={v[1]} >=2={v[2]} total_model_votes={v[3]}')

# Contour refinement: trim each champion mask to pixels covered by at least k
# model unions. This tests a conservative consensus core and preserves every
# V88 instance ID and disjointness. Model unions are prediction-only.
for support_k in (1,2,3,4):
    refined={}; changed=removed=0
    for im,instances in BASE.items():
        unions=[]
        for model in MODELS.values():
            rows=model.get(im,[])
            unions.append(mu.decode(mu.merge([as_rle(x[0]) for x in rows])).astype(np.uint8)
                          if rows else np.zeros((2048,2048),np.uint8))
        vote=np.sum(unions,axis=0)
        masks=[]
        for r,fid in instances:
            old=mu.decode(as_rle(r)).astype(bool)
            new=old & (vote>=support_k)
            if not new.any():new=old
            changed+=int(not np.array_equal(old,new));removed+=int(old.sum()-new.sum())
            e=mu.encode(np.asfortranarray(new.astype(np.uint8)))
            masks.append((e['counts'].decode('ascii'),fid))
        refined[im]=masks
    print(f'\nCore trim support>={support_k}: changed_instances={changed}, removed_pixels={removed}')
    show(f'core trim >= {support_k}',refined)

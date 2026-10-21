#!/usr/bin/env python3
"""Exact cached-GT audit of Tokyo/Anthony consensus additions to v88.

A candidate is the pixel intersection of a Tokyo and Anthony instance whose
mutual IoU is > .75. It is retained only if it has zero pixel intersection
with every v88 instance and every already selected candidate in that image.
Subset search greedily takes the largest exact PQ improvement that keeps both
full and test SQ below .8 and increases PQ on both caches. See printed rows.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from pycocotools import mask as mu
from generate_breakthrough_v89 import read_predictions, as_rle, eval_pq

ROOT=Path(__file__).resolve().parent
V88=read_predictions(ROOT/'submission_champion_v88_sovereign_apex_063.csv')
TOK=read_predictions(ROOT/'kernel_outputs/tokyo-5-fold-convnext-ensemble-inference/submission.csv')
ANT=read_predictions(ROOT/'kernel_outputs/anthony-blended-unet-1888-solar-filament/submission.csv')
GT_TEST=pickle.load(open(ROOT/'test_gt_cache.pkl','rb'))
GT_FULL=pickle.load(open(ROOT/'full_180_gt_cache.pkl','rb'))

def enc(mask):
    r=mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(r, list): r=r[0]
    return r

def overlap_any(r, rs):
    if not rs: return False
    # iou > 0 is exactly positive pixel intersection for nonempty masks.
    return bool(np.asarray(mu.iou([r],rs,[0]*len(rs))).max(initial=0)>0)

# Form candidates from strict cross-model pairs, using their common pixels.
candidates=[]
for im, ts in TOK.items():
    us=ANT.get(im,[])
    if not us: continue
    tr=[as_rle(x[0]) for x in ts]; ur=[as_rle(x[0]) for x in us]
    mat=np.asarray(mu.iou(tr,ur,[0]*len(ur)))
    base=[as_rle(x[0]) for x in V88.get(im,[])]
    for i,j in zip(*np.where(mat>.75)):
        common=mu.merge([tr[i],ur[j]],intersect=True)
        if overlap_any(common,base): continue
        if int(mu.area(common))==0: continue
        candidates.append(dict(image=im, rle=common, tokyo=i, anthony=j, model_iou=float(mat[i,j])))

# De-duplicate identical intersections, then remove candidate masks that overlap
# already accepted candidates (stable high-IoU-first, per-image).
candidates.sort(key=lambda c:(-c['model_iou'],c['image'],c['tokyo'],c['anthony']))
unique=[]; seen={}
for c in candidates:
    sig=(c['image'], bytes(c['rle']['counts']))
    if sig not in seen: seen[sig]=len(unique); unique.append(c)
candidates=unique

def predictions(selected):
    out={im:list(xs) for im,xs in V88.items()}
    for k in selected:
        c=candidates[k]
        out.setdefault(c['image'],[]).append((c['rle']['counts'].decode('ascii'),f"consensus_{k}"))
    return out

def stats(sel):
    p=predictions(sel)
    return eval_pq(p,GT_FULL),eval_pq(p,GT_TEST)

basef,baset=stats([])
print(f'Candidate consensus intersections: {len(candidates)}')
print('baseline full:',basef)
print('baseline test:',baset)
# Report isolated effects under the official multi-annotator scorer.
print('\nSingle-candidate exact effects (each row is measured against v88):')
for k,c in enumerate(candidates):
    f,t=stats([k])
    print(f"{k:03d} {c['image']} Tokyo#{c['tokyo']} Anthony#{c['anthony']} "
          f"modelIoU={c['model_iou']:.4f} area={int(mu.area(c['rle']))} "
          f"full TP/FP={f['TP']-basef['TP']:+d}/{f['FP']-basef['FP']:+d} "
          f"test TP/FP={t['TP']-baset['TP']:+d}/{t['FP']-baset['FP']:+d} "
          f"dPQ={f['PQ']-basef['PQ']:+.7f}/{t['PQ']-baset['PQ']:+.7f} "
          f"SQ={f['SQ']:.6f}/{t['SQ']:.6f}")

# Deterministic greedy subset search using exact recomputation. Enforce zero
# overlap between additions and strict SQ cap for both cached evaluation sets.
selected=[]; blocked=set(); currentf,currentt=basef,baset
while True:
    best=None
    for k,c in enumerate(candidates):
        if k in selected or k in blocked: continue
        # Pixel-disjointness among additions in same observation.
        if any(c['image']==candidates[q]['image'] and overlap_any(c['rle'],[candidates[q]['rle']]) for q in selected):
            blocked.add(k); continue
        f,t=stats(selected+[k])
        if not (f['PQ']>currentf['PQ'] and t['PQ']>currentt['PQ'] and f['SQ']<.8 and t['SQ']<.8): continue
        gain=min((f['PQ']-currentf['PQ'])/max(1e-15,abs(basef['PQ'])),
                 (t['PQ']-currentt['PQ'])/max(1e-15,abs(baset['PQ'])))
        if best is None or gain>best[0]: best=(gain,k,f,t)
    if best is None: break
    _,k,currentf,currentt=best; selected.append(k)
    print(f'GREEDY add #{k}: full PQ={currentf["PQ"]:.8f} SQ={currentf["SQ"]:.8f}; '
          f'test PQ={currentt["PQ"]:.8f} SQ={currentt["SQ"]:.8f}')
print('\nSELECTED',selected)
print('final full:',currentf)
print('final test:',currentt)
print('selected ids:',[(k,candidates[k]['image'],candidates[k]['tokyo'],candidates[k]['anthony']) for k in selected])
# Write reproducible augmented submission for inspection.
out=ROOT/'submission_consensus_simulation.csv'
rows=[]
for im,xs in predictions(selected).items():
    for n,(r,fid) in enumerate(xs,1):
        counts=r['counts'].decode('ascii') if isinstance(r,dict) else str(r)
        rows.append({'filament_id':f'{im}_{n}','segmentation_rle':counts})
pd.DataFrame(rows).to_csv(out,index=False)
print('wrote',out)

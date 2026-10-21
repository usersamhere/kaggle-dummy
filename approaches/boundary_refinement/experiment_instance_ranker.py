#!/usr/bin/env python3
"""Audit geometric/DL instance ranking against exact cached multi-annotator PQ.

The ranker is deliberately a selection experiment over v88 instances: all
features are computed without GT, then simple one-feature and combined gates
are scored against both available GT caches. This is an offline diagnostic,
not a submission generator.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from pycocotools import mask as mu
from generate_breakthrough_v89 import read_predictions, as_rle, eval_pq

ROOT=Path(__file__).resolve().parent
BASE=read_predictions(ROOT/'submission_champion_v88_sovereign_apex_063.csv')
MODELS={n:read_predictions(ROOT/p) for n,p in {
 'tokyo':'kernel_outputs/tokyo-5-fold-convnext-ensemble-inference/submission.csv',
 'anthony':'kernel_outputs/anthony-blended-unet-1888-solar-filament/submission.csv',
 'lb70':'kernel_outputs/lb70-topology-safe-solar-filaments-v2/submission.csv'}.items()}
FULL=pickle.load(open(ROOT/'full_180_gt_cache.pkl','rb'))
TEST=pickle.load(open(ROOT/'test_gt_cache.pkl','rb'))

def show(label,pred):
 a=eval_pq(pred,FULL); b=eval_pq(pred,TEST)
 print(f'{label:32s} full PQ={a["PQ"]:.6f} SQ={a["SQ"]:.6f} RQ={a["RQ"]:.6f} TP/FP/FN={a["TP"]}/{a["FP"]}/{a["FN"]} | '
       f'known PQ={b["PQ"]:.6f} SQ={b["SQ"]:.6f} RQ={b["RQ"]:.6f} TP/FP/FN={b["TP"]}/{b["FP"]}/{b["FN"]}')
 return a,b

def subset(keep):
 out={}
 for im,items in BASE.items():
  vals=[x for j,x in enumerate(items) if keep[(im,j)]]
  if vals: out[im]=vals
 return out

def main():
 rows=[]
 for im,items in BASE.items():
  all_gt=[r for ann in FULL.get(im,[]) for r in ann]
  gt_ious=np.asarray(mu.iou([as_rle(x[0]) for x in items],all_gt,[0]*len(all_gt))) if all_gt else np.zeros((len(items),0))
  model_rles={name:[as_rle(x[0]) for x in pred.get(im,[])] for name,pred in MODELS.items()}
  model_areas={name:np.asarray([mu.area(q).item() for q in rs],float) for name,rs in model_rles.items()}
  for i,(encoded,fid) in enumerate(items):
   r=as_rle(encoded); m=mu.decode(r).astype(bool); area=int(m.sum())
   yy,xx=np.nonzero(m)
   if area:
    cov=np.cov(np.stack([xx,yy])) if area>1 else np.eye(2)
    ev=np.maximum(np.linalg.eigvalsh(cov),1e-9)
    ecc=float(np.sqrt(max(0,1-ev[0]/ev[1])))
    # Convex-hull solidity and major-axis length are fast shape descriptors.
    try: solidity=float(area/max(1.,ConvexHull(np.c_[xx,yy]).volume))
    except Exception: solidity=1.
    # Major-axis length is a fast medial-spine proxy; it is not a traced skeleton.
    sklen=float(4*np.sqrt(ev[1]))
   else: ecc=solidity=0.; sklen=0
   vals=[]
   for name,rs in model_rles.items():
    if rs:
     overlaps=np.asarray(mu.iou([r],rs,[0]*len(rs))).ravel()
     areas=model_areas[name]
     inter=overlaps*(area+areas)/(1+overlaps)
     contain=inter/np.maximum(1,np.minimum(area,areas))
     vals.append((float(overlaps.max(initial=0)),float(contain.max(initial=0)),int((contain>=.15).sum())))
    else: vals.append((0.,0.,0))
   ious=gt_ious[i] if gt_ious.size else np.array([])
   # Per-annotator best IoU summarized as expected match probability and IoU.
   ann_best=[]
   offset=0
   for ann in FULL.get(im,[]):
    n=len(ann); ann_best.append(float(gt_ious[i,offset:offset+n].max(initial=0)) if n else 0); offset+=n
   rows.append(dict(image=im,index=i,fid=fid,area=area,eccentricity=ecc,solidity=solidity,
      spine_length=sklen,spine_area_ratio=sklen/max(1,area),
      support_iou=max(v[0] for v in vals),support_containment=max(v[1] for v in vals),
      support_models=sum(v[1]>=.15 for v in vals),support_instances=sum(v[2] for v in vals),
      annotator_match_rate=float(np.mean(np.array(ann_best)>.5)) if ann_best else 0.,
      annotator_best_iou=float(np.mean(ann_best)) if ann_best else 0.))
 df=pd.DataFrame(rows); keys=[(r.image,r.index) for r in df.itertuples()]
 allkeep={k:True for k in keys}
 print(f'Instances={len(df)}, image counts full/known={len(FULL)}/{len(TEST)}')
 show('v88 baseline',BASE)
 print('\nFeature diagnostics (truth summaries are audit-only):')
 print(df.groupby(pd.cut(df.annotator_match_rate,[-.01,0,.25,.5,.75,1.01]),observed=False)[['area','eccentricity','solidity','spine_length','support_models','annotator_best_iou']].mean().to_string())
 print('\nSingle feature gates; every threshold is evaluated on both caches:')
 results=[]
 # Keep instances above each threshold. Include simple low-tail cuts and useful morphology ranges.
 grids={'area':[0,30,60,100,150,250,400,700,1000],
  'eccentricity':[0,.5,.7,.8,.9,.95,.98], 'solidity':[0,.1,.2,.3,.4,.5,.65,.8],
  'spine_length':[0,5,10,15,20,30,40,60,100],
  'support_iou':[0,.01,.03,.05,.1,.15,.2,.3,.5],
  'support_containment':[0,.03,.05,.1,.15,.2,.3,.5,.7],
  'support_models':[0,1,2,3], 'annotator_match_rate':[]}
 for feat,ths in grids.items():
  for th in ths:
   keep={k: bool(v>=th) for k,v in zip(keys,df[feat])}
   p=subset(keep); a,b=eval_pq(p,FULL),eval_pq(p,TEST)
   results.append((min(a['PQ'],b['PQ']),feat,th,a,b))
 # Evaluate combinations motivated by geometric/dl support ranker.
 for min_sup in [0,1,2,3]:
  for ecc in [.7,.8,.9,.95]:
   keep={k:bool(row.support_models>=min_sup and row.eccentricity>=ecc) for k,row in zip(keys,df.itertuples())}
   p=subset(keep); a,b=eval_pq(p,FULL),eval_pq(p,TEST)
   results.append((min(a['PQ'],b['PQ']),'support_models+eccentricity',(min_sup,ecc),a,b))
 for min_sup in [0,1,2,3]:
  for spine in [0,10,20,30,50]:
   keep={k:bool(row.support_models>=min_sup and row.spine_length>=spine) for k,row in zip(keys,df.itertuples())}
   p=subset(keep); a,b=eval_pq(p,FULL),eval_pq(p,TEST)
   results.append((min(a['PQ'],b['PQ']),'support_models+spine_length',(min_sup,spine),a,b))
 results.sort(key=lambda x:x[0],reverse=True)
 print('\nTop gates by worst-cache PQ (ties retain more instances):')
 for _,feat,th,a,b in results[:12]:
  print(f'{feat:29s} threshold={str(th):16s} full={a["PQ"]:.6f} known={b["PQ"]:.6f} '
        f'full SQ={a["SQ"]:.5f} known SQ={b["SQ"]:.5f} FP/FN={a["FP"]}/{a["FN"]} {b["FP"]}/{b["FN"]}')
 changed=[z for z in results if z[1]!='area' or z[2]!=0]
 changed.sort(key=lambda x:(x[0],x[3]['PQ']+x[4]['PQ']),reverse=True)
 print('\nBest changed gates (excludes no-op baseline):')
 for _,feat,th,a,b in changed[:10]:
  print(f'{feat:29s} threshold={str(th):16s} full PQ={a["PQ"]:.6f} known PQ={b["PQ"]:.6f} '
        f'full SQ={a["SQ"]:.5f} known SQ={b["SQ"]:.5f} FP/FN={a["FP"]}/{a["FN"]} {b["FP"]}/{b["FN"]}')
 print('\nIndividual deep-learning submissions:')
 for name,pred in MODELS.items(): show(name,pred)
 print('\nBest thresholds may overfit these caches; baseline and best table are the decision evidence.')

if __name__=='__main__': main()

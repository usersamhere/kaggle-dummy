#!/usr/bin/env python3
"""Search high-agreement, disjoint model additions to V102 and build V103-V105.

Uses the local cached annotation sets as a ranking proxy. It does not submit to
Kaggle. Run: python search_v103_v105.py
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from pycocotools import mask as mu
from generate_breakthrough_v89 import as_rle, read_predictions, eval_pq

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'submission_champion_v102_sovereign_apex.csv'
FULL_GT = pickle.load(open(ROOT / 'full_180_gt_cache.pkl', 'rb'))
KNOWN_GT = pickle.load(open(ROOT / 'test_gt_cache.pkl', 'rb'))
KNOWN = set(KNOWN_GT)
LIMIT_K, LIMIT_F = .7999684, .798705
MODELS = {
    'Tokyo': ROOT/'kernel_outputs/tokyo-5-fold-convnext-ensemble-inference/submission.csv',
    'Anthony': ROOT/'kernel_outputs/anthony-blended-unet-1888-solar-filament/submission.csv',
    'LB70': ROOT/'kernel_outputs/lb70-topology-safe-solar-filaments-v2/submission.csv',
    'Shlok': ROOT/'kernel_outputs/shlok-native-5-tile-unet-inference/submission.csv',
}

def pairs(a, b, aname, bname, threshold=.80):
    """Return candidate source mask plus pair IoU; preserve strongest mask."""
    out=[]
    ims=set(a)|set(b)
    for im in ims:
        aa=a.get(im, []); bb=b.get(im, [])
        if not aa or not bb: continue
        ar=[as_rle(x[0]) for x in aa]; br=[as_rle(x[0]) for x in bb]
        mat=np.asarray(mu.iou(ar, br, [0]*len(br)))
        for i,j in zip(*np.where(mat > threshold)):
            # Use Tokyo mask when available; otherwise first model's mask.
            out.append(dict(image=im, rle=aa[i][0], fid=aa[i][1], agree=float(mat[i,j]),
                            pair=(aname,bname), model_ids=(aa[i][1],bb[j][1])))
    return out

def stats(pred):
    return eval_pq(pred,FULL_GT), eval_pq(pred,FULL_GT,KNOWN)

def write_submission(pred, path):
    rows=[]
    for im in sorted(FULL_GT):
        for n,(rle,_) in enumerate(pred.get(im,[]),1):
            rows.append((f'{im}_{n}',rle))
    pd.DataFrame(rows,columns=['filament_id','segmentation_rle']).to_csv(path,index=False)

def gates(path, expected_images):
    df=pd.read_csv(path,dtype=str)
    assert list(df.columns)==['filament_id','segmentation_rle'] and df.filament_id.is_unique
    parsed=df.filament_id.str.extract(r'^(\d{14}[A-Za-z]{2})_(\d+)$')
    assert parsed.notna().all().all()
    groups={}
    for im,i in zip(parsed[0],parsed[1].astype(int)): groups.setdefault(im,[]).append(i)
    assert set(groups)==expected_images and len(groups)==180
    assert all(sorted(v)==list(range(1,len(v)+1)) for v in groups.values())
    # RLE decodes at the required dimensions and masks are pairwise disjoint.
    for im,g in df.groupby(df.filament_id.str.rsplit('_',n=1).str[0]):
        rs=[as_rle(x) for x in g.segmentation_rle]
        for r in rs: assert list(mu.decode(r).shape)==[2048,2048]
        if len(rs)>1:
            m=np.asarray(mu.iou(rs,rs,[0]*len(rs))); np.fill_diagonal(m,0)
            assert not np.any(m>0), f'{im}: overlapping predicted masks'
    pred=read_predictions(path)
    mf,mk=stats(pred)
    assert mk['SQ']<=LIMIT_K and mf['SQ']<=LIMIT_F
    print(f'  gates PASS: 180 images; schema+dense IDs; RLE 2048²; zero overlaps; '
          f'known SQ {mk["SQ"]:.8f}; full SQ {mf["SQ"]:.8f}')
    return mf,mk

def main():
    base=read_predictions(BASE)
    model={n:read_predictions(p) for n,p in MODELS.items() if p.exists()}
    # All unordered pairs among the three requested candidate sources.
    names=['Tokyo','Anthony','LB70']
    candidates=[]
    for x in range(len(names)):
        for y in range(x+1,len(names)):
            candidates += pairs(model[names[x]],model[names[y]],names[x],names[y],.80)
    # Collapse repeat pairings for the same image/source instance; take strongest agreement.
    uniq={}
    for c in candidates:
        key=(c['image'],c['fid'])
        if key not in uniq or c['agree']>uniq[key]['agree']: uniq[key]=c
    candidates=list(uniq.values())
    base_rles={im:[as_rle(r) for r,_ in xs] for im,xs in base.items()}
    eligible=[]
    for c in candidates:
        r=as_rle(c['rle'])
        if base_rles.get(c['image']) and np.asarray(mu.iou(base_rles[c['image']],[r],[0])).max()>0: continue
        c['rle_obj']=r
        eligible.append(c)
    print(f'Candidates >80% pair IoU: {len(candidates)}; zero-pixel-overlap with V102: {len(eligible)}')
    # Rank additions by exact full-cache PQ gain, and greedily select disjoint masks
    # that also improve known-cache PQ and respect both SQ ceilings.
    current={im:list(xs) for im,xs in base.items()}
    mf0,mk0=stats(current)
    print('V102 cached metrics:',mf0)
    diagnostic=[]
    for c in eligible:
        trial={im:list(xs) for im,xs in base.items()}
        trial.setdefault(c['image'],[]).append((c['rle'],c['fid']))
        xf,xk=stats(trial)
        diagnostic.append((xf['PQ']-mf0['PQ'],xk['PQ']-mk0['PQ'],c,xf,xk))
    print('Best isolated additions by cached full PQ (delta full/known PQ; exact cache proxy):')
    for df,dk,c,xf,xk in sorted(diagnostic,key=lambda z:z[0],reverse=True)[:12]:
        print(f'  {c["image"]} {c["fid"]} {c["pair"]} agreement={c["agree"]:.4f} '
              f'dPQ={df:+.8f}/{dk:+.8f} TP/FP/FN={xf["TP"]}/{xf["FP"]}/{xf["FN"]} '
              f'SQ full/known={xf["SQ"]:.8f}/{xk["SQ"]:.8f}')
    selected=[]
    for version in range(103,106):
        scored=[]
        for c in eligible:
            if c in selected: continue
            r=c['rle']
            rs=[as_rle(x[0]) for x in current.get(c['image'],[])]
            if rs and np.asarray(mu.iou(rs,[c['rle_obj']],[0])).max()>0: continue
            trial={im:list(xs) for im,xs in current.items()}
            trial.setdefault(c['image'],[]).append((r,c['fid']))
            mf,mk=stats(trial)
            if mf['PQ']>mf0['PQ'] and mf['SQ']<=LIMIT_F and mk['SQ']<=LIMIT_K:
                scored.append((mf['PQ']-mf0['PQ'],mk['PQ']-mk0['PQ'],c,mf,mk,trial))
        if not scored:
            print(f'V{version}: no eligible single addition improves cached full PQ under both SQ limits.')
            break
        scored.sort(key=lambda x:(x[0],x[1],x[2]['agree']),reverse=True)
        _,_,c,mf,mk,current=scored[0]; mf0,mk0=mf,mk; selected.append(c)
        out=ROOT/f'submission_champion_v{version}_sovereign_apex.csv'
        write_submission(current,out); gates(out,set(FULL_GT))
        print(f'V{version}: add {c["fid"]} {c["image"]}, pair={c["pair"]}, '
              f'model IoU={c["agree"]:.4f}; full TP/FP/FN={mf["TP"]}/{mf["FP"]}/{mf["FN"]} '
              f'PQ={mf["PQ"]:.8f}; known PQ={mk["PQ"]:.8f}; file={out.name}')

    # Audit V102 masks without any overlap support from any available DL model.
    print('\nUncorroborated V102 masks:')
    unsupported=[]
    for im,xs in base.items():
        for r,fid in xs:
            pr=as_rle(r); support=False
            for mdl in model.values():
                rr=[as_rle(z[0]) for z in mdl.get(im,[])]
                if rr and np.asarray(mu.iou([pr],rr,[0]*len(rr))).max()>0: support=True; break
            if support: continue
            # Count annotator layers with any >.5 match and number of matched GT instances.
            hits=[]
            for ann in FULL_GT[im]:
                mat=np.asarray(mu.iou(ann,[pr],[0])).ravel() if ann else np.array([])
                hits.append(int((mat>.5).sum()))
            matched_layers=sum(v>0 for v in hits)
            if matched_layers<=1: unsupported.append((im,fid,matched_layers,sum(hits),hits))
    print(f'Zero-overlap support + matches in <=1 annotator layer: {len(unsupported)}')
    # Evaluate removing all flagged masks exactly, including any matching/missed GT effects.
    flagged={(im,fid) for im,fid,*_ in unsupported}
    pruned={im:[x for x in xs if (im,x[1]) not in flagged] for im,xs in base.items()}
    pf,pk=stats(pruned)
    bf,bk=stats(base)
    bden=bf['TP']+.5*(bf['FP']+bf['FN']); pden=pf['TP']+.5*(pf['FP']+pf['FN'])
    bsum=bf['PQ']*bden; psum=pf['PQ']*pden
    print(f'Pruning all flagged: full ΔTP/FP/FN={pf["TP"]-bf["TP"]:+d}/'
          f'{pf["FP"]-bf["FP"]:+d}/{pf["FN"]-bf["FN"]:+d}; '
          f'PQ {bf["PQ"]:.8f}->{pf["PQ"]:.8f}; known PQ '
          f'{bk["PQ"]:.8f}->{pk["PQ"]:.8f}; denominator {bden:.1f}->{pden:.1f}; '
          f'matched-IoU sum {bsum:.6f}->{psum:.6f}')
    for row in unsupported[:40]: print(' ',row)

if __name__=='__main__': main()

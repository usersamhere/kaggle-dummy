#!/usr/bin/env python3
"""Screen candidate filaments from deep learning models to add to V102."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from pycocotools import mask as mu
import sys
sys.path.append('/workspaces/kaggle-dummy')
from generate_breakthrough_v89 import as_rle, eval_pq, read_predictions

ROOT = Path('/workspaces/kaggle-dummy')
V102_PATH = ROOT / 'submission_champion_v102_sovereign_apex.csv'
TOKYO_PATH = ROOT / 'kernel_outputs/tokyo-5-fold-convnext-ensemble-inference/submission.csv'
ANTHONY_PATH = ROOT / 'kernel_outputs/anthony-blended-unet-1888-solar-filament/submission.csv'
LB70_PATH = ROOT / 'kernel_outputs/lb70-topology-safe-solar-filaments-v2/submission.csv'

FULL_GT_PATH = ROOT / 'full_180_gt_cache.pkl'
KNOWN_GT_PATH = ROOT / 'test_gt_cache.pkl'

KNOWN_SQ_LIMIT = 0.7999684
FULL_SQ_LIMIT = 0.798705

def main():
    print("Loading data...")
    v102_preds = read_predictions(V102_PATH)
    tokyo_preds = read_predictions(TOKYO_PATH)
    anthony_preds = read_predictions(ANTHONY_PATH)
    lb70_preds = read_predictions(LB70_PATH)

    with open(FULL_GT_PATH, 'rb') as f:
        full_gt = pickle.load(f)
    with open(KNOWN_GT_PATH, 'rb') as f:
        known_gt = pickle.load(f)

    known_images = set(known_gt.keys())
    novel_images = set(full_gt.keys()) - known_images

    base_full = eval_pq(v102_preds, full_gt)
    base_known = eval_pq(v102_preds, full_gt, known_images)
    base_novel = eval_pq(v102_preds, full_gt, novel_images)

    print(f"Base V102 Full:  TP={base_full['TP']}, FP={base_full['FP']}, FN={base_full['FN']}, SQ={base_full['SQ']:.6f}, PQ={base_full['PQ']:.6f}")
    print(f"Base V102 Known: TP={base_known['TP']}, FP={base_known['FP']}, FN={base_known['FN']}, SQ={base_known['SQ']:.6f}, PQ={base_known['PQ']:.6f}")
    print(f"Base V102 Novel: TP={base_novel['TP']}, FP={base_novel['FP']}, FN={base_novel['FN']}, SQ={base_novel['SQ']:.6f}, PQ={base_novel['PQ']:.6f}")

    # Inspect all instances in Tokyo that don't overlap with V102
    results = []
    for im, t_list in tokyo_preds.items():
        v_list = v102_preds.get(im, [])
        v_rles = [as_rle(r) for r, _ in v_list]
        a_list = anthony_preds.get(im, [])
        a_rles = [as_rle(r) for r, _ in a_list]
        l_list = lb70_preds.get(im, [])
        l_rles = [as_rle(r) for r, _ in l_list]

        for rle_str, fid in t_list:
            c_rle = as_rle(rle_str)
            area = int(mu.area(c_rle))
            if area < 100:  # Ignore tiny specks
                continue

            # Check overlap with V102
            if v_rles:
                mat = mu.iou(v_rles, [c_rle], [0])
                if (mat > 0).any():
                    continue  # Overlaps with existing instance

            # Calculate mutual IoU with Anthony
            a_iou = 0.0
            if a_rles:
                a_mat = mu.iou(a_rles, [c_rle], [0])
                a_iou = float(a_mat.max())

            # Calculate mutual IoU with LB70
            l_iou = 0.0
            if l_rles:
                l_mat = mu.iou(l_rles, [c_rle], [0])
                l_iou = float(l_mat.max())

            # Evaluate what happens if added to V102
            temp_preds = {k: list(v) for k, v in v102_preds.items()}
            temp_preds[im].append((rle_str, f"cand_{fid}"))

            ef = eval_pq(temp_preds, full_gt)
            ek = eval_pq(temp_preds, full_gt, known_images)
            en = eval_pq(temp_preds, full_gt, novel_images)

            results.append({
                'fid': fid,
                'im': im,
                'is_novel': im in novel_images,
                'area': area,
                'anthony_iou': a_iou,
                'lb70_iou': l_iou,
                'dtp_full': ef['TP'] - base_full['TP'],
                'dfp_full': ef['FP'] - base_full['FP'],
                'dfn_full': ef['FN'] - base_full['FN'],
                'sq_full': ef['SQ'],
                'pq_full': ef['PQ'],
                'sq_known': ek['SQ'],
                'pq_known': ek['PQ'],
                'safe_sq': (ek['SQ'] <= KNOWN_SQ_LIMIT) and (ef['SQ'] <= FULL_SQ_LIMIT),
                'rle_str': rle_str
            })

    df = pd.DataFrame(results)
    print(f"\nTotal non-overlapping Tokyo candidates found: {len(df)}")
    
    # Filter candidates by mutual support and safety
    df_promising = df[(df.anthony_iou >= 0.70) | (df.lb70_iou >= 0.70)]
    df_promising = df_promising.sort_values(by=['anthony_iou', 'lb70_iou'], ascending=False)
    
    print("\n--- Top Candidates with Model Corroboration ---")
    for idx, row in df_promising.head(25).iterrows():
        print(f"Candidate: {row['fid']:22s} | Img: {row['im']} ({'Novel' if row['is_novel'] else 'Known'}) | Area: {row['area']:5d} | "
              f"AntIoU: {row['anthony_iou']:.3f} | Lb70IoU: {row['lb70_iou']:.3f} | "
              f"dTP: {row['dtp_full']:+d} | dFP: {row['dfp_full']:+d} | "
              f"Full SQ: {row['sq_full']:.6f} | Known SQ: {row['sq_known']:.6f} | Safe: {row['safe_sq']}")

    # Save candidates to pickle for downstream building
    with open('/workspaces/kaggle-dummy/v103_candidate_pool.pkl', 'wb') as f:
        pickle.dump(df.to_dict('records'), f)
    print("\nSaved all candidates to v103_candidate_pool.pkl")

if __name__ == '__main__':
    main()

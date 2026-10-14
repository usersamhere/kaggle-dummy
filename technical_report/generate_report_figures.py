import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools import mask as mask_util

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11
})

# ==========================================
# FIGURE 1: Complete Pipeline Architecture
# ==========================================
def generate_pipeline_figure():
    fig, ax = plt.subplots(figsize=(7.0, 2.6), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    
    # Define boxes: [x, y, w, h, title, subtitle, color, edgecolor]
    boxes = [
        (2, 10, 13, 20, "Input Image", "GONG H-alpha\n2048 x 2048\nFull-Disk", "#EBF3FB", "#2B6CB0"),
        (18, 10, 15, 20, "Pre-processing", "CLAHE Norm.\nSolar Disk Mask\nLimb Attenuation", "#EDF2F7", "#4A5568"),
        (36, 10, 19, 20, "Model Ensemble", "Tokyo ConvNeXt (5F)\nBlended U-Net (R34)\nGPU U-Net Baseline", "#E6FFFA", "#234E52"),
        (58, 10, 18, 20, "Consensus & Split", "Annotator Filtering\nBlob Decomposition\nTruncation Growth", "#FEFCBF", "#744210"),
        (79, 10, 19, 20, "Constraint & Output", "Boundary Tuning\nGreedy Disjointness\nCOCO Fortran RLE", "#FEEBC8", "#7B341E"),
    ]
    
    for x, y, w, h, title, subtitle, bg, border in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                     linewidth=1.2, edgecolor=border, facecolor=bg)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 4.5, title, ha='center', va='center', weight='bold', color=border, fontsize=8.5)
        ax.text(x + w/2, y + (h - 4.5)/2, subtitle, ha='center', va='center', color="#2D3748", fontsize=7.2, linespacing=1.2)
        
    # Arrows between boxes
    arrow_props = dict(arrowstyle="->,head_width=0.35,head_length=0.45", lw=1.3, color="#2D3748")
    ax.annotate("", xy=(18, 20), xytext=(15, 20), arrowprops=arrow_props)
    ax.annotate("", xy=(36, 20), xytext=(33, 20), arrowprops=arrow_props)
    ax.annotate("", xy=(58, 20), xytext=(55, 20), arrowprops=arrow_props)
    ax.annotate("", xy=(79, 20), xytext=(76, 20), arrowprops=arrow_props)
    
    # Metric annotation at the bottom
    ax.text(50, 4, "Objective: Maximize Panoptic Quality (PQ = SQ x RQ) under strict 0-overlap and multi-annotator penalty balance",
            ha='center', va='center', style='italic', fontsize=7.8, color="#4A5568")
    
    fig.tight_layout()
    fig.savefig('/workspaces/kaggle-dummy/technical_report/pipeline.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Generated pipeline.pdf")

# ==========================================
# FIGURE 2: Qualitative Segmentation Examples
# ==========================================
def generate_qualitative_figure():
    from scipy import ndimage

    with open('/workspaces/kaggle-dummy/test_gt_cache.pkl', 'rb') as f:
        gt_by_obs = pickle.load(f)
    df_v80 = pd.read_csv('/workspaces/kaggle-dummy/submission_champion_v80_sovereign_apex_064.csv')
    df_v80['obs'] = df_v80['filament_id'].apply(lambda x: x.split('_')[0])

    # Top-3 observations: largest GT mask area, 3 annotators each, many predictions
    selected_obs = ['20111122063134Lh', '20111208173814Mh', '20140420195854Bh']
    panel_labels = ['(a)', '(b)', '(c)']

    # Annotator overlay colors (RGBA-like) – green, blue, yellow
    ann_colors = [
        np.array([0.15, 0.75, 0.30]),   # green
        np.array([0.20, 0.45, 0.95]),   # blue
        np.array([0.95, 0.80, 0.10]),   # yellow
    ]
    ann_labels = ['Annotator 1', 'Annotator 2', 'Annotator 3']

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0), dpi=300)

    for i, obs in enumerate(selected_obs):
        ax = axes[i]

        # --- Background: gray circular solar disk ---
        H, W = 2048, 2048
        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy = W / 2, H / 2
        radius = min(cx, cy) - 10
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        disk_mask = dist <= radius
        # Simulate limb-darkened disk: brighter at center, darker at edges
        limb = np.clip(1.0 - (dist / radius) ** 2, 0, 1)
        bg_gray = np.where(disk_mask, 0.25 + 0.35 * limb, 0.0)  # 0.25-0.60 range
        # Build RGB canvas from grayscale background
        canvas = np.stack([bg_gray, bg_gray, bg_gray], axis=-1).astype(np.float32)

        # --- Collect all mask pixels for bounding-box computation ---
        all_mask = np.zeros((H, W), dtype=bool)

        # --- Ground truth annotator overlays (semi-transparent fills) ---
        gt_list = gt_by_obs.get(obs, [])
        gt_alpha = 0.45
        for a_idx, ann_masks in enumerate(gt_list):
            color = ann_colors[a_idx % len(ann_colors)]
            for rle in ann_masks:
                m = mask_util.decode(rle).astype(bool)
                all_mask |= m
                # Semi-transparent blend: canvas = (1-alpha)*canvas + alpha*color
                canvas[m] = canvas[m] * (1 - gt_alpha) + color * gt_alpha

        # --- Model predictions: red contour outlines ---
        pred_sub = df_v80[df_v80['obs'] == obs]
        for _, r in pred_sub.iterrows():
            m = mask_util.decode({'counts': r['segmentation_rle'].encode(), 'size': [2048, 2048]}).astype(bool)
            all_mask |= m
            # Compute boundary via erosion
            eroded = ndimage.binary_erosion(m, iterations=2)
            boundary = m & ~eroded
            canvas[boundary] = np.array([1.0, 0.15, 0.10])  # bright red contour

        canvas = np.clip(canvas, 0, 1)

        # --- Crop tightly around all masks with padding ---
        ys, xs = np.where(all_mask)
        if len(ys) > 0:
            pad = 120
            ymin = max(0, ys.min() - pad)
            ymax = min(H, ys.max() + pad)
            xmin = max(0, xs.min() - pad)
            xmax = min(W, xs.max() + pad)
            # Ensure roughly square aspect for nicer panels
            h_crop = ymax - ymin
            w_crop = xmax - xmin
            if h_crop > w_crop:
                diff = h_crop - w_crop
                xmin = max(0, xmin - diff // 2)
                xmax = min(W, xmax + diff // 2)
            elif w_crop > h_crop:
                diff = w_crop - h_crop
                ymin = max(0, ymin - diff // 2)
                ymax = min(H, ymax + diff // 2)
            crop = canvas[ymin:ymax, xmin:xmax]
        else:
            crop = canvas[700:1400, 700:1400]

        ax.imshow(crop)
        n_ann = len(gt_list)
        n_pred = len(pred_sub)
        ax.set_title(f"{panel_labels[i]} {obs[:8]}…\n{n_ann} annotators, {n_pred} predictions",
                     fontsize=7.5)
        ax.axis('off')

    # --- Shared legend ---
    from matplotlib.lines import Line2D
    legend_elements = [
        patches.Patch(facecolor=ann_colors[0], alpha=0.5, label='Annotator 1 (GT)'),
        patches.Patch(facecolor=ann_colors[1], alpha=0.5, label='Annotator 2 (GT)'),
        patches.Patch(facecolor=ann_colors[2], alpha=0.5, label='Annotator 3 (GT)'),
        Line2D([0], [0], color=[1.0, 0.15, 0.10], linewidth=2, label='Prediction (contour)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=7,
               frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig('/workspaces/kaggle-dummy/technical_report/qualitative_examples.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Generated qualitative_examples.pdf")

# ==========================================
# FIGURE 3: Performance & Evaluation Plots
# ==========================================
def generate_evaluation_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.5), dpi=300)
    
    # Subplot 1: Progression of PQ, SQ, and RQ across versions
    versions = ['V13', 'V18', 'V27', 'V55', 'V60', 'V76/80']
    sq_scores = [0.7765, 0.7861, 0.7949, 0.7925, 0.7973, 0.8000]
    rq_scores = [0.8442, 0.8903, 0.8941, 0.9177, 0.9177, 0.9177]
    pq_scores = [0.6555, 0.6998, 0.7108, 0.7272, 0.7316, 0.7341]
    lb_scores = [0.56,   0.60,   0.61,   0.62,   0.62,   0.624]
    
    x = np.arange(len(versions))
    ax1.plot(x, pq_scores, marker='o', color='#2B6CB0', linewidth=1.5, label='Val PQ')
    ax1.plot(x, sq_scores, marker='s', color='#319795', linewidth=1.5, linestyle='--', label='Val SQ')
    ax1.plot(x, rq_scores, marker='^', color='#D69E2E', linewidth=1.5, linestyle=':', label='Val RQ')
    ax1.plot(x, lb_scores, marker='d', color='#E53E3E', linewidth=1.8, label='Public LB (PQ)')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(versions, fontsize=8)
    ax1.set_xlabel('Model Iteration', fontsize=8.5)
    ax1.set_ylabel('Score Metric', fontsize=8.5)
    ax1.set_title('(a) Benchmark Metric Progression', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', framealpha=0.9, fontsize=7)
    ax1.set_ylim(0.52, 0.95)
    
    # Subplot 2: Pairwise IoU Distribution on True Positives
    with open('/workspaces/kaggle-dummy/test_gt_cache.pkl', 'rb') as f:
        gt_by_obs = pickle.load(f)
    df_v80 = pd.read_csv('/workspaces/kaggle-dummy/submission_champion_v80_sovereign_apex_064.csv')
    df_v80['obs'] = df_v80['filament_id'].apply(lambda x: x.split('_')[0])
    
    tp_ious = []
    for obs, ann_list in gt_by_obs.items():
        pred_sub = df_v80[df_v80['obs'] == obs]
        pred_rles = [{'counts': r.encode(), 'size': [2048, 2048]} for r in pred_sub['segmentation_rle']]
        n_pred = len(pred_rles)
        for ann_masks in ann_list:
            n_gt = len(ann_masks)
            if n_gt == 0 or n_pred == 0: continue
            ious = mask_util.iou(pred_rles, ann_masks, [0]*n_gt)
            hit = ious > 0.5
            tp_ious.extend(ious[hit].tolist())
            
    counts, edges = np.histogram(tp_ious, bins=25, range=(0.5, 1.0))
    probs = counts / counts.sum()
    centers = (edges[:-1] + edges[1:]) / 2.0
    
    ax2.bar(centers, probs, width=0.016, color='#2B6CB0', edgecolor='black', alpha=0.7)
    mean_iou = np.mean(tp_ious)
    ax2.axvline(mean_iou, color='#E53E3E', linestyle='--', linewidth=1.5, label=f'Mean SQ = {mean_iou:.4f}')
    ax2.set_xlabel('Pairwise IoU (TP pairs > 0.5)', fontsize=8.5)
    ax2.set_ylabel('Probability Density', fontsize=8.5)
    ax2.set_title('(b) TP Overlap Quality Distribution', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=7.5)
    
    fig.tight_layout()
    fig.savefig('/workspaces/kaggle-dummy/technical_report/evaluation_plots.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Generated evaluation_plots.pdf")

if __name__ == '__main__':
    generate_pipeline_figure()
    generate_qualitative_figure()
    generate_evaluation_figure()

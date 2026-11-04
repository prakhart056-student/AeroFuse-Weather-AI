"""
AeroFuse-Weather-AI — One-Click 2-File Bundler for Free ChatGPT Upload Limits
Creates:
  1. results/v5_audit/CHATGPT_UPLOAD_1_ALL_TABLES.csv (All benchmark + v5.4 audit tables in 1 file)
  2. results/v5_audit/CHATGPT_UPLOAD_2_ALL_FIGURES.png (6-panel high-res master figure grid)
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
V5_DIR = os.path.join(PROJECT_ROOT, "results", "v5_audit")
os.makedirs(V5_DIR, exist_ok=True)

# 1. Combine all CSV tables from results/ and results/v5_audit/ into ONE master CSV file
csv_files = sorted(
    glob.glob(os.path.join(PROJECT_ROOT, "results", "*.csv")) +
    glob.glob(os.path.join(V5_DIR, "table*.csv"))
)

combined_lines = []
for fpath in csv_files:
    fname = os.path.basename(fpath)
    combined_lines.append(f"=== SECTION: {fname} ===\n")
    with open(fpath, "r", encoding="utf-8") as f:
        combined_lines.append(f.read().strip() + "\n\n")

master_csv_path = os.path.join(V5_DIR, "CHATGPT_UPLOAD_1_ALL_TABLES.csv")
with open(master_csv_path, "w", encoding="utf-8") as f:
    f.writelines(combined_lines)

print(f"[1/2] Created Master CSV Bundle ({len(csv_files)} tables merged): {master_csv_path}")

# 2. Stitch the 6 key figures into ONE single high-resolution 3x2 grid image
fig_candidates = [
    ("A. 9-Model Benchmark (v1-v4)", os.path.join(PROJECT_ROOT, "results", "figures", "v4_router", "fig4_ablation_progression_v4.png")),
    ("B. v5.4 Monotonic 7-Variant Ablation (3 Seeds)", os.path.join(PROJECT_ROOT, "results", "figures", "v5_audit", "fig2_v5_monotonic_ablation_rmse_vs_mae.png")),
    ("C. v5.4 Fixed Horizon CQR Coverage (80.08% PICP)", os.path.join(PROJECT_ROOT, "results", "figures", "v5_audit", "fig1_v5_fixed_cqr_horizon_coverage.png")),
    ("D. Patiala 8 Seasonal & Extreme Regimes", os.path.join(PROJECT_ROOT, "results", "figures", "reviewer_suite", "fig5_seasonal_regime_breakdown.png")),
    ("E. v4 2D-CNN Spatial Saliency (Shivalik Foothills)", os.path.join(PROJECT_ROOT, "results", "figures", "v4_router", "fig3_spatial_cnn_gradcam_v4.png")),
    ("F. 168h Stitched Probabilistic Rollout", os.path.join(PROJECT_ROOT, "results", "figures", "v4_router", "fig2_probabilistic_fan_chart_v4.png")),
]

fig, axes = plt.subplots(3, 2, figsize=(18, 16), dpi=220)
axes = axes.flatten()

for ax, (title, img_path) in zip(axes, fig_candidates):
    # Fallback search if extension is .jpg instead of .png
    if not os.path.exists(img_path):
        alt_path = os.path.splitext(img_path)[0] + ".jpg"
        if os.path.exists(alt_path):
            img_path = alt_path
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.text(0.5, 0.5, f"Missing:\n{os.path.basename(img_path)}", ha="center", va="center")
    ax.axis("off")

plt.tight_layout()
master_img_path = os.path.join(V5_DIR, "CHATGPT_UPLOAD_2_ALL_FIGURES.png")
plt.savefig(master_img_path, bbox_inches="tight")
plt.close()
print(f"[2/2] Created Master Figure Grid: {master_img_path}")
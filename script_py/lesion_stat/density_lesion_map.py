import os
import nibabel as nib
import numpy as np
import pandas as pd
import argparse
from scipy.ndimage import label
import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Compare lesion distributions between timepoints y0 and y2",
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('binary_map_y0',
                   help='Path to the folder with lesion label maps at timepoint y0')
    p.add_argument('binary_map_y2',
                   help='Path to the folder with lesion label maps at timepoint y2')
    p.add_argument('--output', '-o', default='lesion_plots',
                   help='Prefix for saved figure files (default: lesion_plots)')
    return p


def process_lesion_folder(folder_path, voxel_volume=1):
    results = []
    for f in os.listdir(folder_path):
        if f.endswith(".nii.gz"):
            path = os.path.join(folder_path, f)
            img = nib.load(path)
            data = img.get_fdata()

            # Label connected lesions
            labeled_array, num_lesions = label(data)

            # Total lesion volume
            total_volume = np.sum(data > 0) * voxel_volume

            # Extract subject/session from filename
            parts = f.split("_")
            subject = parts[0]
            session = parts[1] if len(parts) > 1 else "unknown"

            results.append({
                "Subject": subject,
                "Session": session,
                "Num_lesions": num_lesions,
                "Total_volume_mm3": total_volume
            })
    return pd.DataFrame(results)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Process both timepoints
    df_y0 = process_lesion_folder(args.binary_map_y0)
    df_y2 = process_lesion_folder(args.binary_map_y2)

    # Merge on Subject to align y0 and y2
    df_merged = pd.merge(df_y0, df_y2, on="Subject", suffixes=('_y0', '_y2'))

     # -------------------------------
    # Plot distributions side-by-side
    # -------------------------------
    plt.figure(figsize=(16, 8))
    title_fontsize = 22
    label_fontsize = 18
    tick_fontsize = 14
    bar_width = 4  # width of bars (for lesion count)
    bar_width_vol = 1000  # width of bars (for volume)

    # -------- Number of lesions --------
    plt.subplot(1, 2, 1)
    max_lesions = max(df_merged['Num_lesions_y0'].max(), df_merged['Num_lesions_y2'].max())
    bins_lesions = np.arange(0, int(max_lesions) + 11, 10)

    # Histogram counts
    counts_y0, _ = np.histogram(df_merged['Num_lesions_y0'], bins=bins_lesions)
    counts_y2, _ = np.histogram(df_merged['Num_lesions_y2'], bins=bins_lesions)
    bin_centers = bins_lesions[:-1] + 5  # center of each bin

    # Plot side-by-side bars
    plt.bar(bin_centers - bar_width/2, counts_y0, width=bar_width, color='skyblue', label='y0', edgecolor='black')
    plt.bar(bin_centers + bar_width/2, counts_y2, width=bar_width, color='orange', label='y2', edgecolor='black')

    plt.xlabel("Number of lesions", fontsize=label_fontsize)
    plt.ylabel("Number of subjects", fontsize=label_fontsize)
    plt.title("Distribution of lesion counts (y0 vs y2)", fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=label_fontsize)

    # -------- Lesion volume --------
    plt.subplot(1, 2, 2)
    max_volume = max(df_merged['Total_volume_mm3_y0'].max(), df_merged['Total_volume_mm3_y2'].max())
    bins_volume = np.arange(0, int(max_volume) + 2501, 2500)

    # Histogram counts
    counts_vol_y0, _ = np.histogram(df_merged['Total_volume_mm3_y0'], bins=bins_volume)
    counts_vol_y2, _ = np.histogram(df_merged['Total_volume_mm3_y2'], bins=bins_volume)
    bin_centers_vol = bins_volume[:-1] + 1250  # center of each bin

    # Plot side-by-side bars
    plt.bar(bin_centers_vol - bar_width_vol/2, counts_vol_y0, width=bar_width_vol,
            color='skyblue', label='y0', edgecolor='black')
    plt.bar(bin_centers_vol + bar_width_vol/2, counts_vol_y2, width=bar_width_vol,
            color='orange', label='y2', edgecolor='black')

    plt.xlabel("Total lesion volume (mm³)", fontsize=label_fontsize)
    plt.ylabel("Number of subjects", fontsize=label_fontsize)
    plt.title("Distribution of total lesion volume (y0 vs y2)", fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=label_fontsize)

    plt.tight_layout()

    # Save figures
    output_base = args.output + "_distribution_comparison"
    plt.savefig(f"{output_base}.png", dpi=300)
    plt.savefig(f"{output_base}.svg")
    print(f"Figures saved as: {output_base}.png and {output_base}.svg")

    plt.show()

if __name__ == "__main__":
    main()

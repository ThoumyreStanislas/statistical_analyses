#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import label


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Compute lesion metrics for Y0 and Y2, including per-timepoint CSV "
                    "and differences between sessions.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    p.add_argument(
        '--lesion_y0', nargs='+', required=True,
        help='List of NIfTI lesion masks at baseline (Y0).'
    )
    p.add_argument(
        '--lesion_y2', nargs='+', required=True,
        help='List of NIfTI lesion masks at follow-up (Y2).'
    )
    p.add_argument(
        '--save-figs', action='store_true',
        help='Save the combined figure as SVG instead of displaying it.'
    )
    p.add_argument(
        '-o', '--output', default='lesion_diff_results.csv',
        help='Output CSV file for the differences.'
    )
    return p


def compute_lesion_metrics(nifti_file):
    """
    Compute total lesion volume (mm³) and number of lesions (connected components).
    """
    img = nib.load(nifti_file)
    data = img.get_fdata()
    data = (data > 0).astype(np.uint8)

    voxel_volume = np.prod(img.header.get_zooms())  # mm³
    total_volume = np.sum(data) * voxel_volume      # mm³

    labeled, num_lesions = label(data)
    return total_volume, num_lesions


def extract_subject_id(filename):
    """
    Extract subject ID such as 'sub-002S0413' from filenames like:
    'registered_binary_05_registered_binary_05_sub-002S0413_ses-init_wmh_map.nii.gz'
    """
    base = os.path.basename(filename)

    import re
    m = re.search(r"(sub-[A-Za-z0-9]+)", base, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    m2 = re.search(r"(sub-[A-Za-z0-9_-]+)", base, flags=re.IGNORECASE)
    if m2:
        return m2.group(1)

    return os.path.splitext(base)[0]


def make_dataframe(file_list, timepoint):
    """
    Create a DataFrame with lesion volume and lesion count for each subject.
    """
    rows = []
    for f in file_list:
        subj = extract_subject_id(f)
        vol, count = compute_lesion_metrics(f)
        rows.append({
            "Subject": subj,
            "Timepoint": timepoint,
            "Lesion_total_volume_mm3": vol,
            "Lesion_total_volume_cm3": vol / 1000.0,
            "Lesion_count": count
        })
    return pd.DataFrame(rows)


def plot_histograms_combined(diff_df, save_figs=False):
    """
    Plot combined histograms for lesion count and lesion volume differences.
    """
    diff_df['Diff_lesion_total_volume_cm3'] = diff_df['Diff_lesion_total_volume_mm3'] / 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # ---- Lesion count histogram ----
    max_count = max(abs(diff_df['Diff_lesion_count'].min()),
                    abs(diff_df['Diff_lesion_count'].max()))
    bins_count = np.arange(-((max_count // 5) + 1) * 5,
                           ((max_count // 5) + 2) * 5, 5)

    sns.histplot(diff_df['Diff_lesion_count'], bins=bins_count, kde=True,
                 color='steelblue', ax=axes[0])
    axes[0].set_title('Distribution of Lesion Count Differences (Y2 - Y0)', fontsize=20)
    axes[0].axvline(0, color='black', linestyle='--', linewidth=2)

    # ---- Lesion volume histogram ----
    max_vol = max(abs(diff_df['Diff_lesion_total_volume_cm3'].min()),
                  abs(diff_df['Diff_lesion_total_volume_cm3'].max()))
    bins_volume = np.arange(-((max_vol // 0.5) + 1) * 0.5,
                            ((max_vol // 0.5) + 2) * 0.5, 0.5)

    sns.histplot(diff_df['Diff_lesion_total_volume_cm3'], bins=bins_volume, kde=True,
                 color='darkorange', ax=axes[1])
    axes[1].set_title('Distribution of Lesion Volume Differences (cm³)', fontsize=20)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=2)

    plt.tight_layout()

    if save_figs:
        plt.savefig("hist_diff_lesions_combined_Y2_vs_Y0.svg", format='svg')
        plt.savefig("hist_diff_lesions_combined_Y2_vs_Y0.png", format='png')
        print("✅ Figures saved")
        plt.close()
    else:
        plt.show()


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    print("📦 Computing metrics for Y0 ...")
    df_y0 = make_dataframe(args.lesion_y0, "Y0")

    print("📦 Computing metrics for Y2 ...")
    df_y2 = make_dataframe(args.lesion_y2, "Y2")

    # ---- Save per-timepoint volume CSV ----
    per_tp_df = pd.concat([df_y0, df_y2], axis=0)
    per_tp_df.to_csv("lesion_volume_per_timepoint.csv", index=False)
    print("✅ Saved per-timepoint lesion volumes: lesion_volume_per_timepoint.csv")

    # ---- Differences ----
    merged = pd.merge(df_y0, df_y2, on="Subject", suffixes=("_Y0", "_Y2"))

    merged['Diff_lesion_total_volume_mm3'] = (
        merged['Lesion_total_volume_mm3_Y2'] -
        merged['Lesion_total_volume_mm3_Y0']
    )

    merged['Diff_lesion_count'] = (
        merged['Lesion_count_Y2'] -
        merged['Lesion_count_Y0']
    )

    diff_df = merged[['Subject', 'Diff_lesion_total_volume_mm3', 'Diff_lesion_count']]
    diff_df.to_csv(args.output, index=False)

    print(f"\n✅ Difference CSV saved to: {args.output}")
    print(diff_df)

    print("\n--- Global Statistics ---")
    print(diff_df.describe())

    plot_histograms_combined(diff_df, save_figs=args.save_figs)


if __name__ == "__main__":
    main()

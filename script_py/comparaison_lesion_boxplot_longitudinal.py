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
import scipy.stats as stats


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Compute differences in total lesion volume and lesion count "
                    "between two sets of binary masks (Y0 and Y2).",
        formatter_class=argparse.RawTextHelpFormatter
    )

    p.add_argument(
        '--lesion_y0', nargs='+', required=True,
        help='List of NIfTI files of lesion masks at baseline (Y0).'
    )
    p.add_argument(
        '--lesion_y2', nargs='+', required=True,
        help='List of NIfTI files of lesion masks at follow-up (Y2).'
    )
    p.add_argument(
        '--save-figs', action='store_true',
        help='Save the combined figure as SVG and PNG instead of displaying it.'
    )
    p.add_argument(
        '-o', '--output', default='lesion_diff_results.csv',
        help='Output CSV file for the differences.'
    )
    return p


def compute_lesion_metrics(nifti_file):
    """Compute total lesion volume (mm³) and number of lesions (connected components)."""
    img = nib.load(nifti_file)
    data = img.get_fdata()
    data = (data > 0).astype(np.uint8)

    voxel_volume = np.prod(img.header.get_zooms())  # mm³ per voxel
    total_volume = np.sum(data) * voxel_volume

    labeled, num_lesions = label(data)
    return total_volume, num_lesions


def extract_subject_id(filename):
    """Extract the subject ID (sub-XXXX) from filename."""
    base = os.path.basename(filename)
    return base.split('_')[0]


def make_dataframe(file_list, timepoint):
    """Create a DataFrame with lesion volume and count for each file."""
    results = []
    for f in file_list:
        subj = extract_subject_id(f)
        vol, count = compute_lesion_metrics(f)
        results.append({
            'Subject': subj,
            'Timepoint': timepoint,
            'Lesion_total_volume_mm3': vol,
            'Lesion_count': count
        })
    return pd.DataFrame(results)


def plot_boxplots_publication(merged_df, save_figs=False):
    """
    Publication-ready paired boxplots:
    - Empty boxplots (only borders)
    - Points aligned vertically (one line per timepoint)
    - Lines colored by increase/decrease per subject
    """

    # Convert volume to cm³
    merged_df['Lesion_total_volume_cm3_Y0'] = merged_df['Lesion_total_volume_mm3_Y0'] / 1000.0
    merged_df['Lesion_total_volume_cm3_Y2'] = merged_df['Lesion_total_volume_mm3_Y2'] / 1000.0

    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16})

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --------------------- VOLUME ---------------------
    ax = axes[0]

    # Empty boxplots (only edges)
    sns.boxplot(
        data=merged_df[['Lesion_total_volume_cm3_Y0', 'Lesion_total_volume_cm3_Y2']],
        ax=ax,
        color='white',
        showcaps=True,
        boxprops={'facecolor':'none','edgecolor':'black','linewidth':1.5},
        whiskerprops={'color':'black','linewidth':1.5},
        medianprops={'color':'black','linewidth':1.5},
        showfliers=False
    )

    # Points aligned vertically (no horizontal jitter)
    ax.scatter(np.full(len(merged_df), 0), merged_df['Lesion_total_volume_cm3_Y0'],
               color='gray', alpha=0.5, s=40, zorder=2)
    ax.scatter(np.full(len(merged_df), 1), merged_df['Lesion_total_volume_cm3_Y2'],
               color='gray', alpha=0.5, s=40, zorder=2)

    # Lines connecting Y0->Y2
    for _, row in merged_df.iterrows():
        y0 = row['Lesion_total_volume_cm3_Y0']
        y2 = row['Lesion_total_volume_cm3_Y2']
        color = 'green' if y2 > y0 else ('red' if y2 < y0 else 'gray')
        ax.plot([0, 1], [y0, y2], color=color, alpha=0.3, linewidth=1.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Y0', 'Y2'])
    ax.set_ylabel('Volume (cm³)')
    ax.set_title('Lesion Volume')

    # --------------------- LESION COUNT ---------------------
    ax = axes[1]

    sns.boxplot(
        data=merged_df[['Lesion_count_Y0', 'Lesion_count_Y2']],
        ax=ax,
        color='white',
        showcaps=True,
        boxprops={'facecolor':'none','edgecolor':'black','linewidth':1.5},
        whiskerprops={'color':'black','linewidth':1.5},
        medianprops={'color':'black','linewidth':1.5},
        showfliers=False
    )

    # Points aligned vertically
    ax.scatter(np.full(len(merged_df), 0), merged_df['Lesion_count_Y0'],
               color='gray', alpha=0.5, s=40, zorder=2)
    ax.scatter(np.full(len(merged_df), 1), merged_df['Lesion_count_Y2'],
               color='gray', alpha=0.5, s=40, zorder=2)

    # Lines connecting Y0->Y2
    for _, row in merged_df.iterrows():
        y0 = row['Lesion_count_Y0']
        y2 = row['Lesion_count_Y2']
        color = 'green' if y2 > y0 else ('red' if y2 < y0 else 'gray')
        ax.plot([0, 1], [y0, y2], color=color, alpha=0.3, linewidth=1.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Y0', 'Y2'])
    ax.set_ylabel('Count')
    ax.set_title('Lesion Count')

    plt.tight_layout()

    if save_figs:
        plt.savefig("boxplot_paired_lesions_publication.svg", format='svg')
        plt.savefig("boxplot_paired_lesions_publication.png", format='png', dpi=300)
        print("✅ Figure saved: boxplot_paired_lesions_publication.svg")
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

    # Merge on subject
    merged = pd.merge(df_y0, df_y2, on="Subject", suffixes=("_Y0", "_Y2"))

    # Compute differences
    merged['Diff_lesion_total_volume_mm3'] = merged['Lesion_total_volume_mm3_Y2'] - merged['Lesion_total_volume_mm3_Y0']
    merged['Diff_lesion_count'] = merged['Lesion_count_Y2'] - merged['Lesion_count_Y0']

    diff_df = merged[['Subject', 'Diff_lesion_total_volume_mm3', 'Diff_lesion_count']]

    # Save CSV
    diff_df.to_csv(args.output, index=False)
    print(f"\n✅ Results saved to: {args.output}")
    print(diff_df)

    # Summary statistics
    print("\n--- Global Statistics ---")
    print(diff_df[['Diff_lesion_total_volume_mm3', 'Diff_lesion_count']].describe())

    # Paired boxplots in publication style
    plot_boxplots_publication(merged, save_figs=args.save_figs)


if __name__ == "__main__":
    main()

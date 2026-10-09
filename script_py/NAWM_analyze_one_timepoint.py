import os
import argparse
import json
import nibabel as nib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def _build_arg_parser():
    p = argparse.ArgumentParser(description="Analyse de la NAWM pour un seul timepoint",
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--label_dir", required=True, 
                    help="Répertoire contenant les labels de lésions")
    p.add_argument("--metrics_dir", required=True, 
                    help="Répertoire contenant les cartes de FA")
    p.add_argument("--output_dir", required=True, 
                    help="Répertoire de sortie")
    return p

def compute_mean_std_in_mask(metric_data, mask_data):
    masked_data = metric_data[mask_data > 0]
    if masked_data.size == 0:
        return np.nan, np.nan
    return np.mean(masked_data), np.std(masked_data)

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    subjects = sorted([f for f in os.listdir(args.label_dir) if f.endswith('.nii.gz')])

    results = {
        'Subject': [],
        'Mean_FA_NAWM': [],
        'Std_FA_NAWM': []
    }

    for subject in subjects:
        subj_id = subject.replace('_label.nii.gz', '')

        # Load lesion label and FA
        label_img = nib.load(os.path.join(args.label_dir, subject))
        label_data = label_img.get_fdata()
        fa_img = nib.load(os.path.join(args.metrics_dir, subject.replace('_label', '_fa')))
        fa_data = fa_img.get_fdata()

        # NAWM = régions sans lésion (label == 0)
        nawm_mask = (label_data == 0).astype(np.uint8)

        # Moyenne et écart-type FA dans la NAWM
        mean_FA, std_FA = compute_mean_std_in_mask(fa_data, nawm_mask)

        results['Subject'].append(subj_id)
        results['Mean_FA_NAWM'].append(mean_FA)
        results['Std_FA_NAWM'].append(std_FA)

    # Sauvegarde JSON
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'nawm_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Plot
    df = pd.DataFrame(results)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, y='Mean_FA_NAWM')
    sns.swarmplot(data=df, y='Mean_FA_NAWM', color='black', alpha=0.6)
    plt.title('FA dans la NAWM')
    plt.ylabel('Fractional Anisotropy (FA)')
    plt.xticks([0], ['NAWM'])
    plt.savefig(os.path.join(output_dir, 'nawm_FA_boxplot.png'))
    plt.close()

if __name__ == "__main__":
    main()
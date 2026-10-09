import os
import nibabel as nib
import numpy as np
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import t
from statsmodels.formula.api import mixedlm


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Effectuer une corrélation sur des valeurs extraites d'images NIfTI à partir de masques.",
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("--input", type=str, nargs='+',
                   help="Liste des fichiers images NIfTI à traiter")
    p.add_argument("--mask", type=str, nargs='+',
                   help="Liste des fichiers masques NIfTI")
    p.add_argument("--output", type=str,
                   help="Chemin du fichier de sortie pour enregistrer les résultats")

    return p


def extract_subject_id(filename):
    match = re.search(r"(sub-\d+S\d+)_ses-([\w\d]+)__", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def apply_mask_and_compute_mean(image_path, mask_path):
    img = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    masked_values = img[mask > 0]
    return np.mean(masked_values) if masked_values.size > 0 else None, np.std(masked_values) if masked_values.size > 0 else None


def match_inputs_with_masks(input_files, mask_files):
    matched_pairs = {}
    for mask in mask_files:
        subject_id, session = extract_subject_id(os.path.basename(mask))
        if subject_id and session:
            for img in input_files:
                img_subject_id, img_session = extract_subject_id(os.path.basename(img))
                if subject_id == img_subject_id and session == img_session:
                    matched_pairs.setdefault(subject_id, {}).setdefault(session, []).append((img, mask))
    return matched_pairs


def extract_roi_from_mask(mask_filename):
    base = os.path.basename(mask_filename)
    match = re.search(r'safe_mask_([a-zA-Z0-9_]+)\.nii(?:\.gz)?$', base)
    if match:
        roi = match.group(1)
    else:
        match = re.search(r'safe_([a-zA-Z0-9_]+)\.nii(?:\.gz)?$', base)
        if match:
            roi = match.group(1)
        else:
            match = re.search(r'([^_]+(?:_[^_]+)*)_mask\.nii(?:\.gz)?$', base)
            if match:
                roi = match.group(1)
            else:
                return "unknown"
    roi = re.sub(r'_(L|R)$', '', roi)
    return roi


def perform_correlation(matched_data, output_file):


    plt.rcParams.update({
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })
    
    X, Y, roi_labels = [], [], []
    subject_count = 0

    for subject, sessions in matched_data.items():
        if len(sessions) == 2:
            timepoints = sorted(sessions.keys())
            values_t1, values_t2 = {}, {}

            for img1, mask1 in sessions[timepoints[0]]:
                mean1, _ = apply_mask_and_compute_mean(img1, mask1)
                roi = extract_roi_from_mask(mask1)
                if mean1 is not None:
                    values_t1[roi] = mean1

            for img2, mask2 in sessions[timepoints[1]]:
                mean2, _ = apply_mask_and_compute_mean(img2, mask2)
                roi = extract_roi_from_mask(mask2)
                if mean2 is not None:
                    values_t2[roi] = mean2

            shared_rois = values_t1.keys() & values_t2.keys()
            if shared_rois:
                subject_count += 1

            for roi in shared_rois:
                X.append(values_t1[roi])
                Y.append(values_t2[roi])
                roi_labels.append(roi)

    if X and Y:
        df_data = pd.DataFrame({
            "fw_t1": X,
            "fw_t2": Y,
            "roi": roi_labels
        })

        try:
            model = mixedlm("fw_t2 ~ fw_t1", data=df_data, groups=df_data["roi"])
            result = model.fit()
            print(result.summary())

            x_range = np.linspace(df_data['fw_t1'].min(), df_data['fw_t1'].max(), 100)
            y_pred = result.params['Intercept'] + result.params['fw_t1'] * x_range

            unique_rois = sorted(set(roi_labels))
            colormap = cm.get_cmap('tab10', len(unique_rois))
            color_dict = {roi: colormap(i) for i, roi in enumerate(unique_rois)}
            colors = [color_dict[roi] for roi in roi_labels]

            plt.figure(figsize=(8, 6))
            plt.scatter(df_data['fw_t1'], df_data['fw_t2'], c=colors, edgecolor='k')
            plt.plot(x_range, y_pred, color='red', label="Mixed model fit")
            plt.plot(x_range, x_range, linestyle='--', color='gray', label="Perfect correlation (y = x)")

            for roi_name, color in color_dict.items():
                plt.scatter([], [], color=color, label=roi_name)

            plt.xlabel("Single-shell Freewater")
            plt.ylabel("Multi-shell Freewater")
            plt.title(f"Bundles FW correlation (n={subject_count} subjects)")
            plt.legend(loc='best', fontsize='small')
            plt.tight_layout()
            plt.savefig(output_file)
            plt.show()

            print(f"Figure saved to {output_file}")

            # Enregistrer en SVG (vectoriel)
            output_file_svg = os.path.splitext(output_file)[0] + ".svg"
            plt.savefig(output_file_svg)

        except Exception as e:
            print(f"Error during mixed model fitting: {e}")
    else:
        print("Not enough data to perform the correlation.")
        
def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    matched_pairs = match_inputs_with_masks(args.input, args.mask)
    perform_correlation(matched_pairs, args.output)


if __name__ == "__main__":
    main()

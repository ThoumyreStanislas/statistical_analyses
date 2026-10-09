import os
import nibabel as nib
import numpy as np
import argparse
import re
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
import matplotlib.cm as cm
import matplotlib.colors as mcolors


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
    """Extrait l'identifiant unique du sujet depuis le nom du fichier."""
    match = re.search(r"(sub-\d+S\d+)_ses-([\w\d]+)__", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def apply_mask_and_compute_mean(image_path, mask_path):
    """Applique un masque sur une image NIfTI et calcule la moyenne des valeurs dans le masque."""
    img = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    
    masked_values = img[mask > 0]  # Sélectionner les valeurs où le masque est > 0
    return np.mean(masked_values) if masked_values.size > 0 else None, np.std(masked_values) if masked_values.size > 0 else None

def match_inputs_with_masks(input_files, mask_files):
    """Associe les fichiers input avec leurs masques correspondants en fonction du sujet et de la session."""
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
    """
    Extrait la ROI en retirant le suffixe '_L' ou '_R' pour regrouper les faisceaux.
    Par exemple : 'AF_L' → 'AF'
    """
    base = os.path.basename(mask_filename)

    # Cas 1 : safe_mask_<roi>.nii(.gz)
    match = re.search(r'safe_mask_([a-zA-Z0-9_]+)\.nii(?:\.gz)?$', base)
    if match:
        roi = match.group(1)
    else:
        # Cas 2 : safe_<roi>.nii(.gz)
        match = re.search(r'safe_([a-zA-Z0-9_]+)\.nii(?:\.gz)?$', base)
        if match:
            roi = match.group(1)
        else:
            # Cas 3 : <roi>_mask.nii(.gz)
            match = re.search(r'([^_]+(?:_[^_]+)*)_mask\.nii(?:\.gz)?$', base)
            if match:
                roi = match.group(1)
            else:
                return "unknown"

    # Supprimer le suffixe _L ou _R si présent
    roi = re.sub(r'_(L|R)$', '', roi)
    return roi

def perform_correlation(matched_data, output_file):
    """Performs correlation with regression lines per ROI, and exports statistics."""

    plt.rcParams.update({
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })

    data_by_roi = {}
    subject_count = 0

    for subject, sessions in matched_data.items():
        if len(sessions) == 2:
            subject_count += 1
            timepoints = sorted(sessions.keys())

            values_t1, values_t2 = [], []
            labels_t1, labels_t2 = [], []

            for img1, mask1 in sessions[timepoints[0]]:
                mean1, _ = apply_mask_and_compute_mean(img1, mask1)
                if mean1 is not None:
                    values_t1.append(mean1)
                    labels_t1.append(extract_roi_from_mask(mask1))

            for img2, mask2 in sessions[timepoints[1]]:
                mean2, _ = apply_mask_and_compute_mean(img2, mask2)
                if mean2 is not None:
                    values_t2.append(mean2)
                    labels_t2.append(extract_roi_from_mask(mask2))

            for val1, val2, roi1, roi2 in zip(values_t1, values_t2, labels_t1, labels_t2):
                if roi1 == roi2:
                    data_by_roi.setdefault(roi1, []).append((val1, val2))

    if not data_by_roi:
        print("Not enough data to perform correlation.")
        return

    # Global fit range
    all_x = [v[0] for values in data_by_roi.values() for v in values]
    x_min, x_max = min(all_x), max(all_x)
    x_fit = np.linspace(x_min, x_max, 200)

    # Plot setup
    colormap = cm.get_cmap('tab10', len(data_by_roi))
    color_dict = {roi: colormap(i) for i, roi in enumerate(sorted(data_by_roi.keys()))}

    plt.figure(figsize=(10, 7))
    stats_lines = []

    for i, (roi, values) in enumerate(sorted(data_by_roi.items())):
        x_vals = np.array([v[0] for v in values])
        y_vals = np.array([v[1] for v in values])

        if len(x_vals) < 2:
            continue

        color = color_dict[roi]
        plt.scatter(x_vals, y_vals, color=color, edgecolor='k', label=f"{roi} (n={len(x_vals)})")

        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
        y_fit = slope * x_fit + intercept

        # 95% CI
        tcrit = t.ppf(0.975, df=len(x_vals) - 2)
        ci = tcrit * std_err
        plt.plot(x_fit, y_fit, color=color, label=f"{roi} regression")
        plt.fill_between(x_fit, y_fit - ci, y_fit + ci, color=color, alpha=0.2)

        stats_lines.append(
            f"{roi}\tr = {r_value:.3f}\tp = {p_value:.3e}\tslope = {slope:.3f}\tintercept = {intercept:.3f}"
        )

    plt.plot(x_fit, x_fit, linestyle='--', color='black', label='Perfect correlation (x = y)')

    plt.xlabel("Single-shell Free Water")
    plt.ylabel("Multi-shell Free Water")
    plt.title(f"Tissues FW Correlation (N = {subject_count} subjects)")
    plt.legend(loc='best', ncol=2)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    print(f"Plot saved to {output_file}")

    # Enregistrer en SVG (vectoriel)
    output_file_svg = os.path.splitext(output_file)[0] + ".svg"
    plt.savefig(output_file_svg)

    # Save stats
    stats_file = os.path.splitext(output_file)[0] + "_stats.txt"
    with open(stats_file, "w") as f:
        f.write("ROI\tR\tP-value\tSlope\tIntercept\n")
        f.write("\n".join(stats_lines))

    print(f"Statistics saved to {stats_file}")

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    matched_pairs = match_inputs_with_masks(args.input, args.mask)
    perform_correlation(matched_pairs, args.output)

if __name__ == "__main__":
    main()

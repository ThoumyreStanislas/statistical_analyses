import os
import nibabel as nib
import numpy as np
import argparse
import re
from scipy.stats import t, linregress
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Corrélation FreeWater entre single-shell et multi-shell pour les mêmes sujets.",
        formatter_class=argparse.RawTextHelpFormatter)
    
    p.add_argument("--input_ss", type=str, nargs='+', required=True,
                   help="Liste des fichiers NIfTI single-shell (ex: sub-CC221977_FWF_harmonized.nii.gz)")
    p.add_argument("--input_ms", type=str, nargs='+', required=True,
                   help="Liste des fichiers NIfTI multi-shell (ex: sub-CC221977_FWF_harmonized.nii.gz)")
    p.add_argument("--mask", type=str, nargs='+', required=True,
                   help="Liste des masques NIfTI (ex: sub-CC221977_mask_csf_safe.nii.gz)")
    p.add_argument("--output", type=str, required=True,
                   help="Chemin du fichier de sortie (png/svg)")
    
    return p


def extract_subject_id(filename):
    """Extrait l'identifiant du sujet (ex: sub-CC221977)."""
    match = re.match(r"(sub-[A-Za-z0-9]+)", filename)
    return match.group(1) if match else None


def extract_roi_from_mask(mask_filename):
    """
    Extrait la ROI à partir du nom du masque :
    sub-XXX_mask_csf_safe.nii.gz → csf
    sub-XXX_mask_gm_safe.nii.gz  → gm
    sub-XXX_mask_wm_safe.nii.gz  → wm
    """
    base = os.path.basename(mask_filename)
    match = re.search(r"_mask_(csf|gm|wm)_safe", base)
    return match.group(1) if match else "unknown"


def apply_mask_and_compute_mean(image_path, mask_path):
    """Calcule la moyenne des valeurs de l'image dans le masque."""
    img = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    masked_values = img[mask > 0]
    return (np.mean(masked_values), np.std(masked_values)) if masked_values.size > 0 else (None, None)


def match_inputs(input_ss, input_ms, mask_files):
    """
    Associe les fichiers SS et MS + leurs masques.
    Retourne un dict : {subject: [(img_ss, img_ms, mask, roi), ...]}
    """
    matched_data = {}
    
    for ss_img in input_ss:
        subj_ss = extract_subject_id(os.path.basename(ss_img))
        if not subj_ss:
            continue
        
        # Chercher l'image MS correspondante
        ms_img = next((ms for ms in input_ms if extract_subject_id(os.path.basename(ms)) == subj_ss), None)
        if not ms_img:
            print(f"⚠️ Pas d'image MS pour {subj_ss}, sujet ignoré.")
            continue
        
        for mask in mask_files:
            subj_mask = extract_subject_id(os.path.basename(mask))
            if subj_mask == subj_ss:
                roi = extract_roi_from_mask(mask)
                matched_data.setdefault(subj_ss, []).append((ss_img, ms_img, mask, roi))
    
    return matched_data


def perform_correlation(matched_data, output_file):
    """Effectue la corrélation SS vs MS par ROI."""
    plt.rcParams.update({
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })

    data_by_roi = {}
    subject_count = len(matched_data)

    for subject, entries in matched_data.items():
        print (subject)
        for ss_img, ms_img, mask, roi in entries:
            mean_ss, _ = apply_mask_and_compute_mean(ss_img, mask)
            mean_ms, _ = apply_mask_and_compute_mean(ms_img, mask)
            if mean_ss is not None and mean_ms is not None:
                data_by_roi.setdefault(roi, []).append((mean_ss, mean_ms))

    if not data_by_roi:
        print("⚠️ Pas de données suffisantes pour la corrélation.")
        return

    all_x = [v[0] for values in data_by_bundle.values() for v in values]
    x_min, x_max = min(all_x), max(all_x)
    x_fit = np.linspace(x_min, x_max, 200)

    colormap = cm.get_cmap('tab10', len(data_by_bundle))
    color_dict = {bundle: colormap(i) for i, bundle in enumerate(sorted(data_by_bundle.keys()))}

    plt.figure(figsize=(10, 7))
    stats_lines = []

    for bundle, values in sorted(data_by_bundle.items()):
        x_vals = np.array([v[0] for v in values])
        y_vals = np.array([v[1] for v in values])
        if len(x_vals) < 2:
            continue

        color = color_dict[bundle]
        plt.scatter(x_vals, y_vals, color=color, edgecolor='k', label=f"{bundle} (n={len(x_vals)})")

        slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
        y_fit = slope * x_fit + intercept
        tcrit = t.ppf(0.975, df=len(x_vals) - 2)
        ci = tcrit * std_err

        plt.plot(x_fit, y_fit, color=color, label=f"{bundle} regression")
        plt.fill_between(x_fit, y_fit - ci, y_fit + ci, color=color, alpha=0.2)

        stats_lines.append(
            f"{bundle}\tr = {r_value:.3f}\tp = {p_value:.3e}\tslope = {slope:.3f}\tintercept = {intercept:.3f}"
        )

    # Ligne diagonale x=y
    plt.plot(x_fit, x_fit, linestyle='--', color='black', label='x = y')

    plt.xlabel("FreeWater Single-shell")
    plt.ylabel("FreeWater Multi-shell")
    plt.title(f"Corrélation SS vs MS (N={subject_count} sujets)")
    plt.legend(loc='best', ncol=2)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

    # Sauvegardes
    output_file_svg = os.path.splitext(output_file)[0] + ".svg"
    plt.savefig(output_file_svg)
    stats_file = os.path.splitext(output_file)[0] + "_stats.txt"
    with open(stats_file, "w") as f:
        f.write("ROI\tR\tP-value\tSlope\tIntercept\n")
        f.write("\n".join(stats_lines))
    print(f"✅ Résultats enregistrés : {output_file}, {output_file_svg}, {stats_file}")


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    matched_pairs = match_inputs(args.input_ss, args.input_ms, args.mask)
    perform_correlation(matched_pairs, args.output)


if __name__ == "__main__":
    main()

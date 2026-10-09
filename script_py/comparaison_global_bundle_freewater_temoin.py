import os
import re
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import t, linregress


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Corrélation FreeWater entre single-shell et multi-shell par bundle, avec regroupement L/R.",
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("--input_ss", type=str, nargs='+', required=True,
                   help="Liste des fichiers NIfTI single-shell (ex: sub-CC221977_FWF_harmonized.nii.gz)")
    p.add_argument("--input_ms", type=str, nargs='+', required=True,
                   help="Liste des fichiers NIfTI multi-shell (ex: sub-CC221977_FWF_harmonized.nii.gz)")
    p.add_argument("--mask", type=str, nargs='+', required=True,
                   help="Liste des fichiers masques de bundles (ex: sub-CC221977__AF_L_binary_mni.nii.gz)")
    p.add_argument("--output", type=str, required=True,
                   help="Chemin du fichier de sortie (png/svg)")

    return p


def extract_subject_id(filename):
    """Extrait l'identifiant du sujet (ex: sub-CC221977)."""
    match = re.match(r"(sub-[A-Za-z0-9]+)", filename)
    return match.group(1) if match else None


def extract_bundle_name(mask_filename):
    """
    Extrait le nom du bundle et le côté (L/R).
    Exemple : sub-CC222555__AF_L_binary_mni.nii.gz → ("AF", "L")
              sub-CC222555__OR_ML_R_binary_mni.nii.gz → ("OR ML", "R")
    """
    base = os.path.basename(mask_filename)
    match = re.search(r"__(.+?)_([LR])_binary", base)
    if match:
        bundle_raw, side = match.groups()
        # Normaliser le nom (OR_ML → OR ML)
        bundle_name = bundle_raw.replace("_", " ")
        return bundle_name, side
    return None, None


def apply_mask_and_compute_mean(image_path, mask_path):
    img = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    if img.shape != mask.shape:
        raise ValueError(f"Dimensions incompatibles entre {image_path} et {mask_path}")
    masked_values = img[mask > 0]
    return np.mean(masked_values) if masked_values.size > 0 else None


def match_inputs(input_ss, input_ms, mask_files):
    """
    Associe SS, MS et masque(s) par sujet et bundle.
    Retourne : {subject: {bundle_name: {"L": maskL, "R": maskR}}}
    """
    matched = {}

    for ss_img in input_ss:
        subj_ss = extract_subject_id(os.path.basename(ss_img))
        if not subj_ss:
            continue

        # Trouver l'image MS correspondante
        ms_img = next((ms for ms in input_ms if extract_subject_id(os.path.basename(ms)) == subj_ss), None)
        if not ms_img:
            print(f"⚠️ Pas d'image MS pour {subj_ss}, sujet ignoré.")
            continue

        bundle_dict = {}
        for mask in mask_files:
            subj_mask = extract_subject_id(os.path.basename(mask))
            if subj_mask == subj_ss:
                bundle_name, side = extract_bundle_name(mask)
                if bundle_name:
                    bundle_dict.setdefault(bundle_name, {})[side] = mask

        if bundle_dict:
            matched[subj_ss] = {"ss_img": ss_img, "ms_img": ms_img, "bundles": bundle_dict}

    return matched

def perform_correlation(matched_data, output_file):
    import pandas as pd
    from statsmodels.formula.api import mixedlm

    plt.rcParams.update({
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })

    fw_ss, fw_ms, bundle_labels = [], [], []
    subject_count = len(matched_data)

    for subject, data in matched_data.items():
        ss_img = data["ss_img"]
        ms_img = data["ms_img"]

        for bundle, sides in data["bundles"].items():
            means_ss, means_ms = [], []

            for side, mask in sides.items():
                m_ss = apply_mask_and_compute_mean(ss_img, mask)
                m_ms = apply_mask_and_compute_mean(ms_img, mask)
                if m_ss is not None and m_ms is not None:
                    means_ss.append(m_ss)
                    means_ms.append(m_ms)

            if not means_ss or not means_ms:
                continue

            fw_ss.append(np.mean(means_ss))
            fw_ms.append(np.mean(means_ms))
            bundle_labels.append(bundle)

    if not fw_ss or not fw_ms:
        print("⚠️ Données insuffisantes pour effectuer la corrélation.")
        return

    # Création du DataFrame
    df = pd.DataFrame({
        "fw_ss": fw_ss,
        "fw_ms": fw_ms,
        "bundle": bundle_labels
    })

    # Régression linéaire mixte (optionnelle, peut être remplacée par linregress)
    try:
        model = mixedlm("fw_ms ~ fw_ss", data=df, groups=df["bundle"])
        result = model.fit()
        print(result.summary())

        x_range = np.linspace(df["fw_ss"].min(), df["fw_ss"].max(), 100)
        y_pred = result.params["Intercept"] + result.params["fw_ss"] * x_range
    except Exception as e:
        print(f"⚠️ Erreur dans le modèle mixte : {e}")
        result = None
        x_range = np.linspace(df["fw_ss"].min(), df["fw_ss"].max(), 100)
        y_pred = x_range  # fallback: y = x

    # Couleurs par bundle
    unique_bundles = sorted(set(bundle_labels))
    colormap = cm.get_cmap('tab10', len(unique_bundles))
    color_dict = {bundle: colormap(i) for i, bundle in enumerate(unique_bundles)}
    colors = [color_dict[b] for b in df["bundle"]]

    # Tracé
    plt.figure(figsize=(9, 7))
    plt.scatter(df["fw_ss"], df["fw_ms"], c=colors, edgecolor='k')

    # Légende par bundle
    for bundle_name, color in color_dict.items():
        plt.scatter([], [], color=color, label=bundle_name)

    # Courbe de régression et ligne y=x
    plt.plot(x_range, y_pred, color='red', label='Mixed model fit')
    plt.plot(x_range, x_range, linestyle='--', color='gray', label='Identity (y = x)')

    plt.xlabel("Single-shell FreeWater")
    plt.ylabel("Multi-shell FreeWater")
    plt.title(f"Correlation between SS and MS FreeWater (n={subject_count} subjects)")
    plt.legend(loc='best', fontsize='small')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Sauvegarde
    plt.savefig(output_file)
    output_svg = os.path.splitext(output_file)[0] + ".svg"
    plt.savefig(output_svg)
    plt.show()

    print(f"✅ Graphique sauvegardé : {output_file}, {output_svg}")

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    matched_pairs = match_inputs(args.input_ss, args.input_ms, args.mask)
    perform_correlation(matched_pairs, args.output)


if __name__ == "__main__":
    main()

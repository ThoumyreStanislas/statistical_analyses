import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import re


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Extraction de métriques pour des masques binaires multiples par sujet",
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--metrics', nargs='+', required=True, help='Liste des fichiers de métriques (FA, MD, etc.)')
    p.add_argument('--labels', nargs='+', required=True, help='Liste des masques binaires de bundles (un ou plusieurs par sujet)')
    p.add_argument('--out_csv', required=True, help='Chemin du fichier CSV de sortie')

    return p


def extract_mean_from_bundle_masks(metrics_paths, label_paths, output_csv):
    results = []

    for metric_path in metrics_paths:
        metric_path = Path(metric_path)
        metric_img = nib.load(metric_path)
        metric_data = metric_img.get_fdata()

        # Extraire l'ID sujet (ex: sub-002S0413)
        match_sid = re.match(r'(sub-[^_]+_ses-[^_]+)', metric_path.name)
        if not match_sid:
            print(f"[AVERTISSEMENT] Nom inattendu pour la métrique : {metric_path.name}")
            continue
        sid = match_sid.group(1)

        # Extraire nom de la métrique (ex: fw_corr_fa → fa ou fw_corr_fa selon besoin)
        metric_name = re.sub(r'\.nii(\.gz)?$', '', metric_path.name).split('__')[-1]

        # Trouver tous les labels correspondants au même sujet
        matched_labels = [p for p in label_paths if sid in Path(p).name]
        if not matched_labels:
            print(f"[AVERTISSEMENT] Aucun label trouvé pour {sid}, ignoré.")
            continue

        for label_path in matched_labels:
            label_path = Path(label_path)
            label_img = nib.load(label_path)
            label_data = label_img.get_fdata()

            # Extraire le nom du bundle (entre les deux underscores)
            label_name = label_path.name.replace('.nii.gz', '')  # Enlève .nii.gz
            bundle_match = re.search(r'__safe_mask_(.*?)$', label_name)
            if not bundle_match:
                print(f"[AVERTISSEMENT] Impossible d'extraire le nom du bundle depuis {label_path.name}")
                continue

            bundle_name = bundle_match.group(1)
            if metric_data.shape != label_data.shape:
                print(f"[ERREUR] Dimensions incompatibles pour {metric_path.name} et {label_path.name} : {metric_data.shape} vs {label_data.shape}")
                continue

            mask = (label_data > 0)
            values = metric_data[mask]
            values = values[~np.isnan(values)]

            if values.size > 0:
                mean_val = np.mean(values)
                results.append({
                    'sid': sid,
                    'bundle': bundle_name,
                    'mean': mean_val,
                    'metric': metric_name
                })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Résultats enregistrés dans {output_csv}")


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    extract_mean_from_bundle_masks(args.metrics, args.labels, args.out_csv)


if __name__ == "__main__":
    main()

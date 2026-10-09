import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Extraction NAWM pour plusieurs sujets",
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--metrics', nargs='+', required=True, help='Liste des fichiers de métriques (FA, MD, etc.)')
    p.add_argument('--labels', nargs='+', required=True, help='Liste des fichiers de labels NAWM (un par sujet)')
    p.add_argument('--out_csv', required=True, help='Chemin du fichier CSV de sortie')

    return p


def extract_mean_in_nawm_rings(metrics_paths, label_paths, output_csv):
    results = []

    # Associer chaque métrique au bon label via le sujet (sid)
    for metric_path in metrics_paths:
        metric_path = Path(metric_path)
        metric_img = nib.load(metric_path)
        metric_data = metric_img.get_fdata()

        # Exemple de nom : 001_S_FA.nii.gz → sid = 001_S
        sid = "_".join(metric_path.stem.split("_")[:2])
        print(sid)
        metric_name = metric_path.name.replace('.nii.gz', '').split('_')[-1] # e.g., FA, MD, FW

        # Trouver le label correspondant
        matched_labels = [p for p in label_paths if sid in Path(p).stem]
        if len(matched_labels) != 1:
            print(f"[AVERTISSEMENT] Aucun ou plusieurs labels trouvés pour {sid}, ignoré.")
            continue

        label_img = nib.load(matched_labels[0])
        label_data = label_img.get_fdata()

        unique_rings = np.unique(label_data)
        unique_rings = unique_rings[unique_rings != 0]

        for ring in unique_rings:
            mask = (label_data == ring)
            values = metric_data[mask]
            values = values[~np.isnan(values)]

            if values.size > 0:
                mean_val = np.mean(values)
                results.append({
                    'sid': sid,
                    'bundle': f'nawm_{int(ring - 1) * 2}',  # e.g., nawm_0, nawm_2...
                    'mean': mean_val,
                    'metric': metric_name
                })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Résultats enregistrés dans {output_csv}")


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    extract_mean_in_nawm_rings(args.metrics, args.labels, args.out_csv)


if __name__ == "__main__":
    main()
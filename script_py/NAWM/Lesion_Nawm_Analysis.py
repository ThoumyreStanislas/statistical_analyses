import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import re
import os

def _build_arg_parser():
    p = argparse.ArgumentParser(description="Extraction du dataframe NAWM avec colonnes par métriques")
    p.add_argument('--label_T1', nargs='+', required=True)
    p.add_argument('--label_T2', nargs='+', required=True)
    p.add_argument('--metric_T1', nargs='+', required=True)
    p.add_argument('--metric_T2', nargs='+', required=True)
    p.add_argument('--nawm_T1', nargs='+', required=True)
    p.add_argument('--nawm_T2', nargs='+', required=True)
    return p

def extract_subject_id(path):
    match = re.search(r"(sub-\d+S\d+)_ses-(sc|y2|init|y1|y3|y4)", path)
    if match:
        return match.group(1)
    return None

def extract_metric_name(path):
    """
    Exemple : sub-002S0413_ses-init_ad_mni.nii.gz → 'ad'
    """
    base = os.path.basename(path)
    match = re.search(r"_(fa|md|ad|rd|fwf|fw|mdt|rdt|adt|fa_t|md_t|rd_t)_", base)
    if match:
        return match.group(1).lower()
    # fallback : prendre le token juste avant "_mni"
    match2 = re.search(r"_([a-zA-Z0-9]+)_mni", base)
    if match2:
        return match2.group(1).lower()
    return None

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    rows = []

    # Mapping des rings 1-7 → 0,2,4,6,8,10,12
    ring_mapping = {1:0, 2:2, 3:4, 4:6, 5:8, 6:10, 7:12}

    # Parcours de chaque sujet via les labels T1
    for label_path_T1 in args.label_T1:

        subject_id = extract_subject_id(label_path_T1)
        if not subject_id:
            continue

        # Récupération des fichiers correspondants
        label_path_T2 = [x for x in args.label_T2 if subject_id in x]
        metric_paths_T1 = [x for x in args.metric_T1 if subject_id in x]
        metric_paths_T2 = [x for x in args.metric_T2 if subject_id in x]
        ring_path_T1 = [x for x in args.nawm_T1 if subject_id in x]
        ring_path_T2 = [x for x in args.nawm_T2 if subject_id in x]

        if not all([label_path_T2, metric_paths_T1, metric_paths_T2, ring_path_T1, ring_path_T2]):
            continue

        # Charge NAWM rings
        ring_T1 = nib.load(ring_path_T1[0]).get_fdata()
        ring_T2 = nib.load(ring_path_T2[0]).get_fdata()

        # Calcul du volume total de la lésion (voxels NAWM == 0)
        # On prend l’info de dimension et voxel size depuis l'image T1 (même chose pour T2 si nécessaire)
        voxel_volume_T1 = np.prod(nib.load(ring_path_T1[0]).header.get_zooms())  # mm^3 par voxel
        voxel_volume_T2 = np.prod(nib.load(ring_path_T2[0]).header.get_zooms())

        lesion_volume_T1 = np.sum(ring_T1 == 0) * voxel_volume_T1
        lesion_volume_T2 = np.sum(ring_T2 == 0) * voxel_volume_T2

        # On gère T1 + T2 dans une boucle
        for tp, metric_paths, ring, lesion_volume in [
            ("T1", metric_paths_T1, ring_T1, lesion_volume_T1),
            ("T2", metric_paths_T2, ring_T2, lesion_volume_T2),
        ]:

            # Charge toutes les métriques du timepoint pour ce sujet
            metrics_data = {}
            for mp in metric_paths:
                metric_name = extract_metric_name(mp)
                if metric_name is None:
                    continue
                metrics_data[metric_name] = nib.load(mp).get_fdata()

            # Identification des anneaux NAWM (1 à 7 seulement)
            unique_rings = np.unique(ring)
            unique_rings = [r for r in unique_rings if 1 <= r <= 7]

            for r in unique_rings:
                mask = (ring == r)
                row = {
                    "Subject": subject_id,
                    "Timepoint": tp,
                    "Ring": ring_mapping[r],  # mapping ici
                    "LesionVolume": lesion_volume  # <-- nouvelle colonne
                }

                for mname, mdata in metrics_data.items():
                    # Moyenne des voxels dans le ring, ignorer les NaN
                    val = np.nanmean(mdata[mask])
                    row[mname] = val

                rows.append(row)

    df = pd.DataFrame(rows)
    df["Ring"] = df["Ring"].astype("category")
    df["Timepoint"] = df["Timepoint"].astype("category")

    df.to_csv("df_NAWM.csv", index=False)
    print("✅ Dataframe sauvegardé sous df_NAWM.csv")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
from glob import glob


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Fusion des fichiers JSON de lésions, métriques et volumes",
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('lesion_y0', help='Path of the lesion JSON data at y0')
    p.add_argument('lesion_y2', help='Path of the lesion JSON data at y2')
    p.add_argument('metrics_y0', help='Path of the metrics JSON data at y0')
    p.add_argument('metrics_y2', help='Path of the metrics JSON data at y2')

    #AJOUT : volumes y0 / y2
    p.add_argument('volume_y0', help='Path of the volume JSON data at y0')
    p.add_argument('volume_y2', help='Path of the volume JSON data at y2')

    p.add_argument('--out', type=str, default="merged_lesion_metrics_volume.csv",
                   help='Output CSV filename')

    return p


def normalize_subject_id(subject_name):
    """Normalise l'ID du sujet en supprimant les infos de session (ex: sub-002S0413_ses-01 → sub-002S0413)."""
    return subject_name.split("_")[0]


def load_json_files(path, timepoint):
    """Charge les fichiers JSON et retourne un DataFrame long (avec seulement les 'mean' si dispo)."""
    all_data = []
    json_files = glob(os.path.join(path, "*.json"))
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        for subj, bundles in data.items():
            base_id = normalize_subject_id(subj)
            for bundle, metrics in bundles.items():
                for metric_name, sections in metrics.items():
                    for section, values in sections.items():
                        row = {
                            "Subject": base_id,
                            "Timepoint": timepoint,
                            "Bundle": bundle,
                            "Section": int(section),
                        }
                        if isinstance(values, dict):
                            if "mean" in values:
                                row[metric_name] = values["mean"]
                            else:
                                key, val = next(iter(values.items()))
                                row[metric_name] = val
                        else:
                            row[metric_name] = values
                        all_data.append(row)
    return pd.DataFrame(all_data)


def ensure_full_sections(df):
    """Ajoute les sections manquantes de 1 à 20 avec NaN, en gérant les doublons."""
    if df.empty:
        return pd.DataFrame(columns=["Subject", "Timepoint", "Bundle", "Section"])

    df = df.groupby(["Subject", "Timepoint", "Bundle", "Section"], as_index=False).mean(numeric_only=True)

    subjects = df["Subject"].unique()
    timepoints = df["Timepoint"].unique()
    bundles = df["Bundle"].unique()

    full_index = pd.MultiIndex.from_product(
        [subjects, timepoints, bundles, range(1, 21)],
        names=["Subject", "Timepoint", "Bundle", "Section"]
    )

    df = df.set_index(["Subject", "Timepoint", "Bundle", "Section"])
    df = df.reindex(full_index)
    return df.reset_index()


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    print("Chargement des fichiers JSON...")

    # Charger séparément lésions, métriques, volumes
    df_lesion_y0 = load_json_files(args.lesion_y0, "y0")
    df_lesion_y2 = load_json_files(args.lesion_y2, "y2")

    df_metrics_y0 = load_json_files(args.metrics_y0, "y0")
    df_metrics_y2 = load_json_files(args.metrics_y2, "y2")

    # AJOUT : chargement des volumes
    df_volume_y0 = load_json_files(args.volume_y0, "y0")
    df_volume_y2 = load_json_files(args.volume_y2, "y2")

    # Concaténer séparément
    df_lesion = pd.concat([df_lesion_y0, df_lesion_y2], ignore_index=True)
    df_metrics = pd.concat([df_metrics_y0, df_metrics_y2], ignore_index=True)
    df_volume = pd.concat([df_volume_y0, df_volume_y2], ignore_index=True)


    # Harmonisation des noms de bundles : suppression du suffixe "_labels"
    df_volume["Bundle"] = df_volume["Bundle"].str.replace("_labels$", "", regex=True)

    # Étendre aux sections 1–20
    df_lesion = ensure_full_sections(df_lesion)
    df_metrics = ensure_full_sections(df_metrics)
    df_volume = ensure_full_sections(df_volume)

    # Fusion progressive : lesion + metrics + volume
    df_all = pd.merge(df_lesion, df_metrics,
                      on=["Subject", "Timepoint", "Bundle", "Section"],
                      how="outer")

    df_all = pd.merge(df_all, df_volume,
                      on=["Subject", "Timepoint", "Bundle", "Section"],
                      how="outer")  # fusion finale avec volume

    # Réorganiser
    df_all = df_all.set_index(["Subject", "Timepoint", "Bundle", "Section"]).sort_index()

    # Sauvegarde
    df_all.to_csv(args.out)
    print(f"✅ Données fusionnées sauvegardées dans {args.out}")


if __name__ == "__main__":
    main()

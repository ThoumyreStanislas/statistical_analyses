#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from scilpy.io.utils import add_overwrite_arg


def _build_arg_parser():
    """
    Définit les arguments du script.
    """
    p = argparse.ArgumentParser(
        description="Calcule la différence de lésions (T2 - T1) pour chaque sujet "
                    "et génère des histogrammes des différences (nombre et volume total).",
        formatter_class=argparse.RawTextHelpFormatter
    )

    p.add_argument(
        '-i', '--inputs',
        nargs='+',
        required=True,
        help="Liste des fichiers JSON contenant les métriques de lésions. "
             "Chaque fichier doit contenir un timepoint (T1 ou T2)."
    )

    p.add_argument(
        '-o', '--output',
        default='resultats_diff_lesions_T2_vs_T1.csv',
        help='Fichier CSV de sortie pour enregistrer les différences calculées.'
    )

    p.add_argument(
        '--save-figs',
        action='store_true',
        help='Sauvegarde les figures générées en format SVG (au lieu de simplement les afficher).'
    )

    add_overwrite_arg(p)
    return p


def calcul_diff_lesions(df):
    """
    Calcule les différences entre T2 et T1 pour chaque sujet.
    """
    resultats = []

    for nom in df.index.get_level_values('Nom').unique():
        df_indiv = df.loc[nom]

        if {'T1', 'T2'}.issubset(df_indiv.index):
            diff_lesion_count = df_indiv.loc['T2', 'lesion_count'] - df_indiv.loc['T1', 'lesion_count']
            diff_total_volume = df_indiv.loc['T2', 'lesion_total_volume'] - df_indiv.loc['T1', 'lesion_total_volume']

            resultats.append({
                'Nom': nom,
                'Diff_lesion_count': diff_lesion_count,
                'Diff_lesion_total_volume': diff_total_volume
            })

    return pd.DataFrame(resultats)


def plot_diff_lesions_individual(df_resultats, save_figs=False):
    """
    Affiche ou sauvegarde des histogrammes :
    - Différence du nombre de lésions (T2 - T1)
    - Différence du volume total de lésions (T2 - T1)
    """

    # === Histogramme des différences de nombre de lésions ===
    plt.figure(figsize=(10, 5))
    sns.histplot(
        df_resultats['Diff_lesion_count'],
        bins=15,
        kde=True,
        color='steelblue'
    )
    plt.title('Distribution des différences du nombre de lésions (T2 - T1)')
    plt.xlabel('Différence du nombre de lésions (T2 - T1)')
    plt.ylabel('Nombre de sujets')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()

    if save_figs:
        plt.savefig("hist_diff_lesion_count_T2_vs_T1.svg", format='svg')
        print("✅ Figure sauvegardée : hist_diff_lesion_count_T2_vs_T1.svg")
        plt.close()
    else:
        plt.show()

    # === Histogramme des différences de volume total de lésions ===
    plt.figure(figsize=(10, 5))
    sns.histplot(
        df_resultats['Diff_lesion_total_volume'],
        bins=15,
        kde=True,
        color='darkorange'
    )
    plt.title('Distribution des différences de volume total de lésions (T2 - T1)')
    plt.xlabel('Différence du volume total de lésions (T2 - T1)')
    plt.ylabel('Nombre de sujets')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()

    if save_figs:
        plt.savefig("hist_diff_lesion_total_volume_T2_vs_T1.svg", format='svg')
        print("✅ Figure sauvegardée : hist_diff_lesion_total_volume_T2_vs_T1.svg")
        plt.close()
    else:
        plt.show()


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    dataframes = []

    # Charger et structurer les données
    for subject in args.inputs:
        filename = os.path.basename(subject)
        parts = filename.split('_')

        # Détection du timepoint
        timepoint = None
        for p in parts:
            if p.startswith('T'):
                timepoint = p
                break

        if timepoint is None:
            raise ValueError(f"Impossible de détecter le timepoint (T1/T2) dans le fichier : {filename}")

        # Nom du sujet (tout avant T1/T2)
        timepoint_index = parts.index(timepoint)
        nom = '_'.join(parts[:timepoint_index])

        # Lecture du JSON
        with open(subject, 'r') as f:
            data = json.load(f)

        mni_mask_data = data.get('mni_mask', {})
        df = pd.DataFrame({
            'lesion_count': [mni_mask_data.get('lesion_count', np.nan)],
            'lesion_total_volume': [mni_mask_data.get('lesion_total_volume', np.nan)],
            'lesion_volume': [mni_mask_data.get('lesion_volume', np.nan)]
        })

        df['Nom'] = nom
        df['Timepoint'] = timepoint
        df.set_index(['Nom', 'Timepoint'], inplace=True)
        dataframes.append(df)

    # Combiner tous les DataFrames
    df_final = pd.concat(dataframes)
    dif = calcul_diff_lesions(df_final)

    # Sauvegarde CSV
    dif.to_csv(args.output, index=False)
    print(f"\n✅ Résultats enregistrés dans : {args.output}\n")
    print(dif)

    # Statistiques globales
    print("\n--- Moyennes globales ---")
    print(dif.mean(numeric_only=True))
    print("\n--- Écarts-types globaux ---")
    print(dif.std(numeric_only=True))

    # Histogrammes individuels
    plot_diff_lesions_individual(dif, save_figs=args.save_figs)


if __name__ == "__main__":
    main()

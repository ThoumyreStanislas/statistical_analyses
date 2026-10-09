import pandas as pd
import argparse
import os

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Filtrer un CSV selon les bundles d'un second CSV.")
    parser.add_argument('--csv1', required=True, help="Chemin vers le premier fichier CSV (à filtrer)")
    parser.add_argument('--csv2', required=True, help="Chemin vers le second fichier CSV (contenant les bundles à garder)")
    parser.add_argument('--output', required=True, help="Chemin du fichier CSV filtré en sortie")
    return parser

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Lecture des deux fichiers CSV
    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    # Extraction des bundles du second fichier
    bundles_to_keep = df2['bundle'].unique()

    # Filtrage du premier fichier
    df_filtered = df1[df1['bundle'].isin(bundles_to_keep)]

    # Sauvegarde du fichier filtré
    df_filtered.to_csv(args.output, index=False)
    print(f"✅ Fichier filtré sauvegardé : {args.output}")
    print(f"Nombre de lignes conservées : {len(df_filtered)}")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import argparse

def _build_arg_parser():
    """Construit l'argument parser pour récupérer plusieurs fichiers .bval en entrée."""
    p = argparse.ArgumentParser(description="Analyse plusieurs fichiers .bval et détermine s'ils sont single-shell ou multi-shell.",
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('bval', nargs='+',  # Accepte plusieurs fichiers
                   help='Liste des fichiers .bval à analyser')

    return p

def classify_bval(bvals):
    """Détermine si les valeurs sont single-shell ou multi-shell en excluant celles ≤ 400."""
    unique_bvals = sorted(set(bvals))  # Valeurs uniques triées
    filtered_bvals = [b for b in unique_bvals if b > 400]  # Exclure bvals ≤ 400

    # Si toutes les valeurs sont ≤ 400 (ex: juste [0]), c'est single-shell
    if not filtered_bvals:  
        category = "single-shell"
    else:
        category = "single-shell" if len(filtered_bvals) == 1 else "multi-shell"

    return category, unique_bvals  # On retourne toutes les bvalues uniques pour référence

def process_bval_files(filepaths):
    """Analyse plusieurs fichiers .bval et retourne un DataFrame avec les infos."""
    data = []

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"⚠️ Erreur : Le fichier '{filepath}' n'existe pas. Il sera ignoré.")
            continue

        with open(filepath, "r") as f:
            bvals = list(map(int, f.read().split()))

        # Récupérer l'identifiant du sujet (ex: sub-002S0413_ses-init)
        filename = os.path.basename(filepath)
        parts = filename.split("_")
        subject = "_".join(parts[:2])  # Extrait "sub-002S0413_ses-init"
        session = parts[1]  # Extrait "ses-y2" ou autre

        category, unique_shells = classify_bval(bvals)

        # Stocker les résultats
        data.append({
            "subject": subject,
            "session": session,
            "category": category,
            "shells": unique_shells
        })

    # Créer un DataFrame
    df = pd.DataFrame(data)
    return df

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    df = process_bval_files(args.bval)

    if not df.empty:
        # Affichage du DataFrame
        print(df)

        # Sauvegarde en CSV
        df.to_csv("bval_analysis.csv", index=False)
        print("\n✅ Résultats sauvegardés dans 'bval_analysis.csv'.")

        # Nombre de sujets en multi-shell (global)
        num_multi_shell = (df["category"] == "multi-shell").sum()
        print(f"\n📌 Nombre total de sujets en multi-shell : {num_multi_shell}")

        # Nombre de sujets en multi-shell pour la session "ses-y2"
        num_multi_shell_y2 = ((df["category"] == "multi-shell") & (df["session"] == "ses-y2")).sum()
        print(f"📌 Nombre de sujets en multi-shell dans 'ses-y2' : {num_multi_shell_y2}")

if __name__ == "__main__":
    main()

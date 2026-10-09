import os
import json
import pandas as pd
import argparse

def _build_arg_parser():
    p = argparse.ArgumentParser(description="Preprocessing script for ComBat harmonization",
                                formatter_class=argparse.RawTextHelpFormatter)
     
    p.add_argument('--metadata', required=True,
                   help='Path to the metadata CSV file containing acquisition information of individuals')
    p.add_argument('--metrics_per_subjects', nargs='+', required=True,
                   help='List of JSON files containing diffusion values per section (output from tractometry_flow)')
    p.add_argument('--output', type=str,
                   help='output files in csv')
    return p


def extract_json_data(json_files):
    data = []

    for file_path in json_files:
        with open(file_path, 'r') as f:
            json_data = json.load(f)

        # Récupérer le nom du sujet (première clé)
        subject = list(json_data.keys())[0]  # Ex: "sub-168S6371_ses-sc"
        subject_clean = subject.replace("sub-", "")  # Nettoyage du nom

        # Parcours des bundles
        for bundle, metrics_dict in json_data[subject].items():
            # Parcours des métriques (ex: "mni_fa_metric")
            for metric, sections in metrics_dict.items():
                # Parcours des sections (ex: "001", "002", etc.)
                for section_id, values in sections.items():
                    # Nouveau comportement : gérer plusieurs formats de données
                    if isinstance(values, dict):
                        # Cas original avec mean
                        mean_value = values.get("mean", None)
                    elif isinstance(values, (int, float)):
                        # Valeur directe
                        mean_value = values
                    elif isinstance(values, list):
                        # Liste : on prend la moyenne
                        if values:
                            mean_value = sum(values) / len(values)
                        else:
                            mean_value = None
                    else:
                        # Type inconnu, ignorer
                        mean_value = None

                    bundles_section = f'{bundle}_{section_id}'

                    data.append({
                        'sid': subject_clean,
                        'bundle': bundles_section,
                        'mean': mean_value,
                        'metric': metric,
                    })

    return pd.DataFrame(data)


def generate_subject_name(row):
    session_mapping = {
        'ADNI3 Initial Visit-Cont Pt': 'ses-init',
        'ADNI Screening': 'ses-sc',
        'ADNI3 Year 1 Visit': 'ses-y1',
        'ADNI3 Year 2 Visit': 'ses-y2',
        'ADNI3 Year 3 Visit': 'ses-y3',
        'ADNI3 Year 4 Visit': 'ses-y4'
    }
    session = row['sessionIdentifier']
    subject = row['subjectIdentifier']
    subject = subject.replace('_', '')
    
    # Vérifie si la session existe dans le dictionnaire, sinon utilise une valeur par défaut
    session_suffix = session_mapping.get(session, 'ses-unknown')  # Utilise 'ses-unknown' si la session n'est pas trouvée
    return f"{subject}_{session_suffix}"

def merge_with_metadata(json_df, csv_metadata):
    """Merges extracted JSON data with selected metadata columns"""
    metadata_df = csv_metadata

    metadata_df = metadata_df.drop_duplicates(subset='sid', keep='first')

    # Vérifier que 'sid' est bien présent dans les deux DataFrames
    if 'sid' not in json_df.columns:
        raise KeyError("❌ ERREUR: 'sid' absent du DataFrame JSON")
    if 'sid' not in metadata_df.columns:
        raise KeyError("❌ ERREUR: 'sid' absent du DataFrame Metadata")

    merged_df = json_df.merge(metadata_df, on='sid', how='left')

    return merged_df

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    metadata_df = pd.read_csv(args.metadata, usecols=['subjectIdentifier', 'sessionIdentifier', 'subjectAge', 'subjectSex', 'siteKey'])

    # Extraction et fusion des données
    df_json = extract_json_data(args.metrics_per_subjects)

    metadata_df['sid'] = metadata_df.apply(generate_subject_name, axis=1)
    first_column = metadata_df.pop('sid')
    metadata_df = metadata_df.drop('subjectIdentifier', axis=1)
    metadata_df = metadata_df.drop('sessionIdentifier', axis=1)
    metadata_df = metadata_df.rename(columns={'subjectSex': 'sex', 'subjectAge': 'age', 'siteKey': 'site'})
    metadata_df['sex'] = metadata_df['sex'].map({'M': 1, 'F': 2}).astype(int)
    metadata_df.insert(0, 'sid', first_column)
    
    df_final = merge_with_metadata(df_json, metadata_df)
    df_final['disease'] = 'HC'
    df_final['handedness'] = 1


    print(df_final['site'].value_counts().idxmax())

    # Création des dossiers de sortie s'ils n'existent pas
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    df_final.to_csv("all_subject.csv", index=False)

    # Séparation des données par métrique
    for metric in df_final['metric'].unique():
        df_metric = df_final[df_final['metric'] == metric]

        # Sauvegarde par site pour chaque métrique
        for site in df_metric['site'].unique():
            df_site = df_metric[df_metric['site'] == site]


            # Nom du fichier de sortie
            output_filename = f"{output_dir}_{metric}_site_{site}.csv"
            df_site.to_csv(output_filename, index=False)
            print(f"✅ Fichier généré : {output_filename}")

if __name__ == "__main__":
    main()

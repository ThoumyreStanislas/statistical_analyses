import os
import json
import pandas as pd
import argparse

def _build_arg_parser():
    p = argparse.ArgumentParser(description="Preprocessing script for ComBat harmonization",
                                formatter_class=argparse.RawTextHelpFormatter)
     
    p.add_argument('--metadata', required=True,
                   help='Path to the metadata CSV file containing acquisition information of individuals')
    p.add_argument('--meta_nawm', required=True,
                   help='List of JSON files containing diffusion values per section (output from tractometry_flow)')
    p.add_argument('--output', type=str,
                   help='output files in csv')
    return p

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
    df_NAWM = pd.read_csv(args.meta_nawm)

    metadata_df['sid'] = metadata_df.apply(generate_subject_name, axis=1)
    first_column = metadata_df.pop('sid')
    metadata_df = metadata_df.drop('subjectIdentifier', axis=1)
    metadata_df = metadata_df.drop('sessionIdentifier', axis=1)
    metadata_df = metadata_df.rename(columns={'subjectSex': 'sex', 'subjectAge': 'age', 'siteKey': 'site'})
    metadata_df['sex'] = metadata_df['sex'].map({'M': 1, 'F': 2}).astype(int)
    metadata_df.insert(0, 'sid', first_column)
    print(metadata_df)
    
    df_final = merge_with_metadata(df_NAWM, metadata_df)
    df_final['disease'] = 'HC'

    print(df_final)

    # Création des dossiers de sortie s'ils n'existent pas
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    #df_final.to_csv("all_nawm.csv", index=False)

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

import os
import pandas as pd
import argparse

def _build_arg_parser():
    p = argparse.ArgumentParser(description="Preprocessing script for ComBat harmonization from CSV input",
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--metadata', required=True,
                   help='Path to the metadata CSV file containing acquisition information of individuals')
    p.add_argument('--input_csv', required=True,
                   help='CSV file already containing tractometry data (e.g. from previous JSON extraction)')
    p.add_argument('--output', type=str, required=True,
                   help='Path prefix for output CSV files')
    return p


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

    session_suffix = session_mapping.get(session, 'ses-unknown')
    return f"{subject}_{session_suffix}"

def merge_with_metadata(json_df, csv_metadata):
    metadata_df = csv_metadata.drop_duplicates(subset='sid', keep='first')

    if 'sid' not in json_df.columns:
        raise KeyError("❌ ERREUR: 'sid' absent du DataFrame d'entrée")
    if 'sid' not in metadata_df.columns:
        raise KeyError("❌ ERREUR: 'sid' absent du DataFrame Metadata")

    merged_df = json_df.merge(metadata_df, on='sid', how='left')
    return merged_df

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_df = pd.read_csv(args.input_csv)

    metadata_df = pd.read_csv(args.metadata, usecols=['subjectIdentifier', 'sessionIdentifier', 'subjectAge', 'subjectSex', 'siteKey'])
    metadata_df['sid'] = metadata_df.apply(generate_subject_name, axis=1)
    print(metadata_df['sid'])
    first_column = metadata_df.pop('sid')
    metadata_df = metadata_df.drop(columns=['subjectIdentifier', 'sessionIdentifier'])
    metadata_df = metadata_df.rename(columns={'subjectSex': 'sex', 'subjectAge': 'age', 'siteKey': 'site'})
    metadata_df['sex'] = metadata_df['sex'].map({'M': 1, 'F': 2}).astype(int)
    metadata_df.insert(0, 'sid', first_column)
    input_df['sid'] = input_df['sid'].str.replace('^sub-', '', regex=True)

    df_final = merge_with_metadata(input_df, metadata_df)
    print(df_final)

    # Ne pas écraser les colonnes existantes si elles sont déjà là
    if 'disease' not in df_final.columns:
        df_final['disease'] = 'HC'

    df_final = df_final[~df_final['bundle'].str.startswith(('MdLF_L', 'MdLF_R'))]

    df_final.to_csv("all_subject.csv", index=False)

    for metric in df_final['metric'].unique():
        df_metric = df_final[df_final['metric'] == metric]
        for site in df_metric['site'].unique():
            df_site = df_metric[df_metric['site'] == site]
            output_filename = f"{args.output}_{metric}_site_{site}.csv"
            df_site.to_csv(output_filename, index=False)
            print(f"✅ Fichier généré : {output_filename}")

if __name__ == "__main__":
    main()
import pandas as pd
import json
import argparse

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('data_T1',
                   help='Path of the volume data for timepoint 1')
    p.add_argument('data_T2',
                   help='Path of the volume data for timepoint 2')
    p.add_argument('--extract_mean_std', action='store_true',
                   help='If set, extracts mean and std for metrics; otherwise extracts raw values.')


    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Charger les fichiers JSON
    with open(args.data_T1, "r") as file1:
        session_1_data = json.load(file1)

    with open(args.data_T2, "r") as file2:
        session_2_data = json.load(file2)

    # Fonction pour extraire les métriques d'un DataFrame
    def extract_metrics(data, session_label, extract_mean_std=True):
        metrics = []
        unique_metrics = set()

        for subject, subject_data in data.items():
            for bundle, bundle_data in subject_data.items():
                subject_id = '_'.join(subject.split('_')[:-1])
                metrics_data = {
                    'subject': subject_id,
                    'session': session_label,
                    'bundle': bundle
                }
                for metric, values in bundle_data.items():
                    unique_metrics.add(metric)

                    if extract_mean_std and isinstance(values, dict):  # Si on veut mean/std et les valeurs sont un dictionnaire
                        metrics_data[f'{metric}_mean'] = values.get('mean', None)
                        metrics_data[f'{metric}_std'] = values.get('std', None)
                    else :  # Si on ne veut que les valeurs brutes
                        metrics_data[f'{metric}'] = values         
                metrics.append(metrics_data)

        return pd.DataFrame(metrics), unique_metrics

    # Extraire les métriques des deux sessions
    metrics_1, metrics_set_1 = extract_metrics(session_1_data, 'session_1', extract_mean_std=args.extract_mean_std)
    metrics_2, metrics_set_2 = extract_metrics(session_2_data, 'session_2', extract_mean_std=args.extract_mean_std)
    unique_metrics = sorted(metrics_set_1.union(metrics_set_2))

    # Obtenir toutes les combinaisons possibles de sujets et de bundles
    all_subjects = pd.concat([metrics_1['subject'], metrics_2['subject']]).unique()
    all_bundles = pd.concat([metrics_1['bundle'], metrics_2['bundle']]).unique()
    reference_table = pd.DataFrame([(subject, bundle) for subject in all_subjects for bundle in all_bundles],
                                    columns=['subject', 'bundle'])

    # Fusionner les deux DataFrames
    merged_df = pd.merge(metrics_1, metrics_2, on=['subject', 'bundle'], how='outer', suffixes=('_session_1', '_session_2'))

    # Étendre le DataFrame pour inclure toutes les combinaisons
    merged_df = pd.merge(reference_table, merged_df, on=['subject', 'bundle'], how='left')

    # Calculer les différences des moyennes pour chaque métrique
    if args.extract_mean_std:
        for metric in unique_metrics:
            session_1_col = f'{metric}_mean_session_1'
            session_2_col = f'{metric}_mean_session_2'
            diff_col = f'{metric}_mean_diff'
            merged_df[diff_col] = abs(merged_df[session_1_col] - merged_df[session_2_col])
    else :
        for metric in unique_metrics:
            session_1_col = f'{metric}_session_1'
            session_2_col = f'{metric}_session_2'
            diff_col = f'{metric}_mean_diff'
            merged_df[diff_col] = abs(merged_df[session_1_col] - merged_df[session_2_col])

    # Moyenne des différences par bundle (exclusion implicite des NaN)
    mean_differences = merged_df.groupby('bundle').mean(numeric_only=True)

    # Filtrer pour ne conserver que les colonnes *_mean_diff
    if args.extract_mean_std:
        mean_differences = mean_differences[[col for col in mean_differences.columns if col.endswith('_mean_diff')]]

    # Trier et afficher les résultats
    for metric in unique_metrics:
        diff_col = f'{metric}_mean_diff'
        if diff_col in mean_differences.columns:
            sorted_diffs = mean_differences.sort_values(by=diff_col, ascending=False)
            print(f"\nTop bundles for metric {metric}:")
            print(sorted_diffs[diff_col].head())  # Affiche les 5 premiers bundles avec la plus grande différence
            # Sauvegarder dans un fichier CSV
            sorted_diffs.to_csv(f"{metric}_mean_diff.csv")

if __name__ == "__main__":
    main()

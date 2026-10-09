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
    p.add_argument('--per_point', action='store_true',
                   help='If set, processes JSON files with per-point structure.')

    return p

  # Fonction pour extraire les métriques d'un DataFrame
def extract_metrics(data, session_label, extract_mean_std=True, per_point=False):
    metrics = []
    unique_metrics = set()

    for bundle, bundle_data in data.items():
        for metric, metric_data in bundle_data.items():
            if per_point:  # Structure avec "per-point"
                for section, values in metric_data.items():
                    # Traiter les cas où "mean" et "std" sont des listes
                    if extract_mean_std:
                        if isinstance(values["mean"], list):
                            for i, (mean_value, std_value) in enumerate(zip(values["mean"], values["std"])):
                                metrics_data = {
                                    'bundle': bundle,
                                    'metric': metric,
                                    'section': section,
                                    'point': i + 1,  # Index du point
                                    'session': session_label,
                                    f'{metric}_mean': mean_value,
                                    f'{metric}_std': std_value
                                }
                                metrics.append(metrics_data)
                        else:  # Cas où "mean" et "std" ne sont pas des listes
                            metrics_data = {
                                'bundle': bundle,
                                'metric': metric,
                                'section': section,
                                'point': section,  # Utiliser la section comme point
                                'session': session_label,
                                f'{metric}_mean': values["mean"],
                                f'{metric}_std': values["std"]
                            }
                            metrics.append(metrics_data)
                    else:  # Valeurs brutes uniquement (pas de mean/std)
                        metrics_data = {
                            'bundle': bundle,
                            'metric': metric,
                            'section': section,
                            'point': section,
                            'session': session_label,
                            f'{metric}': values
                        }
                        metrics.append(metrics_data)
                    unique_metrics.add(metric)
            else:  # Structure classique
                for section, values in metric_data.items():
                    metrics_data = {
                        'bundle': bundle,
                        'metric': metric,
                        'section': section,
                        'session': session_label
                    }
                    if extract_mean_std and isinstance(values, dict):
                        metrics_data[f'{metric}_mean'] = values.get('mean', None)
                        metrics_data[f'{metric}_std'] = values.get('std', None)
                    else:
                        metrics_data[f'{metric}'] = values
                    metrics.append(metrics_data)
                    unique_metrics.add(metric)

    return pd.DataFrame(metrics), unique_metrics

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Charger les fichiers JSON
    with open(args.data_T1, "r") as file1:
        session_1_data = json.load(file1)

    with open(args.data_T2, "r") as file2:
        session_2_data = json.load(file2)

    # Extraire les métriques des deux sessions
    metrics_1, metrics_set_1 = extract_metrics(session_1_data, 'session_1', 
                                               extract_mean_std=args.extract_mean_std, 
                                               per_point=args.per_point)
    metrics_2, metrics_set_2 = extract_metrics(session_2_data, 'session_2', 
                                               extract_mean_std=args.extract_mean_std, 
                                               per_point=args.per_point)
    unique_metrics = sorted(metrics_set_1.union(metrics_set_2))

    # Obtenir toutes les combinaisons possibles de bundles et de sections
    all_bundles = pd.concat([metrics_1['bundle'], metrics_2['bundle']]).unique()
    reference_table = pd.DataFrame({'bundle': all_bundles})

    # Fusionner les deux DataFrames
    merged_df = pd.merge(metrics_1, metrics_2, on=['bundle', 'metric', 'section', 'point'], how='outer', suffixes=('_session_1', '_session_2'))

    # Calculer les différences des moyennes pour chaque métrique
    if args.extract_mean_std:
        for metric in unique_metrics:
            session_1_col = f'{metric}_mean_session_1'
            session_2_col = f'{metric}_mean_session_2'
            diff_col = f'{metric}_mean_diff'
            merged_df[diff_col] = abs(merged_df[session_1_col] - merged_df[session_2_col])
    else:
        for metric in unique_metrics:
            session_1_col = f'{metric}_session_1'
            session_2_col = f'{metric}_session_2'
            diff_col = f'{metric}_diff'
            merged_df[diff_col] = abs(merged_df[session_1_col] - merged_df[session_2_col])

    # Moyenne des différences par bundle (exclusion implicite des NaN)
    mean_differences = merged_df.groupby('bundle').mean(numeric_only=True)

    # Filtrer pour ne conserver que les colonnes *_mean_diff
    if args.extract_mean_std:
        mean_differences = mean_differences[[col for col in mean_differences.columns if col.endswith('_mean_diff')]]

    # Trier et afficher les résultats
    for metric in unique_metrics:
        diff_col = f'{metric}_mean_diff' if args.extract_mean_std else f'{metric}_diff'
        if diff_col in mean_differences.columns:
            sorted_diffs = mean_differences.sort_values(by=diff_col, ascending=False)
            print(f"\nTop bundles for metric {metric}:")
            print(sorted_diffs[diff_col].head())  # Affiche les 5 premiers bundles avec la plus grande différence
            # Sauvegarder dans un fichier CSV
            sorted_diffs.to_csv(f"{metric}_diff.csv" if not args.extract_mean_std else f"{metric}_mean_diff.csv")

if __name__ == "__main__":
    main()

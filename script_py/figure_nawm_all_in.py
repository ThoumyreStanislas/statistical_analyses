import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import numpy as np
import argparse


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--csv_T1', required=True,
                   help='Path to the label lesion for timepoint 1')
    p.add_argument('--csv_T2', required=True,
                   help='Path to the label lesion for timepoint 2')
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    df1 = pd.read_csv(args.csv_T1)
    df2 = pd.read_csv(args.csv_T2)

    df1['dataset'] = 'DF1'
    df2['dataset'] = 'DF2'

    # Fusion
    df_all = pd.concat([df1, df2], ignore_index=True)

    # Garder uniquement certaines métriques
    metrics_to_include = ['fa']  # modifiable
    df_filtered = df_all[df_all['metric'].isin(metrics_to_include)]

    # Extraire le numéro du point NAWM
    df_filtered.loc[:, 'nawm_point'] = df_filtered['bundle'].str.extract(r'nawm_(\d+)').astype(float)

    # Grouper pour la moyenne/écart-type
    df_plot = df_filtered.groupby(['nawm_point', 'metric', 'dataset'])['mean'].agg(['mean', 'std']).reset_index()

    # Graphique par métrique
    for metric in metrics_to_include:
        df_metric = df_plot[df_plot['metric'] == metric]
        df_sub = df_filtered[df_filtered['metric'] == metric]

        plt.figure(figsize=(16, 6))
        ax = sns.barplot(
            data=df_metric,
            x='nawm_point',
            y='mean',
            hue='dataset',
            palette='pastel',
            ci=None,
            errorbar=None
        )

        # Barres d’erreur
        for i, row in df_metric.iterrows():
            plt.errorbar(
                x=row['nawm_point'] + (-0.2 if row['dataset'] == 'DF1' else 0.2),
                y=row['mean'],
                yerr=row['std'],
                fmt='none',
                c='black',
                capsize=3
            )

        # Tests t pour chaque point NAWM
        p_values = {}
        for pt in sorted(df_filtered['nawm_point'].unique()):
            group1 = df_sub[(df_sub['nawm_point'] == pt) & (df_sub['dataset'] == 'DF1')]['mean']
            group2 = df_sub[(df_sub['nawm_point'] == pt) & (df_sub['dataset'] == 'DF2')]['mean']
            if len(group1) > 1 and len(group2) > 1:
                stat, p = ttest_ind(group1, group2, equal_var=False)
                p_values[pt] = p
            else:
                p_values[pt] = np.nan

        # Annoter les p-values
        y_max = df_metric['mean'].max() + df_metric['std'].max() + 0.02
        for pt, p in p_values.items():
            if not np.isnan(p):
                if p < 0.001:
                    label = '***'
                elif p < 0.01:
                    label = '**'
                elif p < 0.05:
                    label = '*'
                else:
                    label = 'ns'
                plt.text(pt, y_max, label, ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.title(f'Mean {metric.upper()} value per NAWM point\n(* = p<0.05, ** = p<0.01, *** = p<0.001)')
        plt.xlabel('NAWM Point')
        plt.ylabel(f'{metric.upper()} Mean')
        plt.xticks(sorted(df_metric['nawm_point'].unique()))
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.legend(title='Dataset')

        # Sauvegarde
        plt.savefig(f'mean_{metric}_per_nawm_point_with_stats.png', dpi=300)

if __name__ == "__main__":
    main()
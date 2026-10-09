import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

def extract_subject_id(path):
    """Extrait uniquement l'identifiant du sujet sans la session (ex: sub-123S4567)."""
    basename = os.path.basename(path)
    for part in basename.split('_'):
        if part.startswith("sub-"):
            return part
    return None

def load_nifti_data(path):
    return nib.load(path).get_fdata()

def analyze_nawm(nawm_T1_paths, nawm_T2_paths, metric_T1_paths, metric_T2_paths):
    results = []

    for nawm_T1_path in nawm_T1_paths:
        subj_id = extract_subject_id(nawm_T1_path)
        print(f"Processing subject: {subj_id}")

        try:
            nawm_T2_path = next(p for p in nawm_T2_paths if subj_id in p)
            metric_T1_path = next(p for p in metric_T1_paths if subj_id in p)
            metric_T2_path = next(p for p in metric_T2_paths if subj_id in p)
        except StopIteration:
            print(f"⚠️ Données manquantes pour le sujet {subj_id}")
            continue

        nawm_T1 = load_nifti_data(nawm_T1_path)
        nawm_T2 = load_nifti_data(nawm_T2_path)
        metric_T1 = load_nifti_data(metric_T1_path)
        metric_T2 = load_nifti_data(metric_T2_path)

        metric_T1_nawm = metric_T1[nawm_T1 > 0]
        metric_T2_nawm = metric_T2[nawm_T2 > 0]

        results.append({
            'subject': subj_id,
            'timepoint': 'T1',
            'metric_mean': np.mean(metric_T1_nawm),
            'metric_std': np.std(metric_T1_nawm)
        })
        results.append({
            'subject': subj_id,
            'timepoint': 'T2',
            'metric_mean': np.mean(metric_T2_nawm),
            'metric_std': np.std(metric_T2_nawm)
        })

    return pd.DataFrame(results)


def plot_nawm(df, output_svg="nawm_metrics_boxplot.svg"):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='timepoint', y='metric_mean')
    sns.stripplot(data=df, x='timepoint', y='metric_mean', color='black', size=5, alpha=0.7, jitter=True)
    plt.title("Distribution des moyennes de la métrique dans la NAWM par sujet")
    plt.ylabel("Moyenne de la métrique")
    plt.xlabel("Timepoint")
    plt.tight_layout()
    plt.savefig(output_svg, format='svg')
    print(f"✅ Figure sauvegardée sous : {output_svg}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse des métriques dans la NAWM entre deux timepoints")
    parser.add_argument("--nawm_T1", nargs='+', required=True, help="Liste des fichiers NAWM à T1")
    parser.add_argument("--nawm_T2", nargs='+', required=True, help="Liste des fichiers NAWM à T2")
    parser.add_argument("--metric_T1", nargs='+', required=True, help="Liste des fichiers de métriques à T1")
    parser.add_argument("--metric_T2", nargs='+', required=True, help="Liste des fichiers de métriques à T2")
    args = parser.parse_args()

    df_nawm = analyze_nawm(args.nawm_T1, args.nawm_T2, args.metric_T1, args.metric_T2)
    
    # Sauvegarde du DataFrame
    csv_path = "nawm_metrics.csv"
    df_nawm.to_csv(csv_path, index=False)
    print(f"✅ Données sauvegardées sous : {csv_path}")

    # Génération et sauvegarde de la figure
    plot_nawm(df_nawm)

import numpy as np
import nibabel as nib
import os
import re
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.stats import kruskal, ttest_rel

warnings.filterwarnings("ignore")


# ============================================================
# ARGUMENT PARSER
# ============================================================
def _build_arg_parser():
    p = argparse.ArgumentParser(description="Analyse des métriques de lésions et NAWM sur deux temps (T1, T2)",
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--label_T1', nargs='+', required=True, help='Path to the label lesion for timepoint 1')
    p.add_argument('--label_T2', nargs='+', required=True, help='Path to the label lesion for timepoint 2')
    p.add_argument('--metric_T1', nargs='+', required=True, help='Paths to all metric data for timepoint 1')
    p.add_argument('--metric_T2', nargs='+', required=True, help='Paths to all metric data for timepoint 2')
    p.add_argument('--nawm_T1', nargs='+', required=True, help='Path to the NAWM map for timepoint 1')
    p.add_argument('--nawm_T2', nargs='+', required=True, help='Path to the NAWM map for timepoint 2')
    p.add_argument('--template', required=False, help='Path to the template metric for projection (optional)')
    p.add_argument('--metadata', required=False, help='Path to the metadata csv (optional)')
    return p


# ============================================================
# UTILITAIRES
# ============================================================
def load_nifti(file_path):
    return nib.load(file_path).get_fdata()


def pair_files(label_t1_files, label_t2_files, metrics_t1_files, metrics_t2_files, nawm_t1_files, nawm_t2_files):
    subjects = set([os.path.basename(f).split('_')[0] for f in label_t1_files])
    pairs = []
    for subject in subjects:
        label_t1 = [f for f in label_t1_files if subject in os.path.basename(f)]
        label_t2 = [f for f in label_t2_files if subject in os.path.basename(f)]
        metrics_t1 = [f for f in metrics_t1_files if subject in os.path.basename(f)]
        metrics_t2 = [f for f in metrics_t2_files if subject in os.path.basename(f)]
        nawm_t1 = [f for f in nawm_t1_files if subject in os.path.basename(f)]
        nawm_t2 = [f for f in nawm_t2_files if subject in os.path.basename(f)]
        if label_t1 and label_t2 and metrics_t1 and metrics_t2 and nawm_t1 and nawm_t2:
            pairs.append((label_t1[0], label_t2[0], metrics_t1[0], metrics_t2[0], nawm_t1[0], nawm_t2[0]))
    return pairs


# ============================================================
# CALCUL DES METRIQUES
# ============================================================
def calculate_metrics(label_t1, label_t2, metrics_t1, metrics_t2, nawm_t1, nawm_t2, template=None, metadata=None):
    label_t1_data = load_nifti(label_t1)
    label_t2_data = load_nifti(label_t2)
    metrics_t1_data = load_nifti(metrics_t1)
    metrics_t2_data = load_nifti(metrics_t2)
    nawm_t1_data = load_nifti(nawm_t1)
    nawm_t2_data = load_nifti(nawm_t2)

    if template:
        template_data = load_nifti(template)

    subjects = os.path.basename(label_t1).split('_')[0]
    formatted_names = re.sub(r"sub-(\d{3})S(\d+)", r"\1_S_\2", subjects)

    # Chargement des métadonnées
    if metadata:
        df_metadata = pd.read_csv(metadata)
        df_metadata_subject = df_metadata[df_metadata['subjectIdentifier'] == formatted_names]
        site_key = df_metadata_subject['siteKey'].unique()[0] if not df_metadata_subject.empty else "NA"
        subject_sex = df_metadata_subject['subjectSex'].unique()[0] if not df_metadata_subject.empty else "NA"
    else:
        site_key = "NA"
        subject_sex = "NA"

    # Détection des lésions
    unique_t1 = np.unique(label_t1_data[label_t1_data > 0])
    unique_t2 = np.unique(label_t2_data[label_t2_data > 0])

    new_lesions = np.setdiff1d(unique_t2, unique_t1)
    new_metric_values = np.mean([
        metrics_t2_data[label_t2_data == lesion].mean() for lesion in new_lesions
    ]) if len(new_lesions) > 0 else np.nan
    new_metric_projected_t1 = np.mean([
        metrics_t1_data[label_t2_data == lesion].mean() for lesion in new_lesions
    ]) if len(new_lesions) > 0 else np.nan
    new_metric_template = np.mean([
        template_data[label_t2_data == lesion].mean() for lesion in new_lesions
    ]) if template and len(new_lesions) > 0 else np.nan
    new_metric_differences = new_metric_projected_t1 - new_metric_values if not np.isnan(new_metric_projected_t1) else np.nan

    disappeared_lesions = np.setdiff1d(unique_t1, unique_t2)
    common_lesions = np.intersect1d(unique_t1, unique_t2)

    common_metric_values = [
        (metrics_t1_data[label_t1_data == lesion].mean(), metrics_t2_data[label_t2_data == lesion].mean())
        for lesion in common_lesions
    ]
    if len(common_metric_values) > 0:
        common_metric_t1, common_metric_t2 = zip(*common_metric_values)
        common_metric_values_mean = (np.mean(common_metric_t1), np.mean(common_metric_t2))
    else:
        common_metric_values_mean = (np.nan, np.nan)

    common_metric_differences = np.mean([
        t1 - t2 for t1, t2 in common_metric_values
    ]) if len(common_metric_values) > 0 else np.nan
    common_metric_template = np.mean([
        template_data[label_t1_data == lesion].mean() for lesion in common_lesions
    ]) if template and len(common_lesions) > 0 else np.nan

    # NAWM
    nawm_labels = np.unique(nawm_t1_data[nawm_t1_data > 0])
    distance_nawm = [0, 2, 4, 6, 8, 10, 12]
    nawm_metric_t1 = {label: metrics_t1_data[nawm_t1_data == label].mean() for label in nawm_labels}
    nawm_metric_t2 = {label: metrics_t2_data[nawm_t2_data == label].mean() for label in nawm_labels}
    nawm_metric_template = {label: template_data[nawm_t1_data == label].mean() for label in nawm_labels} if template else {}

    result_data = {
        "subject": [formatted_names],
        "site": [site_key],
        "sex": [subject_sex],
        "new_lesions_count": [len(new_lesions)],
        "disappeared_lesions_count": [len(disappeared_lesions)],
        "new_metric_values": [new_metric_values],
        "new_metric_projected_t1": [new_metric_projected_t1],
        "new_metric_differences": [new_metric_differences],
        "new_metric_template": [new_metric_template],
        "common_metric_values": [common_metric_values_mean],
        "common_metric_differences": [common_metric_differences],
        "common_metric_template": [common_metric_template],
        "nawm_distances": [distance_nawm],
        "nawm_metric_t1": [nawm_metric_t1],
        "nawm_metric_t2": [nawm_metric_t2],
        "nawm_metric_template": [nawm_metric_template],
    }
    return pd.DataFrame(result_data)


# ============================================================
# COHEN'S D
# ============================================================
def cohen_d(x, y):
    """Calcule le Cohen's d pour échantillons appariés."""
    x, y = np.array(x), np.array(y)
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else np.nan


def compute_cohen_d_by_distance(result_df):
    records = []
    for metric in sorted(result_df["metric_name"].unique()):
        subset = result_df[result_df["metric_name"] == metric]
        for _, row in subset.iterrows():
            nawm_t1 = row["nawm_metric_t1"]
            nawm_t2 = row["nawm_metric_t2"]
            distances = row["nawm_distances"]
            for d in distances:
                if d in nawm_t1 and d in nawm_t2:
                    d_value = cohen_d([nawm_t1[d]], [nawm_t2[d]])
                    records.append({
                        "subject": row["subject"],
                        "site": row["site"],
                        "metric": metric,
                        "distance": d,
                        "cohen_d": d_value
                    })
    return pd.DataFrame(records)


# ============================================================
# PLOT DES METRIQUES ET COHEN'S D
# ============================================================
def plot_metrics_and_cohen(result_df, output_svg="metrics_and_cohen.svg"):
    metrics_left = ["fa-t", "md-t", "ad-t"]
    metrics_right = ["rd-t", "fw", "cohen"]

    df_cohen = compute_cohen_d_by_distance(result_df)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes = axes.flatten()

    def plot_metric(ax, metric_name):
        subset_df = result_df[result_df["metric_name"] == metric_name]
        nawm_data = {"Distance": [], "Metric": [], "Timepoint": [], "Subject": []}
        for _, res in subset_df.iterrows():
            for dist, m_t1, m_t2 in zip(res["nawm_distances"],
                                        res["nawm_metric_t1"].values(),
                                        res["nawm_metric_t2"].values()):
                nawm_data["Distance"] += [dist, dist]
                nawm_data["Metric"] += [m_t1, m_t2]
                nawm_data["Timepoint"] += ["T1", "T2"]
                nawm_data["Subject"] += [res["subject"], res["subject"]]
        df = pd.DataFrame(nawm_data)
        sns.boxplot(x="Distance", y="Metric", hue="Timepoint", data=df,
                    palette="coolwarm", width=0.7, ax=ax, fliersize=2)
        ax.set_title(metric_name.upper(), fontsize=14)
        ax.set_xlabel("Distance (mm)")
        ax.set_ylabel(metric_name.upper())
        ax.legend_.remove()

    # Plot des métriques FA, MD, AD, RD, FW
    for i, metric in enumerate(metrics_left + metrics_right[:-1]):
        plot_metric(axes[i], metric)

    # Dernier subplot = Cohen's d
    ax = axes[-1]
    sns.boxplot(x="distance", y="cohen_d", data=df_cohen,
                palette="viridis", width=0.7, ax=ax)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Cohen's d (T1 vs T2)", fontsize=14)
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Cohen's d")

    plt.tight_layout()
    plt.savefig(output_svg, dpi=300)
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    metric_pattern = re.compile(r'_([a-z0-9]+)_mni', re.IGNORECASE)
    metric_names = set()

    for f in args.metric_T1:
        base = os.path.basename(f)
        m = metric_pattern.search(base)
        if m:
            metric_names.add(m.group(1).lower())

    all_results = []
    for metric in sorted(metric_names):
        print(f"Processing metric: {metric}")
        metric_t1_files = [f for f in args.metric_T1 if f"_{metric}_" in f.lower()]
        metric_t2_files = [f for f in args.metric_T2 if f"_{metric}_" in f.lower()]
        pairs = pair_files(args.label_T1, args.label_T2, metric_t1_files, metric_t2_files, args.nawm_T1, args.nawm_T2)

        for pair in pairs:
            df_metric = calculate_metrics(*pair, args.template, args.metadata)
            if metric in ["ad", "md", "rd", "fa"]:
                df_metric["metric_name"] = metric + "-t"
            else:
                df_metric["metric_name"] = metric
            all_results.append(df_metric)

    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv("all_metrics_results.csv", index=False)
    plot_metrics_and_cohen(result_df)


if __name__ == "__main__":
    main()

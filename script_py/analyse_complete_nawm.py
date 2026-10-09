import argparse
import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

def _build_arg_parser():
    p = argparse.ArgumentParser(description="Analyse globale des métriques de diffusion dans les lésions et NAWM entre T1 et T2")
    p.add_argument('--label_T1', nargs='+', required=True, help="Fichiers labels T1")
    p.add_argument('--label_T2', nargs='+', required=True, help="Fichiers labels T2")
    p.add_argument('--metric_T1', nargs='+', required=True, help="Fichiers métriques T1")
    p.add_argument('--metric_T2', nargs='+', required=True, help="Fichiers métriques T2")
    p.add_argument('--out_dir', required=True, help="Répertoire de sortie pour les résultats")
    return p

def extract_subject_id(path):
    match = re.search(r"(sub-\d+S\d+)_ses-(sc|y2|init|y4|y3|y1)", path)
    if match:
        return match.group(1)
    return None

def extract_region_values(label_data, metric_data, region_id):
    mask = label_data == region_id
    values = metric_data[mask]
    return values[~np.isnan(values)]

def group_metric_files_by_subject(file_list, metrics_names):
    grouped = {}
    for f in file_list:
        subj_id = extract_subject_id(f)
        if not subj_id:
            continue
        for metric in metrics_names:
            if re.search(fr"_{metric}(\.|_)", f, re.IGNORECASE):
                grouped.setdefault(subj_id, {})[metric] = f
    return grouped

def cohens_d(x, y):
    diff = x - y
    return diff.mean() / diff.std(ddof=1)

def build_df_metrics(args):
    metrics_names = ['fa', 'md', 'rd', 'ad', 'fw']  # métriques DTI

    metrics_T1_grouped = group_metric_files_by_subject(args.metric_T1, metrics_names)
    metrics_T2_grouped = group_metric_files_by_subject(args.metric_T2, metrics_names)

    label_T1_grouped = {extract_subject_id(f): f for f in args.label_T1 if extract_subject_id(f)}
    label_T2_grouped = {extract_subject_id(f): f for f in args.label_T2 if extract_subject_id(f)}

    distances = list(range(0, 14, 2))  # ex: 0,2,...12 mm

    records = []

    for subj_id in metrics_T1_grouped.keys():
        if subj_id not in metrics_T2_grouped:
            continue
        if subj_id not in label_T1_grouped or subj_id not in label_T2_grouped:
            continue

        label_T1 = nib.load(label_T1_grouped[subj_id]).get_fdata()
        label_T2 = nib.load(label_T2_grouped[subj_id]).get_fdata()

        for metric in metrics_names:
            if metric not in metrics_T1_grouped[subj_id] or metric not in metrics_T2_grouped[subj_id]:
                continue

            metric_T1 = nib.load(metrics_T1_grouped[subj_id][metric]).get_fdata()
            metric_T2 = nib.load(metrics_T2_grouped[subj_id][metric]).get_fdata()

            for dist in distances:
                vals_T1 = extract_region_values(label_T1, metric_T1, dist)
                vals_T2 = extract_region_values(label_T2, metric_T2, dist)

                if len(vals_T1) == 0 or len(vals_T2) == 0:
                    continue

                mean_T1 = np.mean(vals_T1)
                mean_T2 = np.mean(vals_T2)

                records.append({
                    'subject': subj_id,
                    'metric': metric,
                    'distance_mm': dist,
                    'T1': mean_T1,
                    'T2': mean_T2,
                    'difference': mean_T2 - mean_T1
                })

    df = pd.DataFrame(records)
    return df

def analyse_stats(df_metrics):
    stats_records = []
    for (metric, dist), group in df_metrics.groupby(['metric', 'distance_mm']):
        t_res = ttest_rel(group['T2'], group['T1'], nan_policy='omit')
        cohen_d_val = cohens_d(group['T2'], group['T1'])
        stats_records.append({
            'metric': metric,
            'distance_mm': dist,
            't_value': t_res.statistic,
            'p_value': t_res.pvalue,
            'cohen_d': cohen_d_val
        })
    stats_df = pd.DataFrame(stats_records)
    _, fdrs, _, _ = multipletests(stats_df['p_value'], method='fdr_bh')
    stats_df['p_fdr'] = fdrs
    return stats_df

def plot_all_panels(df_results, out_svg, palette=None):
    sns.set(style="whitegrid")
    metrics = sorted(df_results["metric"].unique())
    if palette is None:
        colors = sns.color_palette("tab10", n_colors=len(metrics))
        palette = dict(zip(metrics, colors))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    ax_t, ax_p, ax_d, ax_fdr = axes.flatten()

    def plot_on(ax, ycol, ylabel, logscale=False, ylim=None):
        for metric in metrics:
            sub = df_results[df_results["metric"] == metric].sort_values("distance_mm")
            if sub.empty:
                continue
            ax.plot(sub["distance_mm"], sub[ycol], marker='o', label=metric, color=palette[metric])
        ax.set_xlabel("Distance from lesion (mm)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(df_results["distance_mm"].unique()))
        if logscale:
            ax.set_yscale('log')
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")

    plot_on(ax_t, "t_value", "t-value")
    plot_on(ax_p, "p_value", "p-value")
    plot_on(ax_d, "cohen_d", "Cohen's d")
    plot_on(ax_fdr, "p_fdr", "FDR-corrected p-value")

    fig.suptitle("DTI statistics by distance (paired y0 → y2)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_svg, format='svg', dpi=300)
    plt.close(fig)

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Building df_metrics...")
    df_metrics = build_df_metrics(args)
    if df_metrics.empty:
        raise RuntimeError("No data collected. Check that labels/metrics exist and distances match label values.")

    print("Running statistics...")
    df_results = analyse_stats(df_metrics)

    out_table = os.path.join(args.out_dir, "dti_distance_stats.csv")
    df_results.to_csv(out_table, index=False)
    print("Saved results table to", out_table)

    out_svg = os.path.join(args.out_dir, "dti_distance_stats_panels.svg")
    plot_all_panels(df_results, out_svg)
    print("Saved combined SVG to", out_svg)

if __name__ == "__main__":
    main()

Vous avez dit :
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse FP-Growth avec comparaison à la moyenne populationnelle,
analyse par bundle et profils (heatmaps lift / cooccurrence).

Organisation des outputs :
📁 global_analyses/
│   ├── graphs/
│   └── tables/
📁 bundles/
│   ├── graphs/
│   └── tables/
📁 profils/
│   ├── graphs/
│   └── tables/
"""

import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from scipy.stats import ttest_1samp


# ===============================================================
# --- Parser
# ===============================================================
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Analyse FP-Growth et profils multi-trend",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument('metrics_csv', help='Fichier CSV des métriques')
    p.add_argument('--p_threshold', type=float, default=0.05,
                   help='Seuil de significativité pour la comparaison à la moyenne')
    return p


# ===============================================================
# --- Répertoires de sortie
# ===============================================================
def ensure_dirs():
    dirs = [
        "global_analyses/graphs",
        "global_analyses/tables",
        "bundles/graphs",
        "bundles/tables",
        "profils/graphs",
        "profils/tables"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ===============================================================
# --- Comparaison à la population
# ===============================================================
def compare_to_population_by_bundle(series, metric, population_df, p_threshold):
    trends = {}
    rel_changes = {}

    if not series.dropna().empty:
        population_df.loc[series.dropna().index, metric] = series.dropna()
    else:
        return pd.DataFrame(columns=["Trend", "RelativeChange(%)"])

    subj = series.index.get_level_values('Subject').unique()[0]
    bundle = series.index.get_level_values('Bundle').unique()[0]
    timepoint = series.index.get_level_values('Timepoint').unique()[0]
    sections = series.index.get_level_values("Section").unique().sort_values()

    for seg in sections:
        val_indiv = series.xs(seg, level="Section").mean()

        if metric == "lesion_total_volume":
            trends[seg] = "lesion_presente" if val_indiv > 0 else "aucune_lesion"
            rel_changes[seg] = np.nan
            continue

        try:
            pop_subdf = population_df.xs((bundle, seg, timepoint),
                                         level=("Bundle", "Section", "Timepoint"))
        except KeyError:
            trends[seg] = "NA"
            rel_changes[seg] = np.nan
            continue

        non_lesion_mask = (pop_subdf["lesion_total_volume"] == 0)
        ref_values = pop_subdf.loc[non_lesion_mask, metric].dropna()
        ref_mean = ref_values.mean()

        if len(ref_values) == 0 or pd.isna(val_indiv):
            trends[seg] = "NA"
            rel_changes[seg] = np.nan
            continue

        t_stat, p_val = ttest_1samp(ref_values, popmean=val_indiv, nan_policy='omit')

        if p_val < p_threshold:
            trends[seg] = "above" if val_indiv > ref_mean else "below"
        else:
            trends[seg] = "equal"

        rel_changes[seg] = ((val_indiv - ref_mean) / abs(ref_mean)) * 100 if ref_mean != 0 else np.nan

    return pd.DataFrame({
        "Trend": pd.Series(trends),
        "RelativeChange(%)": pd.Series(rel_changes)
    })


# ===============================================================
# --- Calcul utilitaire du lift et cooccurrence
# ===============================================================
def compute_lift_and_pct(n_both, n_antecedent, n_consequent, n_total):
    if n_total == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    sup_both = n_both / n_total
    sup_A = n_antecedent / n_total
    sup_C = n_consequent / n_total

    if sup_A == 0 or sup_C == 0:
        lift = np.nan
    else:
        lift = sup_both / (sup_A * sup_C)

    pct_total = (n_both / n_total) * 100
    pct_given_lesion = (n_both / n_antecedent * 100) if (n_antecedent > 0) else np.nan
    confidence = n_both / n_antecedent if (n_antecedent > 0) else np.nan

    return (lift, pct_total, pct_given_lesion, confidence)

# ===============================================================
# --- MAIN
# ===============================================================
def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    ensure_dirs()

    df = pd.read_csv(args.metrics_csv, index_col=[0, 1, 2, 3])
    df.index.names = ['Subject', 'Timepoint', 'Bundle', 'Section']

    exclude_bundles = {"FX_L", "FX_R", "MCP", "AC", "ICP_L", "ICP_R", "PC", "SCP_L", "SCP_R"}
    df = df[~df.index.get_level_values("Bundle").isin(exclude_bundles)]
    df = df[~df.index.get_level_values("Bundle").str.contains("CG")]

    metrics = [
        "lesion_total_volume",
        "FIT_FW_metric",
        "fw_corr_ad_metric",
        "fw_corr_fa_metric",
        "fw_corr_md_metric",
        "fw_corr_rd_metric"
    ]

    # --- Comparaison populationnelle
    results = []
    for metric in metrics:
        for (subj, bundle), sub_df in df.groupby(["Subject", "Bundle"]):
            series = sub_df[metric]
            local = compare_to_population_by_bundle(series, metric, df, args.p_threshold)
            local = local.reset_index()
            local.columns = ["Section", "Trend", "RelativeChange(%)"]
            local["Subject"] = subj
            local["Bundle"] = bundle
            local["Metric"] = metric
            results.append(local)

    results_df = pd.concat(results, ignore_index=True)
    results_df = results_df[["Subject", "Bundle", "Section", "Metric", "Trend", "RelativeChange(%)"]]
    results_df.to_csv("global_analyses/tables/segment_population_trends.csv", index=False)

    results_df = results_df[results_df["Trend"] != "NA"]
    results_df["MetricTrend"] = results_df["Metric"] + "_" + results_df["Trend"]

    # ===============================================================
    # --- PROFILS : Toutes les tendances (above / below / equal)
    # ===============================================================
    diffusion_metrics = [m for m in metrics if m != "lesion_total_volume"]
    all_bundles = sorted(results_df["Bundle"].unique())
    all_sections = sorted(results_df["Section"].unique())
    all_trends = ["above", "below", "equal"]

    for diff_metric in diffusion_metrics:
        for trend in all_trends:
            lift_mat = pd.DataFrame(np.nan, index=all_bundles, columns=all_sections)
            pctcooc_mat = pd.DataFrame(np.nan, index=all_bundles, columns=all_sections)
            pct_given_lesion_mat = pd.DataFrame(np.nan, index=all_bundles, columns=all_sections)
            confidence_mat = pd.DataFrame(np.nan, index=all_bundles, columns=all_sections)

            for bundle in all_bundles:
                for section in all_sections:
                    sub = results_df[(results_df["Bundle"] == bundle) & (results_df["Section"] == section)]
                    subjects_here = sub["Subject"].unique()
                    n_total = len(subjects_here)
                    if n_total == 0:
                        continue

                    lesion_mask = (
                        (sub["Metric"] == "lesion_total_volume") &
                        (sub["Trend"] == "lesion_presente")
                    )
                    subjects_lesion = set(sub.loc[lesion_mask, "Subject"].unique())
                    n_antecedent = len(subjects_lesion)

                    diff_mask = (
                        (sub["Metric"] == diff_metric) &
                        (sub["Trend"] == trend)
                    )
                    subjects_diff = set(sub.loc[diff_mask, "Subject"].unique())
                    n_consequent = len(subjects_diff)

                    subjects_both = subjects_lesion.intersection(subjects_diff)
                    n_both = len(subjects_both)

                    lift, pct_total, pct_given, confidence = compute_lift_and_pct(
                        n_both, n_antecedent, n_consequent, n_total
                    )

                    lift_mat.at[bundle, section] = lift
                    pctcooc_mat.at[bundle, section] = pct_total
                    pct_given_lesion_mat.at[bundle, section] = pct_given
                    confidence_mat.at[bundle, section] = confidence

            # --- Sauvegardes
            lift_mat.to_csv(f"profils/tables/matrix_lift_lesion_vs_{diff_metric}_{trend}.csv", index=True)
            pctcooc_mat.to_csv(f"profils/tables/matrix_pctcooc_lesion_vs_{diff_metric}_{trend}.csv", index=True)
            pct_given_lesion_mat.to_csv(
                f"profils/tables/matrix_pct_given_lesion_lesion_vs_{diff_metric}_{trend}.csv", index=True
            )
            confidence_mat.to_csv(
                f"profils/tables/matrix_confidence_lesion_vs_{diff_metric}_{trend}.csv",
                index=True
            )

            # --- Heatmaps
            for name, mat, label, title, suffix in [
                ("lift", lift_mat, "Lift", ...),
                ("pctcooc", pctcooc_mat, "% sujets (lesion ∧ diff_trend)", ...),
                ("pctgiven", pct_given_lesion_mat, "% parmi les sujets avec lésion", ...),
                ("confidence", confidence_mat, "Confidence",
                f"Confidence : P({diff_metric}_{trend} | lesion_presente)",
                "confidence")
            ]:
                plt.figure(figsize=(max(8, 0.4 * len(all_sections)), max(6, 0.3 * len(all_bundles))))
                sns.heatmap(mat.astype(float), annot=True, fmt=".2f", linewidths=0.5, cbar_kws={'label': label})
                plt.title(title)
                plt.xlabel("Section")
                plt.ylabel("Bundle")
                plt.tight_layout()
                plt.savefig(f"profils/graphs/heatmap_{suffix}_lesion_vs_{diff_metric}_{trend}.png", dpi=300)
                plt.close()

    print("✅ Profils multi-trend générés : above / below / equal pour chaque métrique de diffusion.")

# ===============================================================
# --- Heatmap globale : Confidence (P(diff_metric_trend | lesion))
# ===============================================================

    print("Calcul du heatmap global de confidence...")

    all_bundles = sorted(results_df["Bundle"].unique())
    all_sections = sorted(results_df["Section"].unique())

    confidence_global = pd.DataFrame(np.nan, index=all_bundles, columns=all_sections)

    for bundle in all_bundles:
        for section in all_sections:

            sub = results_df[(results_df["Bundle"] == bundle) &
                            (results_df["Section"] == section)]

            if sub.empty:
                continue

            subjects = sub["Subject"].unique()
            n_total = len(subjects)

            # Sujet avec lésion
            lesion_mask = (
                (sub["Metric"] == "lesion_total_volume") &
                (sub["Trend"] == "lesion_presente")
            )
            subjects_lesion = set(sub.loc[lesion_mask, "Subject"].unique())
            n_lesion = len(subjects_lesion)

            if n_lesion == 0:
                continue

            # Tous trends / toutes métriques diffusion confondues
            diff_mask = (
                (sub["Metric"] != "lesion_total_volume") &
                (sub["Trend"].isin(["above", "below", "equal"]))
            )
            subjects_diff = set(sub.loc[diff_mask, "Subject"].unique())

            # intersection
            subjects_both = subjects_lesion.intersection(subjects_diff)
            n_both = len(subjects_both)

            confidence = n_both / n_lesion if n_lesion > 0 else np.nan

            confidence_global.at[bundle, section] = confidence

    # Sauvegarde CSV
    confidence_global.to_csv("global_analyses/tables/global_confidence_matrix.csv")

    # Heatmap global
    plt.figure(figsize=(max(8, 0.4 * len(all_sections)), max(6, 0.3 * len(all_bundles))))
    sns.heatmap(confidence_global.astype(float),
                annot=True, fmt=".2f", linewidths=0.5,
                cbar_kws={'label': 'Confidence'})
    plt.title("Confidence globale : P(diffusion trend | lésion présente)")
    plt.xlabel("Section")
    plt.ylabel("Bundle")
    plt.tight_layout()
    plt.savefig("global_analyses/graphs/heatmap_global_confidence.png", dpi=300)
    plt.close()

    print("✅ Heatmap de confidence globale généré.")


# ===============================================================
# --- Exécution
# ===============================================================
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse FP-Growth longitudinale (y2 - y0) + comparaison populationnelle
avec analyses globales, par bundle et profils multi-trend (heatmaps lift).
"""

import os
import gc
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from scipy.stats import ttest_1samp


# ===============================================================
# --- ARGUMENTS
# ===============================================================
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Analyse FP-Growth sur variations y2 - y0 avec comparaison à la moyenne populationnelle",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument('metrics_csv', help='Path du fichier CSV (y0 et y2)')
    p.add_argument('--p_threshold', type=float, default=0.05,
                   help='Seuil de significativité pour caractériser la variation')
    return p


# ===============================================================
# --- STRUCTURE DE DOSSIERS
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
# --- COMPARAISON POPULATIONNELLE (variation y2-y0)
# ===============================================================
def compare_diff_to_population(series, metric, population_df, p_threshold):
    trends = {}
    rel_changes = {}

    subj = series.index.get_level_values('Subject').unique()[0]
    bundle = series.index.get_level_values('Bundle').unique()[0]
    sections = series.index.get_level_values('Section').unique().sort_values()

    for seg in sections:
        val_indiv = series.xs(seg, level="Section").mean()

        # --- Cas lésion
        if metric == "lesion_total_volume":
            if val_indiv > 0:
                trends[seg] = "increase"
            elif val_indiv < 0:
                trends[seg] = "decrease"
            else:
                trends[seg] = "constant"
            rel_changes[seg] = np.nan
            continue

        try:
            pop_subdf = population_df.xs((bundle, seg), level=("Bundle", "Section"))
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

        # --- Test t unilatéral
        t_stat, p_val = ttest_1samp(ref_values, popmean=val_indiv, nan_policy='omit')

        if p_val < p_threshold:
            if val_indiv > ref_mean:
                trends[seg] = "increase"
            elif val_indiv < ref_mean:
                trends[seg] = "decrease"
            else:
                trends[seg] = "constant"
        else:
            trends[seg] = "constant"

        rel_changes[seg] = ((val_indiv - ref_mean) / abs(ref_mean)) * 100 if ref_mean != 0 else np.nan

    return pd.DataFrame({
        "Trend": pd.Series(trends),
        "RelativeChange(%)": pd.Series(rel_changes)
    })


# ===============================================================
# --- FONCTION UTILITAIRE : calcul du lift
# ===============================================================
def compute_lift_and_pct(n_both, n_antecedent, n_consequent, n_total):
    if n_total == 0:
        return (np.nan, np.nan, np.nan)
    sup_both = n_both / n_total
    sup_A = n_antecedent / n_total
    sup_C = n_consequent / n_total
    if sup_A == 0 or sup_C == 0:
        lift = np.nan
    else:
        lift = sup_both / (sup_A * sup_C)
    pct_total = (n_both / n_total) * 100
    pct_given_lesion = (n_both / n_antecedent * 100) if (n_antecedent > 0) else np.nan
    return (lift, pct_total, pct_given_lesion)


# ===============================================================
# --- FP-GROWTH UTILE
# ===============================================================
def same_metric(antecedent, consequent):
    a_metric = list(antecedent)[0].split("_")[0]
    c_metric = list(consequent)[0].split("_")[0]
    return a_metric == c_metric


# ===============================================================
# --- MAIN
# ===============================================================
def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    ensure_dirs()

    # --- Chargement des données
    df = pd.read_csv(args.metrics_csv, index_col=[0, 1, 2, 3])
    df.index.names = ['Subject', 'Timepoint', 'Bundle', 'Section']

    exclude_bundles = {"FX_L", "FX_R", "MCP", "AC", "ICP_L", "ICP_R", "PC", "SCP_L", "SCP_R"}
    df = df[~df.index.get_level_values("Bundle").isin(exclude_bundles)]
    df = df[~df.index.get_level_values("Bundle").str.contains("CG")]

    metrics = ["lesion_total_volume", "FIT_FW_metric", "fw_corr_ad_metric",
               "fw_corr_fa_metric", "fw_corr_md_metric", "fw_corr_rd_metric"]

    # --- Différences Y2 - Y0
    y0 = df.xs("y0", level="Timepoint")
    y2 = df.xs("y2", level="Timepoint")
    diff_df = y2 - y0
    diff_df.index.names = ['Subject', 'Bundle', 'Section']
    diff_df_pop = diff_df

    # --- Résultats par sujet et métrique
    results = []
    for metric in metrics:
        for (subj, bundle), sub_df in diff_df.groupby(["Subject", "Bundle"]):
            local = compare_diff_to_population(sub_df[metric], metric, diff_df_pop, args.p_threshold)
            if local.empty:
                continue
            local = local.reset_index(level=0, drop=True).reset_index()
            local.columns = ["Section", "Trend", "RelativeChange(%)"]
            local["Subject"] = subj
            local["Bundle"] = bundle
            local["Metric"] = metric
            results.append(local)

    results_df = pd.concat(results, ignore_index=True)
    results_df = results_df[["Subject", "Bundle", "Section", "Metric", "Trend", "RelativeChange(%)"]]
    results_df.to_csv("global_analyses/tables/segment_population_trends_diff.csv", index=False)

    # --- Nettoyage et enrichissement
    results_df = results_df[results_df["Trend"] != "NA"]
    results_df["MetricTrend"] = results_df["Metric"].astype(str) + "_" + results_df["Trend"].astype(str)

    # --- Résumé global
    summary_df = (
        results_df.groupby(["Metric", "Bundle", "Section", "Trend"])["Subject"]
        .nunique()
        .reset_index(name="Nb_sujets")
    )
    total_per_group = summary_df.groupby(["Metric", "Bundle", "Section"])["Nb_sujets"].transform("sum")
    summary_df["Pourcentage"] = (summary_df["Nb_sujets"] / total_per_group) * 100
    summary_df.to_csv("global_analyses/tables/population_trend_summary_diff.csv", index=False)

    # ===============================================================
    # --- PROFILS (heatmaps multi-trend)
    # ===============================================================
    diffusion_metrics = [m for m in metrics if m != "lesion_total_volume"]
    all_bundles = sorted(results_df["Bundle"].unique())
    all_sections = sorted(results_df["Section"].unique())
    all_trends = ["increase", "decrease", "constant"]

    for diff_metric in diffusion_metrics:
        for trend in all_trends:
            lift_mat = pd.DataFrame(np.nan, index=all_bundles, columns=all_sections)
            pctcooc_mat = pd.DataFrame(np.nan, index=all_bundles, columns=all_sections)
            pct_given_mat = pd.DataFrame(np.nan, index=all_bundles, columns=all_sections)

            for bundle in all_bundles:
                for section in all_sections:
                    sub = results_df[(results_df["Bundle"] == bundle) & (results_df["Section"] == section)]
                    subjects_here = sub["Subject"].unique()
                    n_total = len(subjects_here)
                    if n_total == 0:
                        continue

                    lesion_mask = (
                        (sub["Metric"] == "lesion_total_volume") &
                        (sub["Trend"] == "increase")
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

                    lift, pct_total, pct_given = compute_lift_and_pct(
                        n_both, n_antecedent, n_consequent, n_total
                    )
                    lift_mat.at[bundle, section] = lift
                    pctcooc_mat.at[bundle, section] = pct_total
                    pct_given_mat.at[bundle, section] = pct_given

            # --- Sauvegarde tables
            lift_mat.to_csv(f"profils/tables/matrix_lift_lesion_vs_{diff_metric}_{trend}.csv")
            pctcooc_mat.to_csv(f"profils/tables/matrix_pctcooc_lesion_vs_{diff_metric}_{trend}.csv")
            pct_given_mat.to_csv(f"profils/tables/matrix_pct_given_lesion_vs_{diff_metric}_{trend}.csv")

            # --- Heatmaps
            for name, mat, label, title, suffix in [
                ("lift", lift_mat, "Lift", f"Lift : lesion increase → {diff_metric} ({trend})", "lift"),
                ("pctcooc", pctcooc_mat, "% sujets (lesion ∧ diff_trend)",
                 f"% cooccurrence (lesion increase ∧ {diff_metric}_{trend})", "pctcooc"),
                ("pctgiven", pct_given_mat, "% parmi sujets avec lésion",
                 f"% avec lésion ayant {diff_metric}_{trend}", "pctgiven")
            ]:
                plt.figure(figsize=(max(8, 0.4 * len(all_sections)), max(6, 0.3 * len(all_bundles))))
                sns.heatmap(mat.astype(float), annot=True, fmt=".2f", linewidths=0.5, cbar_kws={'label': label})
                plt.title(title)
                plt.xlabel("Section")
                plt.ylabel("Bundle")
                plt.tight_layout()
                plt.savefig(f"profils/graphs/heatmap_{suffix}_lesion_vs_{diff_metric}_{trend}.png", dpi=300)
                plt.close()

    print("✅ Profils multi-trend générés.")

    # ===============================================================
    # --- FP-GROWTH GLOBAL
    # ===============================================================
    transactions = (
        results_df.groupby(["Subject", "Bundle", "Section"])["MetricTrend"]
        .apply(list)
        .reset_index(name="Transaction")
    )

    unique_items = sorted({str(item) for sublist in transactions["Transaction"] for item in sublist if pd.notna(item)})
    encoded_df = pd.DataFrame(False, index=np.arange(len(transactions)), columns=unique_items)

    for i, row in transactions.iterrows():
        for item in row["Transaction"]:
            encoded_df.at[i, item] = True

    freq_itemsets = fpgrowth(encoded_df, min_support=0.01, use_colnames=True)
    freq_itemsets.to_csv("global_analyses/tables/frequent_patterns_population_trends_diff.csv", index=False)

    rules = association_rules(freq_itemsets, metric="lift", min_threshold=0.1)
    rules.to_csv("global_analyses/tables/association_rules_population_trends_diff.csv", index=False)

    rules = rules[
        (rules['antecedents'].apply(len) == 1) &
        (rules['consequents'].apply(len) == 1)
    ]
    rules = rules[~rules.apply(lambda r: same_metric(r['antecedents'], r['consequents']), axis=1)]
    rules.to_csv("global_analyses/tables/association_rules_population_trends_pairs_diff.csv", index=False)

    rules_focus = rules[rules['antecedents'].apply(lambda x: 'lesion_total_volume' in list(x)[0])]
    rules_focus.to_csv("global_analyses/tables/association_rules_population_lesion_total_volume_diff.csv", index=False)

    # ===============================================================
    # --- FP-GROWTH PAR BUNDLE
    # ===============================================================
    all_rules_focus = []
    for bundle in results_df["Bundle"].unique():
        bundle_df = results_df[results_df["Bundle"] == bundle]
        if bundle_df.empty:
            continue

        transactions_bundle = (
            bundle_df.groupby(["Subject", "Section"])["MetricTrend"]
            .apply(list)
            .reset_index(name="Transaction")
        )

        unique_items_bundle = sorted({item for sublist in transactions_bundle["Transaction"] for item in sublist})
        encoded_df_bundle = pd.DataFrame(False, index=np.arange(len(transactions_bundle)), columns=unique_items_bundle)
        for i, row in transactions_bundle.iterrows():
            for item in row["Transaction"]:
                encoded_df_bundle.at[i, item] = True

        freq_itemsets_bundle = fpgrowth(encoded_df_bundle, min_support=0.01, use_colnames=True)
        rules_bundle = association_rules(freq_itemsets_bundle, metric="lift", min_threshold=0.1)
        rules_bundle = rules_bundle[
            (rules_bundle['antecedents'].apply(len) == 1) &
            (rules_bundle['consequents'].apply(len) == 1)
        ]
        rules_bundle = rules_bundle[
            ~rules_bundle.apply(lambda r: same_metric(r['antecedents'], r['consequents']), axis=1)
        ]

        rules_focus_bundle = rules_bundle[
            rules_bundle['antecedents'].apply(lambda x: 'lesion_total_volume' in list(x)[0])
        ]

        if not rules_focus_bundle.empty:
            rules_focus_bundle["Bundle"] = bundle
            all_rules_focus.append(rules_focus_bundle)

        rules_focus_bundle.to_csv(f"bundles/tables/association_rules_population_lesion_total_volume_diff_{bundle}.csv", index=False)

    if all_rules_focus:
        pd.concat(all_rules_focus, ignore_index=True).to_csv(
            "bundles/tables/association_rules_population_lesion_total_volume_diff_ALL.csv", index=False
        )

    print("\n📁 Sauvegardes terminées dans :")
    print(" - global_analyses/tables/")
    print(" - bundles/tables/")
    print(" - profils/tables/ et profils/graphs/")


# ===============================================================
# --- RUN
# ===============================================================
if __name__ == "__main__":
    main()

import argparse
import ast
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ==============================================================
# ARGUMENTS
# ==============================================================

def _build_arg_parser():
    p = argparse.ArgumentParser(description="Analyse NAWM entre T1 et T2 à partir d'un CSV unique (6 subplots)")
    p.add_argument("csv", help="Fichier CSV en entrée")
    p.add_argument("--out_dir", required=True, help="Répertoire de sortie")
    return p


# ==============================================================
# OUTILS
# ==============================================================

def cohens_d(x, y):
    diff = np.array(y) - np.array(x)
    if diff.std(ddof=1) == 0:
        return np.nan
    return diff.mean() / diff.std(ddof=1)


# ==============================================================
# CONSTRUCTION DES DONNÉES
# ==============================================================

def build_long_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    # Parser les colonnes dictionnaires
    for col in ["nawm_metric_t1", "nawm_metric_t2"]:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() != "" else {})

    # Construire DataFrame long
    long_data = []
    for _, row in df.iterrows():
        metric = row["metric_name"]
        subj = row["subject"]

        for dist, val in row["nawm_metric_t1"].items():
            long_data.append([subj, metric, "T1", int(float(dist)), val])
        for dist, val in row["nawm_metric_t2"].items():
            long_data.append([subj, metric, "T2", int(float(dist)), val])

    long_df = pd.DataFrame(long_data, columns=["subject", "metric", "timepoint", "distance", "value"])

    # Garder distances utiles et renommer
    keep_distances = list(range(1, 8))
    rename_map = {1: 0, 2: 2, 3: 4, 4: 6, 5: 8, 6: 10, 7: 12}
    long_df = long_df[long_df["distance"].isin(keep_distances)]
    long_df["distance_mm"] = long_df["distance"].map(rename_map)

    return long_df


# ==============================================================
# COHEN'S D
# ==============================================================

def compute_cohens_d(long_df):
    results = []
    for (metric, dist), group in long_df.groupby(["metric", "distance_mm"]):
        vals_T1 = group[group["timepoint"] == "T1"]["value"]
        vals_T2 = group[group["timepoint"] == "T2"]["value"]
        if len(vals_T1) == 0 or len(vals_T2) == 0:
            continue
        d = cohens_d(vals_T1.values, vals_T2.values)
        results.append({"metric": metric, "distance_mm": dist, "cohen_d": d})
    return pd.DataFrame(results)


# ==============================================================
# VISUALISATION : FIGURE UNIQUE AVEC 6 SOUS-GRAPHIQUES
# ==============================================================

def plot_summary_figure_with_cohen_legend(long_df, df_cohen, out_path):
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="whitegrid", font_scale=1.5)
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.3, hspace=0.3)

    # Préparer labels pour boxplots uniformes
    long_df["distance_label"] = long_df["distance_mm"].astype(int).astype(str)

    metrics_col1 = ["md-t", "fa-t", "ad-t"]
    metrics_col2 = ["rd-t", "fw"]

    # Fonction marge 15%
    def set_ylim_with_margin(ax, data_metric):
        if not data_metric.empty:
            val_min = data_metric["value"].min()
            val_max = data_metric["value"].max()
            margin = 0.15 * (val_max - val_min)
            ax.set_ylim(val_min - margin, val_max + margin)

    # Première colonne : MD/FA/AD
    for i, metric in enumerate(metrics_col1):
        ax = fig.add_subplot(gs[i, 0])
        data_metric = long_df[long_df["metric"] == metric]
        sns.boxplot(
            data=data_metric,
            x="distance_label",
            y="value",
            hue="timepoint",
            ax=ax,
            palette="Set2"
        )
        set_ylim_with_margin(ax, data_metric)
        ax.set_title(metric.upper(), fontsize=18)
        if ax.legend_:
            ax.legend_.remove()
        ax.set_ylabel("", fontsize=14)
        if i < 2:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Distance (mm)", fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

    # Deuxième colonne : RD/FWF
    for i, metric in enumerate(metrics_col2):
        ax = fig.add_subplot(gs[i, 1])
        data_metric = long_df[long_df["metric"] == metric]
        sns.boxplot(
            data=data_metric,
            x="distance_label",
            y="value",
            hue="timepoint",
            ax=ax,
            palette="Set2"
        )
        set_ylim_with_margin(ax, data_metric)
        ax.set_title(metric.upper(), fontsize=18)
        if ax.legend_:
            ax.legend_.remove()
        ax.set_ylabel("", fontsize=14)
        if i < 2:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Distance (mm)", fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

    # Troisième ligne, deuxième colonne : Cohen's d
    ax = fig.add_subplot(gs[2, 1])
    sns.lineplot(
        data=df_cohen,
        x="distance_mm",
        y="cohen_d",
        hue="metric",
        marker="o",
        ax=ax
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Cohen's d", fontsize=18)
    ax.set_ylabel("Cohen's d (T2 - T1)", fontsize=14)
    ax.set_xlabel("Distance from lesion (mm)", fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)


    # Extraire légende Cohen's d et placer à droite
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Metric", loc="center right",
                   fontsize=12, title_fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # laisser place pour légende à droite
    fig.suptitle("NAWM metrics evolution (T1 vs T2) and Cohen's d", fontsize=20, y=0.99)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================
# MAIN
# ==============================================================

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("→ Chargement du CSV...")
    long_df = build_long_dataframe(args.csv)

    print("→ Calcul du Cohen’s d...")
    df_cohen = compute_cohens_d(long_df)

    out_csv = os.path.join(args.out_dir, "cohen_d_results.csv")
    df_cohen.to_csv(out_csv, index=False)

    print("→ Génération de la figure récapitulative (3x2)...")
    out_fig = os.path.join(args.out_dir, "nawm_summary_3x2.png")
    plot_summary_figure(long_df, df_cohen, out_fig)

    print("✅ Figure générée :", out_fig)
    print("✅ Résultats Cohen’s d :", out_csv)


if __name__ == "__main__":
    main()

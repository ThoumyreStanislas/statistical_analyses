import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from scipy.stats import ttest_ind

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Visualisation groupée SSe et MS par bundle avec boxplot + scatter + lignes intra-sujet.",
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--csv", type=str, required=True,
                   help="CSV généré avec les moyennes de métriques par bundle")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Dossier de sortie pour enregistrer les figures")
    return p

def plot_family(ax, df, family, groups_order):
    """Trace un subplot pour une famille de métrique donnée (FA ou MD)"""
    df_family = df[df["metric"].str.startswith(family)]
    if df_family.empty:
        print(f"⚠️ Aucune donnée pour {family}, graphique ignoré.")
        return

    df_family["group"] = df_family["metric"].str.extract(r"_(SS|SSe|MS)$")
    df_family["group"] = pd.Categorical(df_family["group"], categories=groups_order, ordered=True)

    bundles_order = sorted(df_family["bundle"].unique())
    df_family["bundle"] = pd.Categorical(df_family["bundle"], categories=bundles_order, ordered=True)

    # Boxplot
    sns.boxplot(
        data=df_family,
        x="bundle",
        y="mean",
        hue="group",
        order=bundles_order,
        hue_order=groups_order,
        whis=[0, 100],
        fliersize=0,
        dodge=True,
        boxprops={'alpha': 0.6},
        ax=ax
    )

    # Décalage horizontal pour scatter
    hue_levels = groups_order
    hue_offsets = {grp: i - (len(hue_levels)-1)/2 for i, grp in enumerate(hue_levels)}
    width = 0.25

    # Scatter + lignes intra-sujet
    for sid, group in df_family.groupby("sid"):
        for bundle, subgrp in group.groupby("bundle"):
            xs, ys = [], []
            subgrp_sorted = subgrp.set_index("group").reindex(groups_order).dropna()
            for grp, row in subgrp_sorted.iterrows():
                offset = hue_offsets[grp] * width
                x_pos = bundles_order.index(bundle) + offset
                xs.append(x_pos)
                ys.append(row["mean"])
                ax.scatter(x_pos, row["mean"], marker="o", s=40, alpha=0.5, color="grey")
            if len(xs) > 1:
                ax.plot(xs, ys, linestyle="-", linewidth=1.2, alpha=0.3, color="grey")

    # Test statistique SS vs SSe et annotation
    for bundle in bundles_order:
        vals_SS = df_family[(df_family["bundle"]==bundle) & (df_family["group"]=="SS")]["mean"]
        vals_SSe = df_family[(df_family["bundle"]==bundle) & (df_family["group"]=="SSe")]["mean"]
        if len(vals_SS) > 0 and len(vals_SSe) > 0:
            stat, p = ttest_ind(vals_SS, vals_SSe)
            if p < 0.05:  # Significatif
                y_max = max(vals_SS.max(), vals_SSe.max())
                x1 = bundles_order.index(bundle) + hue_offsets["SS"]*width
                x2 = bundles_order.index(bundle) + hue_offsets["SSe"]*width
                y = y_max + (y_max*0.05)
                ax.plot([x1, x2], [y, y], color='black', linewidth=1.5)
                ax.text((x1+x2)/2, y + y_max*0.02, "*", ha='center', va='bottom', fontsize=20)

    # Définir des limites Y plus serrées pour agrandir visuellement les boxplots
    min_val = df_family["mean"].min()
    max_val = df_family["mean"].max()
    padding = (max_val - min_val) * 0.05
    ax.set_ylim(min_val - padding, max_val + padding)

    # Lignes horizontales
    for y in ax.get_yticks():
        ax.axhline(y, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)

    # Label Y = métrique étudiée
    ax.set_ylabel(f"{family}", fontsize=28)
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=26)
    ax.set_title("")

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    families_order = ["FA", "MD"]  # FWF supprimé
    groups_order = ["SS", "SSe", "MS"]

    # Figure principale sans légende
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 12), sharex=True)
    plt.subplots_adjust(hspace=0)  # Coller complètement les subplots

    for ax, family in zip(axes, families_order):
        plot_family(ax, df, family, groups_order)

    bundles_order = sorted(df["bundle"].unique())
    axes[-1].set_xlabel("Bundle", fontsize=28)
    axes[-1].set_xticks(range(len(bundles_order)))
    axes[-1].set_xticklabels(bundles_order, rotation=45, ha="right", fontsize=24)

    # Extraire légende dans une figure séparée
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig_legend = plt.figure(figsize=(6, 3))
        fig_legend.legend(handles, labels, title="Groupe", title_fontsize=28, fontsize=24, loc="center")
        fig_legend.tight_layout()
        fig_legend.savefig(os.path.join(args.output_dir, "legende.png"), dpi=300)
        plt.close(fig_legend)

    # Supprimer légende dans la figure principale
    for ax in axes:
        ax.get_legend().remove()

    out_svg = os.path.join(args.output_dir, "FA_MD_box_scatter_stats.svg")
    out_png = os.path.join(args.output_dir, "FA_MD_box_scatter_stats.png")
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(out_svg)
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"✅ Graphique principal et légende générés dans {args.output_dir}")

if __name__ == "__main__":
    main()

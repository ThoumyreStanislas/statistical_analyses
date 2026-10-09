#!/usr/bin/env python3

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import os


# -------------------------------------------------------
# Parser robuste pour les frozenset(...)
# -------------------------------------------------------
def parse_frozenset(s):
    """Convert a string 'frozenset({...})' into a list of strings safely."""
    if isinstance(s, (list, set)):
        return list(s)

    if not isinstance(s, str):
        return [str(s)]

    s = s.strip()

    # Cas 1 : déjà un set au format {...}
    if s.startswith("{") and s.endswith("}"):
        try:
            return list(ast.literal_eval(s))
        except:
            return [s]

    # Cas 2 : format frozenset({...})
    if s.startswith("frozenset"):
        try:
            inside = s[len("frozenset("):-1]  # remove frozenset(...)
            return list(ast.literal_eval(inside))
        except:
            return [s]

    return [s]

def clean_metric_name(name):
    """Convert raw metric names into readable labels."""
    if not isinstance(name, str):
        return name

    name = name.replace("_", " ")

    # Mappages spécifiques diffusion / FW
    replacements = {
        "FA metric": "FA",
        "MD metric": "md",
        "RD metric": "rd",
        "AD metric": "ad",
        "fw corr": "",
        "FIT FW": "FWF",
        "metric": "",
        "increase": "increase",
        "decrease": "decrease",
        "constant": "constant",
    }

    for key, val in replacements.items():
        name = name.replace(key, val)

    name = " ".join(name.split()).strip()

    # ---------------------------------------------------
    # Ne PAS modifier les métriques de lésions
    # ---------------------------------------------------
    lesion_keywords = ["lesion", "lesions", "flair", "t2"]  # à adapter
    if any(kw in name.lower() for kw in lesion_keywords):
        return name  # on n’y touche pas

    # ---------------------------------------------------
    # Ajout du -t pour les métriques de diffusion
    # ---------------------------------------------------
    tokens = name.split()
    if len(tokens) >= 2:
        metric = tokens[0].upper()  # FA, MD, RD, AD, FWF...
        rest = " ".join(tokens[1:])

        # Ajouter -t sauf pour FWF et sauf pour les lésions (déjà exclues)
        if metric in ["FA", "MD", "RD", "AD"]:
            metric = metric + "-t"

        name = f"{metric} {rest}".strip()

    return name

# -------------------------------------------------------
# Programme principal
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate Lift & Confidence heatmaps from association rules")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV file containing association rules")
    parser.add_argument("--outdir", type=str, required=True, help="Directory where images will be saved")
    parser.add_argument("--prefix", type=str, default="rules_heatmap", help="Filename prefix for saved figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -------------------------------------------------------
    # Load CSV
    # -------------------------------------------------------
    df = pd.read_csv(args.csv)

    # -------------------------------------------------------
    # Parse and clean antecedents & consequents
    # -------------------------------------------------------
    df["antecedents"] = (
        df["antecedents"]
        .apply(parse_frozenset)
        .apply(lambda x: ", ".join([clean_metric_name(i) for i in x]))
    )

    df["consequents"] = (
        df["consequents"]
        .apply(parse_frozenset)
        .apply(lambda x: ", ".join([clean_metric_name(i) for i in x]))
    )

    # -------------------------------------------------------
    # Build pivot tables
    # NOTE: We place antecedents on X (columns) and consequents on Y (index)
    # -------------------------------------------------------
    pivot_lift = df.pivot(index="consequents", columns="antecedents", values="lift")
    pivot_conf = df.pivot(index="consequents", columns="antecedents", values="confidence")

    # -------------------------------------------------------
    # HEATMAP LIFT
    # -------------------------------------------------------
    plt.figure(figsize=(18, 14)) 
    sns_plot = sns.heatmap(pivot_lift, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"fontsize": 24}, vmin=0.5, vmax=1.6)
    plt.title("Lift Heatmap", fontsize=18)
    plt.xlabel("Antecedents", fontsize=16)
    plt.ylabel("Consequents", fontsize=16)
    plt.tight_layout()

    # Taille des ticks
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=12)

    # Taille des ticks de la colorbar
    cbar = sns_plot.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)

    lift_path = os.path.join(args.outdir, f"{args.prefix}_lift.png")
    plt.savefig(lift_path, dpi=300)
    plt.close()

    # -------------------------------------------------------
    # HEATMAP CONFIDENCE
    # -------------------------------------------------------
    plt.figure(figsize=(18, 14))
    sns_plot = sns.heatmap(pivot_conf, annot=True, fmt=".2f", cmap="viridis", annot_kws={"fontsize": 24}, vmin=0.1, vmax=0.7)
    plt.title("Confidence Heatmap", fontsize=18)
    plt.xlabel("Antecedents", fontsize=16)
    plt.ylabel("Consequents", fontsize=16)

    # Taille des ticks
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()

    # Taille des ticks de la colorbar
    cbar = sns_plot.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)

    conf_path = os.path.join(args.outdir, f"{args.prefix}_confidence.png")
    plt.savefig(conf_path, dpi=300)
    plt.close()

    print("\n✔ Lift heatmap:", lift_path)
    print("✔ Confidence heatmap:", conf_path, "\n")


if __name__ == "__main__":
    main()

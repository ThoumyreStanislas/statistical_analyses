import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Compare diffusion metrics between original and corrected NIfTI values.",
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("--csv_1", type=str, required=True,
                   help="Path to the original CSV (e.g., uncorrected FW)")
    p.add_argument("--csv_2", type=str, required=True,
                   help="Path to the corrected CSV (e.g., corrected FW)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save the plots")

    return p

def assign_condition(sid):
    if "_ses-y2" in sid:
        return "Multi-shell"
    elif "_ses-init" in sid or "_ses-sc" in sid:
        return "Single-shell"
    else:
        return "Unknown"

def format_metric_name(metric):
    short = metric.split('_')[-1].upper() + "-t"
    return short

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    plt.rcParams.update({
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })

    os.makedirs(args.output_dir, exist_ok=True)

    df_orig = pd.read_csv(args.csv_1)
    df_corr = pd.read_csv(args.csv_2)

    df_orig["condition"] = df_orig["sid"].apply(assign_condition)
    df_corr["condition"] = "Multi-shell filtered"

    df_combined = pd.concat([df_orig, df_corr], ignore_index=True)

    for metric in df_combined["metric"].unique():
        df_metric = df_combined[df_combined["metric"] == metric]
        metric_label = format_metric_name(metric)

        # Fixer les bornes Y globales pour garder la même échelle sur tous les graphes
        y_min = df_combined[df_combined["metric"] == metric]["mean"].min()
        y_max = df_combined[df_combined["metric"] == metric]["mean"].max()
        y_range = y_max - y_min
        margin = y_range * 0.1 if y_range > 0 else 0.1  # pour éviter les valeurs constantes

        plt.figure(figsize=(14, 6))
        ax = sns.boxplot(data=df_metric, x="bundle", y="mean", hue="condition")
        ax.set_ylabel(f"{metric_label} Value")
        ax.set_xlabel("Bundle")
        ax.set_ylim(y_min - margin, y_max + margin)  # garder une échelle uniforme
        plt.xticks(rotation=45, ha='right')
        ax.legend(title="Acquisition Type")
        plt.tight_layout()

        # Enregistrer en SVG
        output_file_svg = os.path.join(args.output_dir, f"{metric_label}_boxplot.svg")
        plt.savefig(output_file_svg)

        # Enregistrer en PNG
        output_file_png = os.path.join(args.output_dir, f"{metric_label}_boxplot.png")
        plt.savefig(output_file_png, dpi=300)
        plt.close()

if __name__ == "__main__":
    main()

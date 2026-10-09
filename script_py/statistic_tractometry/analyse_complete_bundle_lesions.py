import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--metrics_T1', required=True,
                   help='Path to the label lesion for timepoint 1')
    p.add_argument('--metrics_T2', nargs='+',  required=True,
                   help='Path to the label lesion for timepoint 2')
    p.add_argument('--lesion_T1', nargs='+', required=True,
                   help='Path to the metric data for timepoint 1')
    p.add_argument('--lesion_T2', nargs='+', required=True,
                   help='Path to the metric data for timepoint 2')          

    return p


def load_and_merge_data(metric_y0_path, lesion_y0_path, metric_y2_path, lesion_y2_path):
    """Charge les fichiers CSV et fusionne les données de métriques et lésions pour y0 et y2."""
    lesion_y0 = pd.read_csv(lesion_y0_path)
    lesion_y2 = pd.read_csv(lesion_y2_path)
    metric_y0 = pd.read_csv(metric_y0_path)
    metric_y2 = pd.read_csv(metric_y2_path)

    df_y0 = pd.merge(metric_y0, lesion_y0, on=["sid", "bundle", "sex", "age", "site", "disease", "handedness"],
                     suffixes=("_metric", "_lesion"))
    df_y2 = pd.merge(metric_y2, lesion_y2, on=["sid", "bundle", "sex", "age", "site", "disease", "handedness"],
                     suffixes=("_metric", "_lesion"))

    df_y0["time"] = "y0"
    df_y2["time"] = "y2"

    df_long = pd.concat([df_y0, df_y2], ignore_index=True)
    return df_long

def compute_correlations(df_long):
    """Calcule la corrélation entre métriques de diffusion et de lésion par bundle et temps."""
    correlations = []
    for bundle in df_long["bundle"].unique():
        for time in ["y0", "y2"]:
            sub = df_long[(df_long["bundle"] == bundle) & (df_long["time"] == time)]
            if sub.empty:
                continue
            corr = sub.groupby(["metric_metric", "metric_lesion"]).apply(
                lambda g: g["mean_metric"].corr(g["mean_lesion"])
            ).reset_index(name="correlation")
            corr["bundle"] = bundle
            corr["time"] = time
            correlations.append(corr)
    return pd.concat(correlations, ignore_index=True)

def paired_t_tests(df_long):
    """Effectue des tests t appariés entre y0 et y2 pour chaque paire de métriques."""
    pivot_df = df_long.pivot_table(index=["sid", "bundle", "metric_metric", "metric_lesion"],
                                   columns="time", values=["mean_metric", "mean_lesion"]).reset_index()
    pivot_df.columns = ["sid", "bundle", "metric_metric", "metric_lesion",
                        "mean_metric_y0", "mean_metric_y2", "mean_lesion_y0", "mean_lesion_y2"]

    results = []
    for (metric_m, metric_l), group in pivot_df.groupby(["metric_metric", "metric_lesion"]):
        if group[["mean_metric_y0", "mean_metric_y2"]].dropna().shape[0] > 1 and \
           group[["mean_lesion_y0", "mean_lesion_y2"]].dropna().shape[0] > 1:
            t_metric = ttest_rel(group["mean_metric_y0"], group["mean_metric_y2"], nan_policy="omit")
            t_lesion = ttest_rel(group["mean_lesion_y0"], group["mean_lesion_y2"], nan_policy="omit")

            results.append({
                "metric_metric": metric_m,
                "metric_lesion": metric_l,
                "t_stat_metric": t_metric.statistic,
                "p_value_metric": t_metric.pvalue,
                "t_stat_lesion": t_lesion.statistic,
                "p_value_lesion": t_lesion.pvalue,
            })

    return pd.DataFrame(results)

def run_mixed_model(df_long, metric_name, lesion_name):
    """Modèle linéaire mixte pour une paire de métriques diffusion/lésion donnée."""
    sub_df = df_long[(df_long["metric_metric"] == metric_name) &
                     (df_long["metric_lesion"] == lesion_name)].copy()

    sub_df["time_encoded"] = sub_df["time"].map({"y0": 0, "y2": 1})

    model = smf.mixedlm("mean_metric ~ mean_lesion * time_encoded", sub_df, groups=sub_df["sid"])
    result = model.fit()
    return result.summary()

def plot_correlation_heatmap(correlations_df):
    """Affiche une heatmap des corrélations entre métriques diffusion et lésion."""
    heatmap_data = correlations_df.pivot_table(index="metric_metric", columns="metric_lesion", values="correlation")
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Corrélations entre métriques de diffusion et de lésion (tous bundles)")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    # Remplacer les chemins par ceux de tes fichiers
    df = load_and_merge_data(
        args.metrics_T1,
        args.metrics_T2,
        args.lesion_T1,
        args.lesion_T2
    )

    # Analyse
    correlations_df = compute_correlations(df)
    test_results = paired_t_tests(df)

    # Visualisation
    plot_correlation_heatmap(correlations_df)

    # Modèle linéaire mixte sur une paire choisie (exemple)
    example_metric = df["metric_metric"].unique()[0]
    example_lesion = df["metric_lesion"].unique()[0]
    model_summary = run_mixed_model(df, example_metric, example_lesion)
    print(model_summary)

    # Sauvegarde optionnelle
    correlations_df.to_csv("correlations_df.csv", index=False)
    test_results.to_csv("test_results.csv", index=False)
    
if __name__ == "__main__":
    main()
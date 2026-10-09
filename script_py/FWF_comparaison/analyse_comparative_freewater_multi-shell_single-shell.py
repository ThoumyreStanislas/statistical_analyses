import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Effectuer une corrélation sur des valeurs extraites d'images NIfTI à partir de masques.",
        formatter_class=argparse.RawTextHelpFormatter)
    
    p.add_argument("--csv_1", type=str, 
                   help="")
    p.add_argument("--csv_2", type=str,
                   help="")
    
    return p


def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    # Charger les fichiers
    df_before = pd.read_csv(args.csv_1)
    df_after = pd.read_csv(args.csv_2)

    # Ajouter la colonne pour indiquer la source
    df_before['harmonized'] = 'before'
    df_after['harmonized'] = 'after'

    # Fusionner
    df_all = pd.concat([df_before, df_after], ignore_index=True)

    # Créer une colonne groupe combinée
    df_all['group'] = df_all['handedness'].astype(str) + '_' + df_all['harmonized']

    # Palette harmonisée
    palette = {
        '1_before': '#4C72B0',  # single-shell before
        '2_before': '#DD8452',  # multi-shell before
        '1_after': '#8DBAE4',   # single-shell after
        '2_after': '#F2C5A9'    # multi-shell after
    }

    # Ordre des groupes
    hue_order = ['1_before', '1_after', '2_before', '2_after']

    # Noms lisibles pour la légende
    group_labels = {
        '1_before': 'single-shell before',
        '1_after': 'single-shell after',
        '2_before': 'multi-shell before',
        '2_after': 'multi-shell after'
    }

    # Tracer et sauvegarder un graphe pour chaque métrique
    metrics = df_all['metric'].unique()
    for metric in metrics:
        df_metric = df_all[df_all['metric'] == metric]
        
        plt.figure(figsize=(16, 6))
        ax = sns.boxplot(
            data=df_metric, 
            x='bundle', 
            y='mean', 
            hue='group', 
            palette=palette,
            hue_order=hue_order
        )

        plt.title(f"Comparaison des valeurs moyennes - {metric}")
        plt.xlabel("Bundle")
        plt.ylabel("Valeur moyenne")
        plt.xticks(rotation=45)

        # Modifier la légende avec les nouveaux labels
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [group_labels[label] for label in labels]
        plt.legend(handles=handles, labels=new_labels, title="Groupe")
        
        plt.tight_layout()
        
        # Sauvegarde
        base_filename = f"boxplot_{metric}"
        plt.savefig(f"{base_filename}.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
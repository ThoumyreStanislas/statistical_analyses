import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Analyse FP-Growth avec comparaison à la moyenne populationnelle par segment",
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('rules_csv',
                   help='Path du fichier CSV avec toutes les associations')
    return p

# ---------- Helpers de renommage ----------
FWC_PATTERN = re.compile(
    r"^fw_corr_(fa|ad|rd|md)_metric_(below|above|equal|increase|decrease|constant)$",
    re.IGNORECASE
)
FWF_PATTERN = re.compile(
    r"^FIT_FW_metric_(below|above|equal|increase|decrease|constant)$",
    re.IGNORECASE
)

LESION_MAP = {
    "lesion_total_volume_lesion_presente": "With lesion",
    "lesion_total_volume_aucune_lesion": "Without lesion",
    "lesion_total_volume_constant": "Lesion volume constant",
    "lesion_total_volume_increase": "Lesion volume increase",
    "lesion_total_volume_decrease": "Lesion volume decrease",
}

def rename_token(tok: str) -> str:
    t = tok.strip()
    if t in LESION_MAP:
        return LESION_MAP[t]
    m = FWF_PATTERN.match(t)
    if m:
        state = m.group(1).lower()
        return f"FWF {state}"
    m = FWC_PATTERN.match(t)
    if m:
        metric = m.group(1).upper()
        state  = m.group(2).lower()
        return f"{metric}-t {state}"
    return t

def pretty_items(cell: str) -> str:
    """Nettoie frozenset(...) et renomme chaque token (si plusieurs tokens, les joint avec &)."""
    if pd.isna(cell):
        return cell
    s = str(cell)
    s = re.sub(r"^frozenset\((.*)\)$", r"\1", s)
    s = re.sub(r"[{}']", "", s).strip()
    if not s:
        return s
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    renamed = [rename_token(t) for t in tokens]
    return " & ".join(renamed)
    
def main ():

    parser = _build_arg_parser()
    args = parser.parse_args()

    # Charger le fichier
    df = pd.read_csv(args.rules_csv)

    # Nettoyage / renommage
    for col in ['antecedents', 'consequents']:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(pretty_items)

    # Construire la matrice lift
    lift_matrix = df.pivot(index="consequents", columns="antecedents", values="lift")

    # Heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        lift_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        cbar_kws={'label': 'Lift'},
        annot_kws={"size": 12}  # Taille des annotations dans les cellules
    )

    # Titre et labels des axes
    plt.title("Lift matrix", fontsize=18)
    plt.ylabel("Consequents", fontsize=14)
    plt.xlabel("Antecedents", fontsize=14)

    # Taille des ticks
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # Modifier la colorbar
    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.label.set_size(14)  # Taille du label "Lift"
    colorbar.ax.yaxis.label.set_rotation(270)  # Orientation verticale (défaut)
    colorbar.ax.tick_params(labelsize=12)
    colorbar.ax.yaxis.labelpad = 15   # Taille des ticks sur la colorbar

    plt.tight_layout()
    plt.savefig("matrix_association.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
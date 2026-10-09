import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Transforme un CSV d'associations en tableau visuel",
        formatter_class=argparse.RawTextHelpFormatter
    )
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

# ---------- Pipeline principal ----------
def csv_to_table_figure(csv_path: str) -> str:
    df = pd.read_csv(csv_path)

    # Vérifie présence de 'lift' et tronque les colonnes jusqu'à 'lift'
    if 'lift' not in df.columns:
        raise ValueError("La colonne 'lift' est absente du CSV.")
    last_idx = df.columns.get_loc('lift')
    keep_cols = list(df.columns[:last_idx+1])
    df = df.loc[:, keep_cols]

    # Nettoyage / renommage
    for col in ['antecedents', 'consequents']:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(pretty_items)

    # Arrondir les colonnes numériques à 4 décimales
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].round(4)

    # --- Tri personnalisé : ordre des catégories ---
    PRIORITY_ORDER = {
        "Without lesion": 0,
        "With lesion": 1,
        "Lesion volume constant": 2,
        "Lesion volume decrease": 3,
        "Lesion volume increase": 4,
    }

    def lesion_priority(row):
        text = str(row.get('antecedents', '')) + " " + str(row.get('consequents', ''))
        for key, pr in PRIORITY_ORDER.items():
            if key in text:
                return pr
        return 99

    df["__priority__"] = df.apply(lesion_priority, axis=1)

    # Trier par priorité puis par lift (descendant)
    df_sorted = df.sort_values(by=["__priority__", "lift"], ascending=[True, False]).reset_index(drop=True)

    # Récupère la liste des priorités (pour la coloration ligne par ligne)
    priorities = df_sorted["__priority__"].tolist()

    # Prépare le DataFrame d'affichage sans la colonne interne
    df_display = df_sorted.drop(columns="__priority__")

    # Création de la figure-table
    n_rows, n_cols = df_display.shape
    fig_w = max(6, min(1.6 * n_cols, 28))
    fig_h = max(4, min(0.4 * n_rows + 1.5, 50))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    # Coloration ligne par ligne basée sur la priorité
    colors = {
        0: "#f0f8ff",  # Without lesion (bleu clair)
        1: "#fff5f5",  # With lesion (rouge clair)
        2: "#f5fff5",  # Constant (vert très clair)
        3: "#ffffe0",  # Decrease (jaune pâle)
        4: "#f9f0ff",  # Increase (violet clair)
        99: "#ffffff"  # Autres
    }

    # Colorer l'en-tête
    for j in range(n_cols):
        hdr = table[(0, j)]
        hdr.set_facecolor("#dddddd")
        try:
            hdr.get_text().set_weight('bold')
        except Exception:
            pass

    # Coloration des cellules de données
    for i in range(n_rows):
        pr = priorities[i] if i < len(priorities) else 99
        row_color = colors.get(pr, "#ffffff")
        for j in range(n_cols):
            cell = table[(i+1, j)]  # +1 car 0 = header
            cell.set_facecolor(row_color)

    # ⚡ élargir un peu la première colonne
    for i in range(n_rows + 1):  # +1 car inclut l'en-tête
        cell = table[(i, 0)]
        cell.set_width(cell.get_width() * 1.3)

    # Sauvegarde en PNG
    out_path = os.path.splitext(csv_path)[0] + "_table.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    csv_to_table_figure(args.rules_csv)

if __name__ == "__main__":
    main()

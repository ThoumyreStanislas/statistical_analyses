import os
import nibabel as nib
import numpy as np
import argparse
import re
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Effectuer une corrélation sur des valeurs extraites d'images NIfTI à partir de masques.",
        formatter_class=argparse.RawTextHelpFormatter)
    
    p.add_argument("--input", type=str, nargs='+', 
                   help="Liste des fichiers images NIfTI à traiter")
    p.add_argument("--mask", type=str, nargs='+',
                   help="Liste des fichiers masques NIfTI")
    p.add_argument("--output", type=str, 
                   help="Chemin du fichier de sortie pour enregistrer les résultats")
    
    return p

def extract_subject_id(filename):
    """Extrait l'identifiant unique du sujet depuis le nom du fichier."""
    match = re.search(r"(sub-\d+S\d+)_ses-([\w\d]+)__", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def apply_mask_and_compute_mean(image_path, mask_path):
    """Applique un masque sur une image NIfTI et calcule la moyenne des valeurs dans le masque."""
    img = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    
    masked_values = img[mask > 0]  # Sélectionner les valeurs où le masque est > 0
    return np.mean(masked_values) if masked_values.size > 0 else None, np.std(masked_values) if masked_values.size > 0 else None

def match_inputs_with_masks(input_files, mask_files):
    """Associe les fichiers input avec leurs masques correspondants en fonction du sujet et de la session."""
    matched_pairs = {}
    
    for mask in mask_files:
        subject_id, session = extract_subject_id(os.path.basename(mask))
        if subject_id and session:
            for img in input_files:
                img_subject_id, img_session = extract_subject_id(os.path.basename(img))
                if subject_id == img_subject_id and session == img_session:
                    matched_pairs.setdefault(subject_id, {})[session] = (img, mask)
    
    return matched_pairs

def perform_correlation(matched_data, output_file):
    """Effectue une corrélation entre les valeurs extraites pour chaque sujet."""
    X, Y = [], []
    
    for subject, sessions in matched_data.items():
        if len(sessions) == 2:  # Vérifier qu'il y a bien 2 timepoints
            timepoints = sorted(sessions.keys())  # Trier par timepoint (ex: sc et y2)
            img1, mask1 = sessions[timepoints[0]]
            img2, mask2 = sessions[timepoints[1]]
            
            mean1 = apply_mask_and_compute_mean(img1, mask1)
            mean2 = apply_mask_and_compute_mean(img2, mask2)
            
            if mean1 is not None and mean2 is not None:
                X.append(mean1)
                Y.append(mean2)
    
    if X and Y:
        X = np.array(X).flatten()  # Assurer que X est un vecteur 1D
        Y = np.array(Y).flatten()  # Assurer que Y est un vecteur 1D

        correlation, p_value = pearsonr(X, Y)
        slope, intercept, _, _, std_err = linregress(X, Y)
        
        # Calcul de l'intervalle de confiance à 95 %
        n = len(X)
        tcrit = t.ppf(0.975, df=n-2)  # Valeur critique de Student pour 95%
        ci = tcrit * std_err

        # Génération des valeurs pour la droite de régression
        x_range = np.linspace(min(X), max(X), 100)
        y_fit = slope * x_range + intercept
        y_fit_upper = y_fit + ci
        y_fit_lower = y_fit - ci

        plt.figure(figsize=(6, 6))
        plt.scatter(X, Y, label="Données", color='blue')
        plt.plot(x_range, y_fit, color='red', label="Droite de corrélation")
        plt.fill_between(x_range, y_fit_lower, y_fit_upper, color='red', alpha=0.2, label="IC 95%")
        plt.xlabel("Valeurs de single-shell")
        plt.ylabel("Valeurs au multi-shell")
        plt.title(f"Corrélation des valeurs dans les masques\n r={correlation:.2f}, p={p_value:.3f}")
        plt.legend()
        plt.savefig(output_file)
        plt.show()
        
        print(f"Graphique enregistré dans {output_file}")
        print(f"Coefficient de corrélation de Pearson: r={correlation:.2f}, p={p_value:.3f}")
    else:
        print("Pas assez de données appariées pour effectuer la corrélation.")

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    matched_pairs = match_inputs_with_masks(args.input, args.mask)
    perform_correlation(matched_pairs, args.output)

if __name__ == "__main__":
    main()

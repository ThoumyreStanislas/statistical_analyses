import os
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob

def load_nifti(file):
    return nib.load(file).get_fdata()

def compute_bundle_mean(metric_img, mask_files):
    """Combine les masques L/R s'ils existent, puis calcule la moyenne des valeurs de la métrique"""
    combined_mask = None
    for mask_file in mask_files:
        mask = load_nifti(mask_file) > 0
        combined_mask = mask if combined_mask is None else combined_mask | mask

    if combined_mask is None or np.sum(combined_mask) == 0:
        return np.nan  # Aucun voxel dans le masque

    return float(np.nanmean(metric_img[combined_mask]))

def main():
    parser = argparse.ArgumentParser(description="Extraction des moyennes de métriques par faisceau (MS, MSp, SSe, SS) et génération CSV.")
    parser.add_argument("--fa_dir", required=True, help="Dossier contenant les cartes FA corrigées (par sujet).")
    parser.add_argument("--md_dir", required=True, help="Dossier contenant les cartes MD corrigées (par sujet).")
    parser.add_argument("--fwf_dir", required=True, help="Dossier contenant les cartes FWF harmonisées (par sujet).")
    parser.add_argument("--bundle_dir", required=True, help="Dossier contenant les masques de faisceaux (binary).")
    parser.add_argument("--output_csv", required=True, help="Chemin du fichier CSV de sortie.")
    args = parser.parse_args()

    results = []

    subjects = sorted([os.path.basename(s) for s in glob(os.path.join(args.bundle_dir, "sub-*"))])

    for subj in subjects:
        sid = subj
        print(f"🔄 Traitement du sujet : {sid}")

        # FA maps
        fa_maps = {
            "FA_MS": os.path.join(args.fa_dir, sid, f"{sid}__fw_corr_fa_MS.nii.gz"),
            "FA_MSp": os.path.join(args.fa_dir, sid, f"{sid}__fw_corr_fa_MSp.nii.gz"),
            "FA_SSe": os.path.join(args.fa_dir, sid, f"{sid}__fw_corr_fa_SSe.nii.gz"),
            "FA_SS": os.path.join(args.fa_dir, sid, f"{sid}__fw_corr_fa_SS.nii.gz"),
        }

        # MD maps
        md_maps = {
            "MD_MS": os.path.join(args.md_dir, sid, f"{sid}__fw_corr_md_MS.nii.gz"),
            "MD_MSp": os.path.join(args.md_dir, sid, f"{sid}__fw_corr_md_MSp.nii.gz"),
            "MD_SSe": os.path.join(args.md_dir, sid, f"{sid}__fw_corr_md_SSe.nii.gz"),
            "MD_SS": os.path.join(args.md_dir, sid, f"{sid}__fw_corr_md_SS.nii.gz"),
        }

        # FWF maps
        fwf_maps = {
            "FWF_MS": os.path.join(args.fwf_dir, sid, f"{sid}_FWF_harmonized_MS.nii.gz"),
            "FWF_MSp": os.path.join(args.fwf_dir, sid, f"{sid}_FWF_harmonized_MSp.nii.gz"),
            "FWF_SSe": os.path.join(args.fwf_dir, sid, f"{sid}_FWF_harmonized_SSe.nii.gz"),
            "FWF_SS": os.path.join(args.fwf_dir, sid, f"{sid}_FWF_harmonized_SS.nii.gz"),
        }

        # Fusionne toutes les cartes (FA, MD, FWF)
        all_maps = {**fa_maps, **md_maps, **fwf_maps}

        # Charge les bundles dispo
        bundle_files = sorted(glob(os.path.join(args.bundle_dir, sid, f"{sid}__*_binary_mni.nii.gz")))

        if not bundle_files:
            print(f"⚠️ Aucun masque trouvé pour {sid}, on passe.")
            continue

        # Regroupe par nom de faisceau sans _L/_R
        bundle_dict = {}
        for bf in bundle_files:
            bname = os.path.basename(bf).replace(f"{sid}__", "").replace("_binary_mni.nii.gz", "")
            bname_core = bname.replace("_L", "").replace("_R", "")
            bundle_dict.setdefault(bname_core, []).append(bf)

        # Calcul des moyennes
        for metric_name, metric_file in all_maps.items():
            if not os.path.exists(metric_file):
                print(f"⚠️ {metric_name} introuvable pour {sid}, ignoré.")
                continue

            metric_img = load_nifti(metric_file)

            for bundle, masks in bundle_dict.items():
                mean_val = compute_bundle_mean(metric_img, masks)
                results.append([sid, bundle, mean_val, metric_name])

    # Sauvegarde CSV
    df = pd.DataFrame(results, columns=["sid", "bundle", "mean", "metric"])
    df.to_csv(args.output_csv, index=False)
    print(f"✅ CSV généré : {args.output_csv}")

if __name__ == "__main__":
    main()

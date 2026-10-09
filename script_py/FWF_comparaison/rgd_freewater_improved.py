import os
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
import glob

def _build_arg_parser():
    p = argparse.ArgumentParser(description="RGD regularization + alignment of single-shell FW maps")
    p.add_argument("--data_ss", type=str, required=True, help="Répertoire single-shell FW maps (UKB/data_ss/FWF)")
    p.add_argument("--data_ms", type=str, required=True, help="Répertoire multi-shell FW maps (UKB/data_ms/FWF)")
    p.add_argument("--output_dir", type=str, required=True, help="Répertoire de sortie")
    p.add_argument("--n_iter", type=int, default=50, help="Nombre d'itérations RGD")
    p.add_argument("--step_size", type=float, default=0.1, help="Taille du pas gradient descent")
    p.add_argument("--lambda_min", type=float, default=0.05, help="Lambda min pour recherche auto")
    p.add_argument("--lambda_max", type=float, default=0.5, help="Lambda max pour recherche auto")
    p.add_argument("--lambda_step", type=float, default=0.05, help="Pas pour la recherche lambda")
    p.add_argument("--n_boot", type=int, default=20, help="Nombre de réplicas pour bootstrap")
    p.add_argument("--sigma_noise", type=float, default=0.02, help="Sigma bruit pour bootstrap")
    return p

# ---------------------------
# FONCTION RGD
# ---------------------------
def rgd_regularize(fw_map, lambda_reg=0.2, n_iter=50, step_size=0.1, valid_mask=None):
    fw_est = fw_map.copy()
    target_mean = np.mean(fw_map[valid_mask]) if valid_mask is not None else np.mean(fw_map)

    for i in range(n_iter):
        laplacian = gaussian_filter(fw_est, sigma=1) - fw_est
        fw_est = fw_est - step_size * ((fw_est - fw_map) - lambda_reg * laplacian)

        # Renormalisation : on ajuste l'échelle pour préserver la moyenne
        current_mean = np.mean(fw_est[valid_mask]) if valid_mask is not None else np.mean(fw_est)
        if current_mean != 0:
            fw_est *= (target_mean / current_mean)

    return fw_est

# ---------------------------
# BOOTSTRAP POUR INCERTITUDE
# ---------------------------
def bootstrap_uncertainty(fw_map, n_boot=20, sigma_noise=0.02, lambda_reg=0.2, n_iter=50, step_size=0.1):
    boot_maps = []
    for _ in range(n_boot):
        fw_noisy = fw_map + np.random.normal(0, sigma_noise, fw_map.shape)  # pas de clip
        fw_rgd = rgd_regularize(fw_noisy, lambda_reg=lambda_reg, n_iter=n_iter, step_size=step_size)
        boot_maps.append(fw_rgd)
    fw_std = np.std(np.stack(boot_maps, axis=0), axis=0)
    return fw_std

def main ():

    parser = _build_arg_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------
    # LISTER SUJETS
    # ---------------------------
    fw_ss_files = sorted(glob.glob(os.path.join(args.data_ss, "sub-*_FWF_harmonized.nii.gz")))
    fw_ms_files = sorted(glob.glob(os.path.join(args.data_ms, "sub-*_FWF_harmonized.nii.gz")))

    subjects_ss = [os.path.basename(f).split("_")[0] for f in fw_ss_files]
    subjects_ms = [os.path.basename(f).split("_")[0] for f in fw_ms_files]
    subjects_common = sorted(list(set(subjects_ss).intersection(set(subjects_ms))))
    print(f"{len(subjects_common)} sujets trouvés en single-shell et multi-shell.")

    # ---------------------------
    # TRAITEMENT
    # ---------------------------
    results = []

    for subj in subjects_common:
        print(f"Traitement de {subj}...")
        fw_single_path = os.path.join(args.data_ss, f"{subj}_FWF_harmonized.nii.gz")
        fw_multi_path = os.path.join(args.data_ms, f"{subj}_FWF_harmonized.nii.gz")

        fw_single_img = nib.load(fw_single_path)
        fw_multi_img = nib.load(fw_multi_path)
        fw_single = fw_single_img.get_fdata()
        fw_multi = fw_multi_img.get_fdata()
        affine = fw_single_img.affine

        valid_mask = ~np.isnan(fw_multi)

        # ---------------------------
        # Recherche automatique du meilleur lambda
        # ---------------------------
        best_corr = -1
        best_lambda = args.lambda_min
        best_fw_rgd = None
        lambda_values = np.arange(args.lambda_min, args.lambda_max + 1e-5, args.lambda_step)

        for lam in lambda_values:
            fw_rgd = rgd_regularize(fw_single, lambda_reg=lam, n_iter=args.n_iter, step_size=args.step_size)
            corr = np.corrcoef(fw_rgd[valid_mask].flatten(), fw_multi[valid_mask].flatten())[0,1]
            if corr > best_corr:
                best_corr = corr
                best_lambda = lam
                best_fw_rgd = fw_rgd

        print(f"-> Meilleur lambda : {best_lambda:.2f} avec corr = {best_corr:.3f}")

        # ---------------------------
        # Scaling voxel-wise
        # ---------------------------
        fw_rgd_flat = best_fw_rgd[valid_mask].flatten()
        fw_multi_flat = fw_multi[valid_mask].flatten()
        A = np.vstack([fw_rgd_flat, np.ones_like(fw_rgd_flat)]).T
        a, b = np.linalg.lstsq(A, fw_multi_flat, rcond=None)[0]
        fw_rgd_scaled = best_fw_rgd * a + b   # plus de clip

        # ---------------------------
        # Bootstrap pour cartes d'incertitude
        # ---------------------------
        fw_uncertainty = bootstrap_uncertainty(fw_single, n_boot=args.n_boot,
                                            sigma_noise=args.sigma_noise,
                                            lambda_reg=best_lambda,
                                            n_iter=args.n_iter,
                                            step_size=args.step_size)

        # ---------------------------
        # Sauvegarde cartes
        # ---------------------------
        nib.save(nib.Nifti1Image(best_fw_rgd, affine), os.path.join(args.output_dir, f"{subj}_FWF_single_RGD.nii.gz"))
        nib.save(nib.Nifti1Image(fw_rgd_scaled, affine), os.path.join(args.output_dir, f"{subj}_FWF_single_RGD_scaled.nii.gz"))
        nib.save(nib.Nifti1Image(fw_uncertainty, affine), os.path.join(args.output_dir, f"{subj}_FWF_single_RGD_uncertainty.nii.gz"))

        # ---------------------------
        # Statistiques
        # ---------------------------
        mean_single = np.mean(fw_single[valid_mask])
        mean_rgd = np.mean(best_fw_rgd[valid_mask])
        mean_scaled = np.mean(fw_rgd_scaled[valid_mask])
        mean_multi = np.mean(fw_multi[valid_mask])
        mean_uncertainty = np.mean(fw_uncertainty[valid_mask])

        std_single = np.std(fw_single[valid_mask])
        std_rgd = np.std(best_fw_rgd[valid_mask])
        std_scaled = np.std(fw_rgd_scaled[valid_mask])
        std_multi = np.std(fw_multi[valid_mask])

        corr_before = np.corrcoef(fw_single[valid_mask].flatten(), fw_multi[valid_mask].flatten())[0,1]
        corr_rgd = np.corrcoef(best_fw_rgd[valid_mask].flatten(), fw_multi[valid_mask].flatten())[0,1]
        corr_scaled = np.corrcoef(fw_rgd_scaled[valid_mask].flatten(), fw_multi[valid_mask].flatten())[0,1]

        results.append({
            "subject": subj,
            "best_lambda": best_lambda,
            "mean_single": mean_single,
            "mean_RGD": mean_rgd,
            "mean_scaled": mean_scaled,
            "mean_multi": mean_multi,
            "mean_uncertainty": mean_uncertainty,
            "std_single": std_single,
            "std_RGD": std_rgd,
            "std_scaled": std_scaled,
            "std_multi": std_multi,
            "corr_before": corr_before,
            "corr_RGD": corr_rgd,
            "corr_scaled": corr_scaled
        })

    # ---------------------------
    # Sauvegarde CSV
    # ---------------------------
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "FW_RGD_scaled_uncertainty_stats.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"Pipeline terminé. Cartes et statistiques sauvegardées dans {args.output_dir}")

if __name__ == "__main__":
    main()

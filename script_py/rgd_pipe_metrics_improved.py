import os
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
import glob

def _build_arg_parser():
    p = argparse.ArgumentParser(description="RGD + scaling + uncertainty for FA-t and MD-t")
    p.add_argument("--data_ss", type=str, required=True, help="Répertoire single-shell (FA-t / MD-t)")
    p.add_argument("--data_ms", type=str, required=True, help="Répertoire multi-shell (FA-t / MD-t)")
    p.add_argument("--metric", type=str, required=True, choices=["FA", "MD"], help="Métrique : FA ou MD")
    p.add_argument("--output_dir", type=str, required=True, help="Répertoire de sortie")
    p.add_argument("--n_iter", type=int, default=50, help="Nombre d'itérations RGD")
    p.add_argument("--step_size", type=float, default=0.1, help="Taille du pas gradient descent")
    p.add_argument("--lambda_min", type=float, default=0.05, help="Lambda min pour recherche auto")
    p.add_argument("--lambda_max", type=float, default=0.5, help="Lambda max pour recherche auto")
    p.add_argument("--lambda_step", type=float, default=0.05, help="Pas pour recherche lambda")
    p.add_argument("--n_boot", type=int, default=20, help="Nombre de réplicas bootstrap")
    p.add_argument("--sigma_noise", type=float, default=0.01, help="Sigma bruit pour bootstrap")
    return p

# ---------------------------
# FONCTION RGD
# ---------------------------
def rgd_regularize(map_data, lambda_reg=0.2, n_iter=50, step_size=0.1, clip_range=None):
    map_est = map_data.copy()
    for _ in range(n_iter):
        laplacian = gaussian_filter(map_est, sigma=1) - map_est
        map_est = map_est - step_size * ((map_est - map_data) - lambda_reg * laplacian)
        if clip_range is not None:
            map_est = np.clip(map_est, clip_range[0], clip_range[1])
    return map_est

# ---------------------------
# BOOTSTRAP POUR INCERTITUDE
# ---------------------------
def bootstrap_uncertainty(map_data, n_boot=20, sigma_noise=0.01, lambda_reg=0.2, n_iter=50, step_size=0.1, clip_range=None):
    boot_maps = []
    for _ in range(n_boot):
        noisy_map = np.clip(map_data + np.random.normal(0, sigma_noise, map_data.shape), clip_range[0], clip_range[1])
        map_rgd = rgd_regularize(noisy_map, lambda_reg=lambda_reg, n_iter=n_iter, step_size=step_size, clip_range=clip_range)
        boot_maps.append(map_rgd)
    return np.std(np.stack(boot_maps, axis=0), axis=0)

def main ():

    parser = _build_arg_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------
    # LISTER SUJETS COMMUNS
    # ---------------------------
    pattern = f"sub-*__fw_corr_{args.metric.lower()}.nii.gz"
    ss_files = sorted(glob.glob(os.path.join(args.data_ss, pattern)))
    ms_files = sorted(glob.glob(os.path.join(args.data_ms, pattern)))

    subjects_ss = [os.path.basename(f).split("__")[0] for f in ss_files]
    subjects_ms = [os.path.basename(f).split("__")[0] for f in ms_files]
    
    subjects_common = sorted(list(set(subjects_ss).intersection(set(subjects_ms))))
    print(f"{len(subjects_common)} sujets trouvés en single-shell et multi-shell.")

    # ---------------------------
    # DOMAINES POUR CLIP
    # ---------------------------
    clip_ranges = {"FA": (0,1), "MD": (0,0.003)}  # MD en mm²/s typique
    clip_range = clip_ranges[args.metric]

    # ---------------------------
    # TRAITEMENT
    # ---------------------------
    results = []

    for subj in subjects_common:
        print(f"Traitement de {subj} pour {args.metric}-t...")
        ss_path = os.path.join(args.data_ss, f"{subj}__fw_corr_{args.metric.lower()}.nii.gz")
        ms_path = os.path.join(args.data_ms, f"{subj}__fw_corr_{args.metric.lower()}.nii.gz")

        ss_img = nib.load(ss_path)
        ms_img = nib.load(ms_path)
        ss_data = ss_img.get_fdata()
        ms_data = ms_img.get_fdata()
        affine = ss_img.affine

        valid_mask = ~np.isnan(ms_data)

        # ---------------------------
        # Recherche automatique du meilleur lambda
        # ---------------------------
        best_corr = -1
        best_lambda = args.lambda_min
        best_rgd = None
        lambda_values = np.arange(args.lambda_min, args.lambda_max + 1e-5, args.lambda_step)

        for lam in lambda_values:
            rgd_map = rgd_regularize(ss_data, lambda_reg=lam, n_iter=args.n_iter, step_size=args.step_size, clip_range=clip_range)
            corr = np.corrcoef(rgd_map[valid_mask].flatten(), ms_data[valid_mask].flatten())[0,1]
            if corr > best_corr:
                best_corr = corr
                best_lambda = lam
                best_rgd = rgd_map

        print(f"-> Meilleur lambda : {best_lambda:.2f}, corr = {best_corr:.3f}")

        # ---------------------------
        # Scaling voxel-wise
        # ---------------------------
        A = np.vstack([best_rgd[valid_mask].flatten(), np.ones(np.sum(valid_mask))]).T
        b, c = np.linalg.lstsq(A, ms_data[valid_mask].flatten(), rcond=None)[0]
        scaled_map = np.clip(best_rgd * b + c, clip_range[0], clip_range[1])

        # ---------------------------
        # Bootstrap pour cartes d'incertitude
        # ---------------------------
        uncertainty_map = bootstrap_uncertainty(ss_data, n_boot=args.n_boot, sigma_noise=args.sigma_noise,
                                                lambda_reg=best_lambda, n_iter=args.n_iter, step_size=args.step_size,
                                                clip_range=clip_range)

        # ---------------------------
        # Sauvegarde cartes
        # ---------------------------
        nib.save(nib.Nifti1Image(best_rgd, affine), os.path.join(args.output_dir, f"{subj}_{args.metric}-t_RGD.nii.gz"))
        nib.save(nib.Nifti1Image(scaled_map, affine), os.path.join(args.output_dir, f"{subj}_{args.metric}-t_RGD_scaled.nii.gz"))
        nib.save(nib.Nifti1Image(uncertainty_map, affine), os.path.join(args.output_dir, f"{subj}_{args.metric}-t_RGD_uncertainty.nii.gz"))

        # ---------------------------
        # Statistiques
        # ---------------------------
        mean_ss = np.mean(ss_data[valid_mask])
        mean_rgd = np.mean(best_rgd[valid_mask])
        mean_scaled = np.mean(scaled_map[valid_mask])
        mean_ms = np.mean(ms_data[valid_mask])
        mean_uncertainty = np.mean(uncertainty_map[valid_mask])

        std_ss = np.std(ss_data[valid_mask])
        std_rgd = np.std(best_rgd[valid_mask])
        std_scaled = np.std(scaled_map[valid_mask])
        std_ms = np.std(ms_data[valid_mask])

        corr_before = np.corrcoef(ss_data[valid_mask].flatten(), ms_data[valid_mask].flatten())[0,1]
        corr_rgd = np.corrcoef(best_rgd[valid_mask].flatten(), ms_data[valid_mask].flatten())[0,1]
        corr_scaled = np.corrcoef(scaled_map[valid_mask].flatten(), ms_data[valid_mask].flatten())[0,1]

        results.append({
            "subject": subj,
            "best_lambda": best_lambda,
            "mean_ss": mean_ss,
            "mean_RGD": mean_rgd,
            "mean_scaled": mean_scaled,
            "mean_ms": mean_ms,
            "mean_uncertainty": mean_uncertainty,
            "std_ss": std_ss,
            "std_RGD": std_rgd,
            "std_scaled": std_scaled,
            "std_ms": std_ms,
            "corr_before": corr_before,
            "corr_RGD": corr_rgd,
            "corr_scaled": corr_scaled
        })

    # ---------------------------
    # Sauvegarde CSV
    # ---------------------------
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, f"{args.metric}-t_RGD_scaled_uncertainty_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"Pipeline terminé. Résultats sauvegardés dans {args.output_dir}")

if __name__ == "__main__":
    main()
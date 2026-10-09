#!/bin/sh

#SBATCH --mail-user=stanislasth@outlook.fr
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-descotea
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=256GB

parallel -P 10 <  recon_all_clinical_all_in_one.sh

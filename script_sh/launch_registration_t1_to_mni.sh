#!/bin/sh

#SBATCH --mail-user=stanislasth@outlook.fr
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-descotea
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --mem=256GB

parallel -P 40 < t1_to_mni_registration.sh

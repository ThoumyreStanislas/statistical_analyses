#!/bin/sh

#SBATCH --mail-user=stanislasth@outlook.fr
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-descotea
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=128GB

zip -r data_ADNI3_abstract_OHBM.zip data_ADNI3_abstract_OHBM

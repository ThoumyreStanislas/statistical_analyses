#!/bin/sh

#SBATCH --mail-user=stanislasth@outlook.fr
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-descotea
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --mem=128GB

module load StdEnv/2020
module load nextflow/21.10.3

nextflow ~/librairies/tractometry_flow/main.nf --input raw_AD_T1 --use_provided_centroids True \
 --output_dir results_tm_AD_T1 -with-singularity ~/scratch/containers_scilus_1.6.0.sif \
 -w results_tm_AD_T1/work -resume

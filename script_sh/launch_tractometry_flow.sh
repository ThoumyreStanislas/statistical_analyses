#!/bin/sh


#SBATCH --mail-user=stanislasth@outlook.fr
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-descotea
#SBATCH --cpus-per-task=40
#SBATCH --nodes=4
#SBATCH --time=10:00:00
#SBATCH --mem=128GB

module load StdEnv/2020
module load nextflow/21.10.3

nextflow ~/librairies/tractometry_flow/main.nf --input tractometry_flow/raw_CN_T1/ --use_provided_centroids True \
 --output_dir tractometry_flow/results_tm_CN_T1 -with-singularity ~/scratch/containers_scilus_1.6.0.sif \
 -w tractometry_flow/work -resume 

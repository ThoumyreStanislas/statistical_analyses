#!/bin/sh

#SBATCH --mail-user=stanislasth@outlook.fr
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-descotea
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=128GB

module load StdEnv/2020
module load nextflow/21.10.3

nextflow ~/librairies/rbx_flow/main.nf --input RBx_flow/raw --atlas_directory ../atlas \
 -with-singularity ~/scratch/containers_scilus_1.6.0.sif -w RBx_flow/work -resume

#!/bin/sh

#SBATCH --mail-user=stanislasth@outlook.fr
#SBATCH --mail-type=ALL
#SBATCH --account=def-descotea
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=64GB

module load StdEnv/2020
module load nextflow/21.10.3

nextflow ~/libraries/flow/dmriqc_flow/main.nf --input results_tf/  \
    -with-singularity ~/projects/rrg-descotea/containers/scilus_1.6.0.sif  \
    -profile input_qc,tractoflow_qc_all \
    -resume
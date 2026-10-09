#!/bin/sh

#SBATCH --mail-user=stanislasth@outlook.fr
#SBATCH --mail-type=ALL
#SBATCH --account=def-descotea
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --mem=256GB

module load StdEnv/2020
module load nextflow/21.10.3

nextflow ~/libraries/flow/tractoflow/main.nf --input input/  \
    -with-singularity ~/projects/rrg-descotea/containers/scilus_1.6.0.sif  \
    --output_dir results_tf --run_pft_tracking false --run_local_tracking true \
    -resume
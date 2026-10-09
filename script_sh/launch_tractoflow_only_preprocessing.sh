#!/bin/sh

#SBATCH --mail-user=stanislasth@outlook.fr
#SBATCH --mail-type=ALL
#SBATCH --account=def-descotea
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

module load StdEnv/2020
module load nextflow/21.10.3

nextflow ~/libraries/flow/tractoflow/main.nf --input raw --output_dir results_tf \
    -profile only_preprocessing  -with-singularity ~/projects/rrg-descotea/containers/scilus_1.6.0.sif -resume

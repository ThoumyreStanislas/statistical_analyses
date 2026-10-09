#!/bin/sh

#SBATCH --mail-user=stanislasth@outlook.fr
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-descotea
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=256GB

module load StdEnv/2020
module load nextflow/21.10.3

nextflow ~/libraries/flow/freewater_flow/main.nf --input raw \
 -with-singularity ~/projects/rrg-descotea/containers/scilus_2.1.0.sif \
 --nb_threads 10 -resume

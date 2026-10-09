#!/bin/sh

#SBATCH --mail-user=stanislasth@outlook.fr
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-descotea
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=128GB

module load StdEnv/2020
module load nextflow/21.10.3

nextflow ~/libraries/flow/noddi_flow/main.nf --input raw \
 -with-singularity ~/projects/rrg-descotea/containers/scilus_2.1.1.sif \
 --b_thr 20 -resume
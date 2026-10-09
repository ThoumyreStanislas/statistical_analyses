#!/bin/sh

#SBATCH --mail-user=stanislasth@outlook.fr
#SBATCH --mail-type=ALL
#SBATCH --account=def-frheault
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

for subject in *; do \
	for image in ${subject}/*/*/*/; do dcm2niix -z o ${image} ; done 
	for session in ${subject}/*/*; do \
		ses=$(basename ${session});
		mkdir -p ~/scratch/databases/data_all_in_one/${subject}/${ses%%_*};
		mv ${subject}/*DTI*/${ses}/*/*.nii.gz ~/scratch/databases/data_all_in_one/${subject}/${ses%%_*}/dwi.nii.gz ;
		mv ${subject}/*-SPGR/${ses}/*/*.nii.gz ~/scratch/databases/data_all_in_one/${subject}/${ses%%_*}/t1.nii.gz ;
		mv ${subject}/*FLAIR*/${ses}/*/*.nii.gz ~/scratch/databases/data_all_in_one/${subject}/${ses%%_*}/flair.nii.gz ;
		mv ${subject}/*T2*/${ses}/*/*.nii.gz ~/scratch/databases/data_all_in_one/${subject}/${ses%%_*}/t2.nii.gz ; done ; done



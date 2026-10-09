#!/bin/bash
#SBATCH --mail-user=thos2102@usherbrooke.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=v100:1
#SBATCH --account=rrg-descotea

python ~/librairies/SHiVAi/apptainer/run_shiva.py --in raw/ --out shiva_results --input_type standard --prediction WMH \
	 --brain_seg shiva_gpu --config ~/librairies/SHiVAi/apptainer/config_example.yml --remove_intermediates 


#shiva --in raw/ --out shiva_results --input_type standard --prediction WMH --brain_seg shiva_gpu --config ~/librairies/SHiVAi/apptainer/config_example.yml --remove_intermediates --containerized_nodes

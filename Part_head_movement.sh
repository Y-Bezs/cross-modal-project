#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --account=jenseno-opm
#SBATCH --qos bbdefault
#SBATCH --time 3:00:00
#SBATCH --mem-per-cpu=72G
#SBATCH --mail-type=END
#SBATCH --array 31-35

set -e

module purge; module load bluebear
module load bear-apps/2021b
module load Python/3.9.6-GCCcore-11.2.0
module load SciPy-bundle/2021.10-foss-2021b
module load MNE-Python/1.1.1-foss-2021b



python Part_head_movement.py ${SLURM_ARRAY_TASK_ID}

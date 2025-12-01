#!/bin/bash
#SBATCH -p gpu_lowp
#SBATCH --job-name=thinned_mfld
#SBATCH --time=02:00:00        
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --chdir=/nfs/ghome/live/jwornbard/hudson
#SBATCH --output=thinned_mfld_%A_%a.out
#SBATCH --error=thinned_mfld_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-sr675-[31,33-34]


# Get the line corresponding to this array task
JOB_PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Job params: $JOB_PARAMS"
eval "$(/nfs/ghome/live/jwornbard/.local/miniforge3/bin/conda shell.bash hook 2>/dev/null)"
conda activate thinned_mfld

date

## Check if the environment is correct.
which pip
which python

python /nfs/ghome/live/jwornbard/hudson/thinned_mfld/main.py $JOB_PARAMS

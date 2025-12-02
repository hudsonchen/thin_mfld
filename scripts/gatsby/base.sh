#!/bin/bash
#SBATCH -p cpu
#SBATCH --job-name=thinned_mfld
#SBATCH --time=10:00:00        
#SBATCH --cpus-per-task=64
#SBATCH --mem=10G
#SBATCH --chdir=/nfs/ghome/live/jwornbard/hudson
#SBATCH --output=thinned_mfld_%A_%a.out
#SBATCH --error=thinned_mfld_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclusive

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"

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

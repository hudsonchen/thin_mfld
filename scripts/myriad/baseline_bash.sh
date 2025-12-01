#$ -l mem=10G
#$ -pe smp 8
#$ -l h_rt=1:00:0
#$ -R y
#$ -S /bin/bash
#$ -wd /home/ucabzc9/Scratch/
#$ -j y
#$ -N thinned_mfld

JOB_PARAMS=$(sed "${SGE_TASK_ID}q;d" "$1")
echo "Job params: $JOB_PARAMS"

# Running date and nvidia-smi is useful to get some info in case the job crashes.

# module unload compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load compilers/gnu/10.2.0

## Load conda
module -f unload compilers
module load compilers/gnu/4.9.2
module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate thinned_mfld

date

## Check if the environment is correct.
which pip
which python

python /home/ucabzc9/Scratch/thinned_mfld/main.py $JOB_PARAMS
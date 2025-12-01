#!/bin/bash

jobs_in_parallel=$(wc -l < "$1")
echo $jobs_in_parallel
echo $1

# Submit a SLURM job array
sbatch --array=1-${jobs_in_parallel} /nfs/ghome/live/jwornbard/hudson/thinned_mfld/scripts/myriad/vlm_base.sh "$1"

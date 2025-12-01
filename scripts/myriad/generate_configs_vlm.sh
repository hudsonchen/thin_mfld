#!/bin/bash

# Output config file
OUT_FILE="$HOME/thinned_mfld/scripts/myriad/vlm_configs.txt"

# Truncate the file first
> "$OUT_FILE"

for seed in {0..9}
do
  for kernel in sobolev
  do
    for g in 0
    do
      for particle_num in 16 64 256 1024
      do
        for zeta in 1.0 0.1 0.01
        do
          echo "--seed $seed --dataset vlm --g $g --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --bandwidth 1.0 --step_num 200 --thinning kt --kernel $kernel --zeta $zeta" >> "$OUT_FILE"
        done
      done
    done
  done
done

for seed in {0..9}
do
  for thinning in false random
  do
    for particle_num in 16 64 256 1024
    do
      for zeta in 1.0 0.1 0.01
        do
          echo "--seed $seed --dataset vlm --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --step_num 200 --thinning $thinning --kernel sobolev --zeta $zeta" >> "$OUT_FILE"
        done
    done
  done
done
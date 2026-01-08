#!/bin/bash

# Output config file
OUT_FILE="$HOME/thinned_mfld/scripts/gatsby/mmd_flow_configs.txt"

# Truncate the file first
> "$OUT_FILE"


for seed in {0..4}
do
  for thinning in false kt random
  do
    for particle_num in 64 256 1024
    do
      for zeta in 0.0
        do
          for noise_scale in 3e-4
            do
              echo "--seed $seed --dataset mmd_flow --particle_num $particle_num --step_size 1.0 --noise_scale $noise_scale --step_num 15000 --thinning $thinning --kernel gaussian --zeta $zeta" >> "$OUT_FILE"
            done
        done
    done
  done
done

for seed in {0..4}
do
  for thinning in kt random
  do
    for particle_num in 4096
    do
      for zeta in 0.0
        do
        for noise_scale in 3e-4
          do
            echo "--seed $seed --dataset mmd_flow --particle_num $particle_num --step_size 1.0 --noise_scale $noise_scale --step_num 15000 --thinning $thinning --kernel gaussian --zeta $zeta" >> "$OUT_FILE"
          done
        done
    done
  done
done

for seed in {0..4}
do
  for thinning in rbm
  do
    for particle_num in 64 256 1024 4096
    do
      for zeta in 0.0
        do
          for noise_scale in 3e-4
            do
              echo "--seed $seed --dataset mmd_flow --particle_num $particle_num --step_size 1.0 --noise_scale $noise_scale --step_num 15000 --thinning $thinning --kernel gaussian --zeta $zeta" >> "$OUT_FILE"
            done
        done
    done
  done
done

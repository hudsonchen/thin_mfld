for seed in 3 4 5 6 7 8 9
do
  for thinning in false random
  do
    for particle_num in 64 256
    do
      for zeta in 1.0 0.1 0.01
        do
          /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --seed $seed --dataset vlm --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --step_num 200 --thinning $thinning --kernel sobolev --zeta $zeta
        done
    done
  done
done

for seed in 3 4 5 6 7 8 9
do
  for kernel in sobolev
  do
    for g in 0
    do
      for particle_num in 64 256
      do
        for zeta in 1.0 0.1 0.01
          do
            /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --seed $seed --dataset vlm --g $g --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --bandwidth 1.0 --step_num 200 --thinning kt --kernel $kernel --zeta $zeta
          done
      done
    done
  done
done
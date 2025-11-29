# for seed in 0 1 2
# do
#   for thinning in random
#   do
#     for particle_num in 64
#     do
#       for zeta in 0.1 0.01 0.001
#         do
#           /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --seed $seed --dataset vlm --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --step_num 100 --thinning $thinning --kernel sobolev --zeta $zeta
#         done
#     done
#   done
# done

for seed in 0 1 2
do
  for kernel in gaussian
  do
    for g in 0
    do
      for particle_num in 64
      do
        for zeta in 0.1 0.01 0.001
          do
            /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --seed $seed --dataset vlm --g $g --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --bandwidth 1.0 --step_num 100 --thinning kt --kernel $kernel --zeta $zeta
          done
      done
    done
  done
done
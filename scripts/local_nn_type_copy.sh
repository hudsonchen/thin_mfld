for seed in 0 1 2
do
  for kernel in gaussian
    do
      for g in 0
      do
      for particle_num in 16 64 1024
      do
      /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --dataset covertype --seed $seed --g $g --particle_num $particle_num --step_size 0.01 --noise_scale 0.001 --bandwidth 0.3 --step_num 50 --thinning kt --kernel $kernel
      done
    done
    done
done

# for seed in 0 1 2
# do
#   for thinning in random false
#   do
#     for particle_num in 16 64 1024
#     do
#     /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --dataset covertype --seed $seed --particle_num $particle_num --step_size 0.01 --noise_scale 0.001 --step_num 50 --thinning $thinning --kernel sobolev
#     done
#   done
# done
for seed in 0 2
do
  for thinning in random false
  do
    /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --seed $seed --particle_num 1024 --step_size 0.01 --noise_scale 0.01 --step_num 10000 --thinning $thinning --kernel sobolev
  done
done

for seed in 0 2
do
  for kernel in sobolev
    do
      for g in 0 1
      do
      /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --seed $seed --g $g --particle_num 1024 --step_size 0.01 --noise_scale 0.01 --bandwidth 1.0 --step_num 10000 --thinning kt --kernel $kernel
      done
    done
done
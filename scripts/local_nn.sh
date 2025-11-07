for seed in 0 1 2
do
  for thinning in random false
  do
    /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --seed $seed --particle_num 64 --step_size 0.001 --noise_scale 0.0 --step_num 10000 --thinning $thinning --kernel sobolev
  done
done

for seed in 0 1 2
do
  for bandwidth in 0.1 1.0
  do
    for kernel in sobolev gaussian
    do
      /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --seed $seed --particle_num 64 --step_size 0.001 --noise_scale 0.0 --bandwidth $bandwidth --step_num 10000 --thinning kt --kernel $kernel
    done
  done
done
for seed in 0 1 2
do
  for thinning in kt random false
  do
    /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --seed $seed --particle_num 1024 --step_size 0.1 --noise_scale 0.1 --step_num 10000 --thinning $thinning --kernel sobolev
  done
done
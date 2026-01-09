for seed in {0..4}
do
  for thinning in kt random rbm
  do
    for particle_num in 4096
    do
      for zeta in 0.1
        do
          /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --seed $seed --dataset vlm --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --step_num 200 --thinning $thinning --kernel sobolev --zeta $zeta --save_path './results/'
        done
    done
  done
done

# for seed in {0..2}
# do
#   for kernel in sobolev
#   do
#     for g in 0
#     do
#       for particle_num in 16 64 256 1024
#       do
#         for zeta in 1.0
#           do
#             /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --seed $seed --dataset vlm --g $g --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --bandwidth 1.0 --step_num 200 --thinning kt --kernel $kernel --zeta $zeta --save_path './results/'
#           done
#       done
#     done
#   done
# done


# for seed in {0..2}
# do
#   for kernel in sobolev
#   do
#     for g in 0
#     do
#       for particle_num in 16 64 256 1024
#       do
#         for zeta in 1.0
#           do
#             /home/zongchen/miniconda3/envs/thin_mfld/bin/python main.py --seed $seed --dataset vlm --g $g --particle_num $particle_num --step_size 0.0001 --noise_scale 0.001 --bandwidth 1.0 --step_num 200 --thinning kt --kernel $kernel --zeta $zeta --save_path './results/'
#           done
#       done
#     done
#   done
# done
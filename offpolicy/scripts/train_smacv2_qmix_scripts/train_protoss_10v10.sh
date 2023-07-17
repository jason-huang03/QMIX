env="StarCraft2v2"
map="10gen_protoss"
algo="qmix"
units="10v10"

exp="tune2"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python ../train/train_smac.py --env_name ${env} \
     --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --units ${units}\
      --seed 1 --n_training_threads 128 --buffer_size 5000 --lr 5e-4 --batch_size 32 --use_soft_update \
       --hard_update_interval_episode 200 --num_env_steps 10000000 \
       --log_interval 3000 --eval_interval 20000 \
       --use_global_all_local_state --gain 1 \
       --user_name "jason-huang03"
    echo "training is done!"
done


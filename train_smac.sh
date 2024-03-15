env='smac'
scenerio='8m'
seed=0
exp_name="debug"

# MCTS
K=10
N=100
mcts_rho=0.25
mcts_lambda=0.8

# AWPO
PG_type='sharp'
awac_lambda=2
adv_clip=3.0

python main.py --opr train_sync --case $env --env_name $scenerio --exp_name $exp_name --seed $seed \
    --num_cpus 64 --num_gpus 1 --train_on_gpu --reanalyze_on_gpu --selfplay_on_gpu \
    --data_actors 1 --num_pmcts 4 --reanalyze_actors 30 \
    --test_interval 1000 --test_episodes 32 --target_model_interval 200 \
    --batch_size 256 --num_simulations $N --sampled_action_times $K \
    --training_steps 1000000 --last_step 10000 --lr 5e-4 --lr_adjust_func const --max_grad_norm 5 \
    --total_transitions 2000000 --start_transition 500 --discount 0.99 \
    --target_value_type pred-re --revisit_policy_search_rate 1.0 --use_off_correction \
    --value_transform_type vector --use_mcts_test \
    --use_priority --use_max_priority \
    --PG_type $PG_type --awac_lambda $awac_lambda --adv_clip $adv_clip \
    --mcts_rho $mcts_rho --mcts_lambda $mcts_lambda

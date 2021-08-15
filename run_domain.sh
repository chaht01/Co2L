# representation learning
python main_domain.py --learning_rate 0.01 --temp 0.1  --current_temp 0.2 --past_temp 0.01 --n_task 20 --cosine --syncBN --mem_size 200 --start_epoch 100 --epochs 20 --batch_size 512

# linear eval
python main_linear_buffer_domain.py --learning_rate 1 --ckpt save_domain_random_500/r-mnist_models/r-mnist_mlp_lr_0.01_decay_0.0001_bsz_512_temp_0.1_trial_0_100_20_cosine --logpt save_domain_random_500/logs/r-mnist_mlp_lr_0.01_decay_0.0001_bsz_512_temp_0.1_trial_0_100_20_cosine --target_task 19

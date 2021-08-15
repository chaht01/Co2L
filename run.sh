# representation learning
python main.py --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 200 --epochs 100 --start_epoch 500 --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine --syncBN

# linear eval
python main_linear_buffer.py --learning_rate 1 --target_task 4 --ckpt ./save_random_200/cifar10_models/cifar10_32_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.5_momentum_1.000_trial_0_500_100_0.2_0.01_1.0_cosine_warm/ --logpt ./save_random_200/logs/cifar10_32_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.5_momentum_1.000_trial_0_500_100_0.2_0.01_1.0_cosine_warm/

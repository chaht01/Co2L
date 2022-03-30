# Co^2L: Contrastive Continual Learning (ICCV 2021)

Code for [Co^2L: Contrastive Continual Learning](https://arxiv.org/abs/2106.14413). 
Our code is based on the implementation of [Supervised Contrastive Learning](https://github.com/HobbitLong/SupContrast). 

If you find this code useful, please reference in our paper:

```
@InProceedings{Cha_2021_ICCV,
    author    = {Cha, Hyuntak and Lee, Jaeho and Shin, Jinwoo},
    title     = {Co2L: Contrastive Continual Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {9516-9525}
}
```

# Instruction

Different from other continual learning methods, Co^2L needs pre-training part for learning representations since Co^2L based on contrastive representation learning schemes. Thus, you can get the results reported on our paper from linear evaluation with pre-trained representations. Please follow below two commands. 

## Representation Learning
```
python main.py --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 200 --epochs 100 --start_epoch 500 --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine --syncBN
```

## Linear Evaluation
```
python main_linear_buffer.py --learning_rate 1 --target_task 4 --ckpt ./save_random_200/cifar10_models/cifar10_32_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.5_momentum_1.000_trial_0_500_100_0.2_0.01_1.0_cosine_warm/ --logpt ./save_random_200/logs/cifar10_32_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.5_momentum_1.000_trial_0_500_100_0.2_0.01_1.0_cosine_warm/
```

# Issue

If you have troulbe with NaN loss while training representation learning, you may find solutions from [SupCon issue page](https://github.com/HobbitLong/SupContrast/issues). Please check your training works perfectly on SupCon first. 

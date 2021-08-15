'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/main_ce.py
'''

from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
from torchvision import transforms, datasets
from datasets import TinyImagenet


def set_loader(opt, replay_indices):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'tiny-imagenet':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))


    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size,opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task)) # tasks learned so far.

    if opt.dataset == 'cifar10':
        subset_indices = []
        _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)

        _train_targets = np.array(_train_dataset.targets)
        for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()
        subset_indices += replay_indices.tolist()

        ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
        print(ut)
        print(uc)

        weights = np.array([0.] * len(subset_indices))
        for t, c in zip(ut, uc):
            weights[_train_targets[subset_indices] == t] = 1./c

        train_dataset =  Subset(_train_dataset, subset_indices)

        subset_indices = []
        _val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)

    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        _train_dataset = TinyImagenet(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)

        _train_targets = np.array(_train_dataset.targets)
        for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()
        subset_indices += replay_indices.tolist()

        ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
        print(ut)
        print(uc)

        weights = np.array([0.] * len(subset_indices))
        for t, c in zip(ut, uc):
            weights[_train_targets[subset_indices] == t] = 1./c

        train_dataset =  Subset(_train_dataset, subset_indices)

        subset_indices = []
        _val_dataset = TinyImagenet(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)

    else:
        raise ValueError(opt.dataset)

    train_sampler = WeightedRandomSampler(torch.Tensor(weights), len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader, uc


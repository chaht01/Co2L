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
import torchvision
from torch.utils.data import Dataset, Subset, WeightedRandomSampler, ConcatDataset
from torchvision import transforms, datasets

def set_loader(opt, replay_indices, replay_degrees, degree_list):
    # construct data loader
    class FixedRotation(object):
        """
        Defines a fixed rotation for a numpy array.
        """

        def __init__(self, deg):
            """
            Initializes the rotation with a random angle.
            :param deg_min: lower extreme of the possible random angle
            :param deg_max: upper extreme of the possible random angle
            """
            self.degrees = deg

        def __call__(self, x):
            """
            Applies the rotation.
            :param x: image to be rotated
            :return: rotated image
            """
            return torchvision.transforms.functional.rotate(x, self.degrees)

    unique_degrees, unique_counts = np.unique(replay_degrees, return_counts=True)
    subsets = []
    weights = []
    for degree, count in zip(unique_degrees, unique_counts):
        mask = np.array(replay_degrees) == degree
        masked_replay_indices = np.array(replay_indices)[mask]
        masked_weights = [1./count] * len(masked_replay_indices)
        weights += masked_weights
        prev_train_dataset = datasets.MNIST(root=opt.data_folder,
                                            transform=
                                              transforms.Compose([
                                                FixedRotation(degree),
                                                transforms.RandomResizedCrop(size=28, scale=(0.7, 1.)),
                                                transforms.ToTensor()
                                              ])
                                            ,
                                            download=True)
        subsets.append(Subset(prev_train_dataset, masked_replay_indices.tolist()))
    _train_dataset = datasets.MNIST(root=opt.data_folder, transform=transforms.Compose([
                                                FixedRotation(degree_list[-1]),
                                                transforms.ToTensor()
                                              ]), download=True)
    train_dataset = ConcatDataset([*subsets, _train_dataset])
    weights += [1. / len(_train_dataset)] * len(_train_dataset)

    val_sets = []
    for degree in degree_list:
        val_transform = transforms.Compose([
            FixedRotation(degree_list[-1]),
            transforms.ToTensor(),
        ])

        _val_dataset = datasets.MNIST(root=opt.data_folder, train=False, transform=val_transform, download=True)
        val_sets.append(_val_dataset)
    val_dataset = ConcatDataset(val_sets)
    print('val set size', len(val_dataset))


    train_sampler = WeightedRandomSampler(torch.Tensor(weights), len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader



'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
'''

from __future__ import print_function

import os
import copy
import sys
import argparse
import time
import math
import random
import numpy as np

import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset, WeightedRandomSampler, ConcatDataset

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, load_model
from networks.mlp import SupConMLP
from losses_negative_only import SupConLoss


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--target_task', type=int, default=0)

    parser.add_argument('--resume_target_task', type=int, default=None)

    parser.add_argument('--replay_policy', type=str, choices=['random'], default='random')

    parser.add_argument('--mem_size', type=int, default=200)

    parser.add_argument('--cls_per_task', type=int, default=10)

    parser.add_argument('--n_task', type=int, default=20)

    parser.add_argument('--distill_power', type=float, default=1.0)

    parser.add_argument('--current_temp', type=float, default=0.2,
                        help='temperature for loss function')

    parser.add_argument('--past_temp', type=float, default=0.01,
                        help='temperature for loss function')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=None)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--dataset', type=str, default='r-mnist',
                        choices=['r-mnist'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=28, help='parameter for RandomResizedCrop')



    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    opt.save_freq = opt.epochs // 2


    if opt.dataset == 'r-mnist':
        opt.n_cls = 10
        opt.cls_per_task = 10
    else:
        pass


    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '~/data/'
    opt.model_path = './save_domain_{}_{}/{}_models'.format(opt.replay_policy, opt.mem_size, opt.dataset)
    opt.tb_path = './save_domain_{}_{}/{}_tensorboard'.format(opt.replay_policy, opt.mem_size, opt.dataset)
    opt.log_path = './save_domain_{}_{}/logs'.format(opt.replay_policy, opt.mem_size, opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_{}_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp,
               opt.trial,
               opt.start_epoch if opt.start_epoch is not None else opt.epochs, opt.epochs
               )

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)


    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_folder = os.path.join(opt.log_path, opt.model_name)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    return opt


def set_replay_samples(opt, model, prev_indices=None, prev_degrees=None, degree_list=None):


    class IdxDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            return self.indices[idx], self.dataset[idx]

    # construct data loader
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])


    val_dataset = datasets.MNIST(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)


    observed_so_far_cnt = 0
    if opt.target_task == 0:
        return [], []
    else:
        if opt.target_task == 1:
            return torch.randperm(len(val_dataset))[:opt.mem_size].tolist(), [degree_list[-1]] * opt.mem_size
        else:
            print("reservoir start")
            observed_so_far_cnt = (opt.target_task - 1) * len(val_dataset)
            ret_indices = copy.deepcopy(prev_indices)
            ret_degrees = copy.deepcopy(prev_degrees)

            for idx in range(len(val_dataset)):
                p = random.random()
                p = random.randint(0, observed_so_far_cnt-1)
                if p < opt.mem_size:
                    ret_indices[p] = idx
                    ret_degrees[p] = degree_list[-1]
                observed_so_far_cnt += 1

            return ret_indices, ret_degrees


def set_loader(opt, replay_indices=None, replay_degrees=None):
    # construct data loader
    class DomainDataset(Dataset):
        def __init__(self, dataset, domain_id):
            self.dataset = dataset
            self.domain_id = domain_id
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            return self.domain_id, self.dataset[idx]

    class Rotation(object):
        """
        Defines a fixed rotation for a numpy array.
        """

        def __init__(self, deg_min = 0, deg_max = 180):
            """
            Initializes the rotation with a random angle.
            :param deg_min: lower extreme of the possible random angle
            :param deg_max: upper extreme of the possible random angle
            """
            self.deg_min = deg_min
            self.deg_max = deg_max
            self.degrees = np.random.uniform(self.deg_min, self.deg_max)

        def __call__(self, x):
            """
            Applies the rotation.
            :param x: image to be rotated
            :return: rotated image
            """
            return torchvision.transforms.functional.rotate(x, self.degrees)

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

    rot = Rotation()
    print(rot.degrees)
    train_transform = transforms.Compose([
      rot,
      transforms.RandomResizedCrop(size=28, scale=(0.7, 1.)),
      transforms.ToTensor()
    ])

    if opt.target_task > 0:
        unique_degrees = np.unique(replay_degrees)
        subsets = []
        for domain_id, degree in enumerate(unique_degrees):
            mask = np.array(replay_degrees) == degree
            masked_replay_indices = np.array(replay_indices)[mask]


            prev_train_dataset = datasets.MNIST(root=opt.data_folder,
                                                transform=TwoCropTransform(
                                                  transforms.Compose([
                                                    FixedRotation(degree),
                                                    transforms.RandomResizedCrop(size=28, scale=(0.7, 1.)),
                                                    transforms.ToTensor()
                                                  ])
                                                ),
                                                download=True)
            subsets.append(DomainDataset(Subset(prev_train_dataset, masked_replay_indices.tolist()), domain_id))
        _train_dataset = DomainDataset(datasets.MNIST(root=opt.data_folder, transform=TwoCropTransform(train_transform), download=True), len(unique_degrees))

        train_dataset = ConcatDataset([*subsets, _train_dataset])
    else:
        train_dataset = DomainDataset(datasets.MNIST(root=opt.data_folder, transform=TwoCropTransform(train_transform), download=True), 0)


    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    print('loader dataset size', len(train_loader.dataset))
    return train_loader, rot.degrees



def set_model(opt):
    model = SupConMLP()
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, model2, criterion, optimizer, epoch, opt):


    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (domain_labels, (images, labels)) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels + domain_labels * opt.cls_per_task
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        with torch.no_grad():
            prev_task_mask = labels < opt.target_task * opt.cls_per_task
            prev_task_mask = prev_task_mask.repeat(2)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features, encoded = model(images, return_feat=True)

        # IRD (current)
        if opt.target_task > 0:
            features1_prev_task = features

            features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), opt.current_temp)
            logits_mask = torch.scatter(
                torch.ones_like(features1_sim),
                1,
                torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                0
            )
            logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
            features1_sim = features1_sim - logits_max1.detach()
            row_size = features1_sim.size(0)
            logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

        # Asym SupCon
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels, target_labels=list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task)))

        # IRD (past)
        if opt.target_task > 0:
            with torch.no_grad():
                features2_prev_task = model2(images)

                features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), opt.past_temp)
                logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                features2_sim = features2_sim - logits_max2.detach()
                logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

            loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
            loss += opt.distill_power * loss_distill

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0 or idx +1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg, model2


def main():
    opt = parse_option()

    target_task = opt.target_task

    # build model and criterion
    model, criterion = set_model(opt)
    model2, _ = set_model(opt)
    model2.eval()

    # build optimizer
    optimizer = set_optimizer(opt, model)

    replay_indices = []
    replay_degrees = []
    degree_list = []

    if opt.resume_target_task is not None:
        load_file = os.path.join(opt.save_folder, 'last_{policy}_{target_task}.pth'.format(policy=opt.replay_policy ,target_task=opt.resume_target_task))
        model, optimizer = load_model(model, optimizer, load_file)
        replay_indices = np.load(
          os.path.join(opt.log_folder, 'replay_indices_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=opt.resume_target_task))
        ).tolist()
        replay_degrees = np.load(
          os.path.join(opt.log_folder, 'replay_degrees_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=opt.resume_target_task))
        ).tolist()
        degree_list = np.load(
          os.path.join(opt.log_folder, 'degree_list_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=opt.resume_target_task))
        ).tolist()
        print(len(replay_indices), len(replay_degrees), len(degree_list))
        print(np.unique(replay_degrees, return_counts=True))

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    original_epochs = opt.epochs

    for target_task in range(0 if opt.resume_target_task is None else opt.resume_target_task+1, opt.n_task):
        opt.target_task = target_task
        model2 = copy.deepcopy(model)

        print('Start Training current task {}'.format(opt.target_task))

        # acquire replay sample indices
        replay_indices, replay_degrees = set_replay_samples(opt, model, prev_indices=replay_indices, prev_degrees=replay_degrees, degree_list=degree_list)
        print(len(replay_indices), len(replay_degrees))
        print(np.unique(replay_degrees, return_counts=True))
        print(degree_list)

        # build data loader (dynamic: 0109)
        train_loader, curr_degree = set_loader(opt, replay_indices, replay_degrees)
        degree_list.append(curr_degree)
        np.save(
          os.path.join(opt.log_folder, 'replay_indices_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=target_task)),
          np.array(replay_indices))
        np.save(
          os.path.join(opt.log_folder, 'replay_degrees_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=target_task)),
          np.array(replay_degrees))
        np.save(
          os.path.join(opt.log_folder, 'degree_list_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=target_task)),
          np.array(degree_list))


        # training routine
        if target_task == 0 and opt.start_epoch is not None:
            opt.epochs = opt.start_epoch
        else:
            opt.epochs = original_epochs

        for epoch in range(1, opt.epochs + 1):

            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, model2 = train(train_loader, model, model2, criterion, optimizer, epoch, opt)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # tensorboard logger
            logger.log_value('loss_{target_task}'.format(target_task=target_task), loss, epoch)
            logger.log_value('learning_rate_{target_task}'.format(target_task=target_task), optimizer.param_groups[0]['lr'], epoch)



        # save the last model
        save_file = os.path.join(
            opt.save_folder, 'last_{policy}_{target_task}.pth'.format(policy=opt.replay_policy ,target_task=target_task))
        save_model(model, optimizer, opt, opt.epochs, save_file)

if __name__ == '__main__':
    main()

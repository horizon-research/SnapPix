from coded.coded_layer import CodedLayer, CodedBaeline
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import json
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from packaging import version
from timm.models import create_model

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
import utils
from dataset import build_pretraining_dataset
from engine_for_mean_std import compute_mean_std
from optim_factory import create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_pretrain_samples_collate

import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(
        'Get Mean and Std of Pixels', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument(
        '--mask_type',
        default='tube',
        choices=['random', 'tube'],
        type=str,
        help='encoder masked strategy')
    parser.add_argument(
        '--decoder_mask_type',
        default='run_cell',
        choices=['random', 'run_cell'],
        type=str,
        help='decoder masked strategy')

    parser.add_argument(
        '--mask_ratio', default=0.9, type=float, help='mask ratio of encoder')
    parser.add_argument(
        '--decoder_mask_ratio',
        default=0.0,
        type=float,
        help='mask ratio of decoder')
    parser.add_argument(
        '--input_size',
        default=112,
        type=int,
        help='images input size for backbone')

    parser.add_argument(
        '--lr',
        type=float,
        default=1.5e-4,
        metavar='LR',
        help='learning rate (default: 1.5e-4)')
    
    parser.add_argument(
        '--wd',
        type=float,
        default=1.5e-4,
        metavar='weight decay',
        help='weight decay')

    # Dataset parameters
    parser.add_argument(
        '--data_path',
        default='/your/data/annotation/path',
        type=str,
        help='dataset path')
    # Dataset parameters
    parser.add_argument(
        '--val_path',
        default='/your/data/annotation/path',
        type=str,
        help='dataset path')
    # add output_dir
    parser.add_argument(
        '--output_dir',
        default='/your/output/dir',
        type=str,   
        help='output directory')
    
    parser.add_argument(
        '--data_root', default='', type=str, help='dataset path root')
    parser.add_argument(
        '--fname_tmpl',
        default='img_{:05}.jpg',
        type=str,
        help='filename_tmpl for rawframe data')
    parser.add_argument(
        '--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--num_sample', type=int, default=1)
    parser.add_argument(
        '--device',
        default='cuda',
        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help=
        'Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument(
        '--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument(
        '--patch_norm_coded', default=False, action='store_true')
    
    parser.add_argument(
        '--only_run_two_baselines', default=False, action='store_true')
    return parser.parse_args()


def main(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    patch_size = (8,8)
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 1,
                        args.input_size // patch_size[0],
                        args.input_size // patch_size[1])
    args.patch_size = patch_size

    # get dataset
    dataset_train = build_pretraining_dataset(args, decorrelation=True, fix_sample=True)

    dataset_val = build_pretraining_dataset(args, decorrelation=True, val=True)

    # Set the number of tasks to 1 for a single GPU setup
    num_tasks = 1
    global_rank = 0  # Since there's only one GPU, set the rank to 0
    sampler_rank = global_rank
    total_batch_size = args.batch_size * num_tasks
    # Calculate the number of training steps per epoch
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    # Use a regular (non-distributed) sampler since we’re only using one GPU
    sampler_train = torch.utils.data.RandomSampler(dataset_train)  # Replace DistributedSampler with RandomSampler
    print("Sampler_train = %s" % str(sampler_train))

    collate_func = None


    if args.num_workers > 0:
        persistent_workers = True
    else:
        persistent_workers = False

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_func,
        worker_init_fn=utils.seed_worker,
        persistent_workers=persistent_workers)
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_func,
        worker_init_fn=utils.seed_worker,
        persistent_workers=persistent_workers,
        shuffle=False)

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" %
          (total_batch_size * num_training_steps_per_epoch))
    


    for epoch in range(args.epochs):
        train_mean, train_std = compute_mean_std(data_loader_train, device='cuda')
        val_mean, val_std = compute_mean_std(data_loader_val, device='cuda')
        
        
        
    print(f"Train Mean: {train_mean:.4f} Train Std: {train_std:.4f}")
    print(f"Val Mean: {val_mean:.4f} Val Std: {val_std:.4f}")


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)

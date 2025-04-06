from coded.coded_layer import CodedLayer, CodedBaseline
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
from engine_for_decorrelation import train_one_epoch, val_one_epoch, val_one_epoch_with_matrix
from optim_factory import create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_pretrain_samples_collate

import matplotlib.pyplot as plt


def record_pattern(coded_layer, epoch, save_dir):
    pattern = coded_layer.get_pattern()
    # save sparsity to log , AttributeError: 'numpy.ndarray' object has no attribute 'numel'
    sparsity = 1 - pattern.sum() /  pattern.size
    print(f"Epoch[{epoch}]: Sparsity: {sparsity}\n")
    with open(os.path.join(save_dir, 'log.txt'), 'a') as f:
        f.write(f"Epoch[{epoch}]: Sparsity: {sparsity}\n")
    # save plot
    pattern_plot_path = os.path.join(save_dir, f'pattern_{epoch}.png')
    plot_4x4_grid_of_patterns(pattern, pattern_plot_path)
    # save pattern
    pattern_path = os.path.join(save_dir, f'pattern_{epoch}.pth')
    torch.save(pattern, pattern_path)

def plot_4x4_grid_of_patterns(pattern, save_path):
    """
    Plots a 4x4 grid of 8x8 patterns from a binary-coded weight matrix and saves the plot as an image.

    Parameters:
    - pattern (numpy.ndarray): The binary-coded weight matrix with shape (1, 16, 8, 8).
    - save_dir (str): The directory path where the plot image will be saved.

    Returns:
    - str: File path of the saved image.
    """
    # Check if the input pattern has the expected shape
    if pattern.shape != (1, 16, 8, 8):
        raise ValueError("Pattern must have shape (1, 16, 8, 8)")
    
    # File path for saving the image
    file_path = save_path

    # Create the plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("4x4 Grid of 8x8 Binary Coded Weight Patterns")

    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            single_pattern = pattern[0, i * 4 + j]
            ax.imshow(single_pattern, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(file_path)
    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser(
        'Decorrelation Training', add_help=False)
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
    parser.add_argument(
        '--test_pattern',
        default=False,
        action='store_true',
        help='Flag to test a specific pattern')

    parser.add_argument(
        '--pattern',
        default='',
        type=str,
        help='Path to the pattern file to be tested')

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
    dataset_train = build_pretraining_dataset(args, decorrelation=True)

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
        drop_last=True,
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

    if args.only_run_two_baselines:
        coded_layer = CodedBaseline()
        coded_layer.long_exposure = True
        record_pattern(coded_layer, "Long_Exposure", args.output_dir)
        decor_L1 = val_one_epoch(data_loader_val, device, coded_layer, patch_norm_coded=args.patch_norm_coded)
        print(f"Long_Exposure\t Val Decorrelation Loss: {decor_L1}")
        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(f"Long_Exposure\t Val Decorrelation Loss: {decor_L1}\n")
        coded_layer.long_exposure = False
        record_pattern(coded_layer, "Short_Exposure", args.output_dir)
        decor_L1 = val_one_epoch(data_loader_val, device, coded_layer, patch_norm_coded=args.patch_norm_coded)
        print(f"Short_Exposure\t Val Decorrelation Loss: {decor_L1}")
        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(f"Short_Exposure\t Val Decorrelation Loss: {decor_L1}\n")

        return

    if args.test_pattern:
        
        


        # Save an identity matrix heatmap first
        identity_matrix = np.eye(64)  # Assuming a 16x16 identity matrix
        identity_heatmap_path = os.path.join(os.path.dirname(args.pattern), 'identity_heatmap.png')
        plt.figure(figsize=(10, 8))
        plt.imshow(identity_matrix, cmap="coolwarm", interpolation='nearest')
        plt.colorbar()
        # set min and max values for the colorbar
        plt.clim(0, 1)
        plt.title("Identity Matrix Heatmap")
        plt.savefig(identity_heatmap_path)
        plt.close()

        coded_layer = CodedLayer(init_pattern_path=args.pattern)
        coded_layer = coded_layer.to(device)
        avg_loss, avg_corr_matrix = val_one_epoch_with_matrix(data_loader_val, device, coded_layer, patch_norm_coded=args.patch_norm_coded)
        
        # Log the average loss
        pattern_folder = os.path.dirname(args.pattern)
        with open(os.path.join(pattern_folder, 'log.txt'), 'a') as f:
            f.write(f"Pattern Test\t Avg Loss: {avg_loss}\n")
        
        # Save the correlation matrix as a heatmap
        heatmap_path = os.path.join(pattern_folder, 'heatmap.png')
        plt.figure(figsize=(10, 8))
        avg_corr_matrix = avg_corr_matrix.cpu().numpy()
        plt.imshow(avg_corr_matrix, cmap="coolwarm", interpolation='nearest')
        plt.colorbar()
        plt.clim(0, 1)
        plt.title("Average Correlation Matrix")
        plt.savefig(heatmap_path)
        plt.close()

        return


    coded_layer = CodedLayer() 

    
    optimizer = torch.optim.AdamW(coded_layer.parameters(), lr=args.lr, weight_decay=args.wd)

    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    coded_layer = coded_layer.to(device)

    # mkdir for outputdir
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    record_pattern(coded_layer, "init", args.output_dir)
    decor_L1 = val_one_epoch(data_loader_val, device, coded_layer, patch_norm_coded=args.patch_norm_coded)
    print(f"Epoch [init/{args.epochs}]\t Val Decorrelation Loss: {decor_L1}")
    with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
        f.write(f"Epoch [init/{args.epochs}]\t Val Decorrelation Loss: {decor_L1}\n")

    coded_layer.sparse_random = True
    record_pattern(coded_layer, "init_sparse", args.output_dir)
    decor_L1 = val_one_epoch(data_loader_val, device, coded_layer, patch_norm_coded=args.patch_norm_coded) 
    print(f"Epoch [init_sparse/{args.epochs}]\t Val Decorrelation Loss: {decor_L1}")
    # log to log.txt in output_dir
    with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
        f.write(f"Epoch [init_sparse/{args.epochs}]\t Val Decorrelation Loss: {decor_L1}\n")
    coded_layer.sparse_random = False

    for epoch in range(args.epochs):
        train_decor_mse = train_one_epoch(data_loader_train, optimizer, device, coded_layer, patch_norm_coded=args.patch_norm_coded)
        decor_L1 = val_one_epoch(data_loader_val, device, coded_layer, patch_norm_coded=args.patch_norm_coded)

        # print the decorrelation loss
        print(f"Epoch [{epoch+1}/{args.epochs}]\t Train Decorrelation Loss (MSE): {train_decor_mse}")
        print(f"Epoch [{epoch+1}/{args.epochs}]\t Val Decorrelation Loss (L1): {decor_L1}")
        # log to log.txt in output_dir
        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(f"Epoch [{epoch+1}/{args.epochs}]\t Train Decorrelation Loss (MSE): {train_decor_mse}\n")
            f.write(f"Epoch [{epoch+1}/{args.epochs}]\t Val Decorrelation Loss (L1): {decor_L1}\n")
        
        record_pattern(coded_layer, epoch, args.output_dir)
        
        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)

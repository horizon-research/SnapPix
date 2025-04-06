# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import torch
from einops import rearrange

from coded.constant import IMAGENET_DEFAULT_MEAN_GRAY, IMAGENET_DEFAULT_STD_GRAY

import utils
import torch
import math

def compute_mean_std(data_loader: Iterable, device: torch.device):
    total_pixels = 0
    running_mean = 0
    running_var = 0
    S = 0  # Sum of squared differences from mean
    
    for step, batch in enumerate(data_loader):
        images, _, _ = batch  # Only taking the images
        images = images.to(device, non_blocking=True)
        
        # Compute batch statistics
        batch_pixels = images.numel()  # Total number of pixels in the batch
        total_pixels += batch_pixels
        batch_mean = images.mean().item()
        batch_var = images.var().item()
        
        # Incremental mean update
        delta = batch_mean - running_mean
        running_mean += delta * batch_pixels / total_pixels

        # Incremental variance update
        S += batch_pixels * (batch_var + delta ** 2 * batch_pixels / total_pixels)
        running_var = S / (total_pixels - 1)
        
        # Print update every 10 steps
        if step % 10 == 0:
            print(f"Step [{step}/{len(data_loader)}] Mean: {running_mean:.4f} Std: {math.sqrt(running_var):.4f}")
            # also print S and total_pixels
            print(f"Step [{step}/{len(data_loader)}] S: {S:.4f} Total Pixels: {total_pixels}")

    # Final mean and std for all data
    return running_mean, math.sqrt(running_var)

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

from coded.constant import K710_SSV2_MEAN, K710_SSV2_STD

import utils

def compute_corr_matrix(images_flat, eps=1e-6):
    # Mean-center each pixel vector
    images_flat = images_flat - images_flat.mean(dim=0, keepdim=True)
    
    # Compute covariance matrix
    cov_matrix = torch.mm(images_flat.T, images_flat) / (images_flat.size(0) - 1)
    
    # Compute standard deviations with epsilon added
    std_dev = torch.sqrt(torch.diag(cov_matrix) + eps)
    std_matrix = std_dev.unsqueeze(0) * std_dev.unsqueeze(1)  # Outer product for normalization

    # Compute correlation matrix
    corr_matrix = cov_matrix / std_matrix

    return corr_matrix

def decorrelation_loss(coded_images, L1=False):
    coded_images = coded_images.view(coded_images.shape[0], 1, 1, 14, 8, 14, 8)
    coded_images = coded_images.permute(0, 1, 2, 3, 5, 4, 6)
    images_flat = coded_images.reshape(-1, 64) 
    
    # Calculate correlation matrix
    corr_matrix = compute_corr_matrix(images_flat)

    # chek nan
    if torch.isnan(corr_matrix).any():
        import ipdb; ipdb.set_trace()
    
    # Mask the diagonal to exclude it from the loss
    off_diagonal_mask = ~torch.eye(corr_matrix.size(0), dtype=bool, device=coded_images.device)
    
    # Compute decorrelation loss as the sum of off-diagonal absolute correlations
    if not L1:
        decorrelation_loss = torch.nn.functional.mse_loss(corr_matrix[off_diagonal_mask], torch.zeros_like(corr_matrix[off_diagonal_mask]))
    else:
        decorrelation_loss = torch.nn.functional.l1_loss(corr_matrix[off_diagonal_mask], torch.zeros_like(corr_matrix[off_diagonal_mask]))

    return decorrelation_loss

def decorrelation_loss_with_matrix(coded_images, L1=False):
    coded_images = coded_images.view(coded_images.shape[0], 1, 1, 14, 8, 14, 8)
    coded_images = coded_images.permute(0, 1, 2, 3, 5, 4, 6)
    images_flat = coded_images.reshape(-1, 64) 
    
    # Calculate correlation matrix
    corr_matrix = compute_corr_matrix(images_flat)

    # chek nan
    if torch.isnan(corr_matrix).any():
        import ipdb; ipdb.set_trace()
    
    # Mask the diagonal to exclude it from the loss
    off_diagonal_mask = ~torch.eye(corr_matrix.size(0), dtype=bool, device=coded_images.device)
    
    # Compute decorrelation loss as the sum of off-diagonal absolute correlations
    if not L1:
        decorrelation_loss = torch.nn.functional.mse_loss(corr_matrix[off_diagonal_mask], torch.zeros_like(corr_matrix[off_diagonal_mask]))
    else:
        decorrelation_loss = torch.nn.functional.l1_loss(corr_matrix[off_diagonal_mask], torch.zeros_like(corr_matrix[off_diagonal_mask]))

    return decorrelation_loss, corr_matrix.abs()

def patch_norm_coded_image(coded_image, long_exposed_image):
    long_exposed_image = long_exposed_image.view(long_exposed_image.shape[0], 1, 1, 14, 8, 14, 8)
    with torch.no_grad():
        mean = long_exposed_image.mean(dim=[4, 6], keepdim=True)
    coded_image = coded_image.view(coded_image.shape[0], 1, 1, 14, 8, 14, 8)
    coded_image = (coded_image - mean) # don't divide by std, the training will be instable
    coded_image = coded_image.view(coded_image.shape[0], 1, 1, 112, 112)
    return coded_image

def train_one_epoch(data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    coded_layer: torch.nn.Module,
                    patch_norm_coded=True):
    first_batch_size = 0
    batch_count = 0 
    total_loss = 0
    for step, batch in enumerate(data_loader):
        images, bool_masked_pos, decode_masked_pos = batch
        images = images.to(device, non_blocking=True)

        coded_image = coded_layer(images)
        coded_image = (coded_image - K710_SSV2_MEAN) / K710_SSV2_STD

        if patch_norm_coded:
            long_exposed_image = images.mean(dim=2, keepdim=True) 
            long_exposed_image = (long_exposed_image - K710_SSV2_MEAN) / K710_SSV2_STD
            coded_image = patch_norm_coded_image(coded_image, long_exposed_image)


        loss = decorrelation_loss(coded_image)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            import ipdb; ipdb.set_trace()
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(2)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        if first_batch_size == 0:
            first_batch_size = coded_image.size(0)
        total_loss += loss_value * coded_image.size(0) / first_batch_size
        batch_count += coded_image.size(0) / first_batch_size


        if step % 10 == 0:
            print(f"Train Step [{step}/{len(data_loader)}]\t Loss: {loss_value}")

        
    return total_loss / batch_count


def val_one_epoch(data_loader: Iterable,
                    device: torch.device,
                    coded_layer: torch.nn.Module,
                    patch_norm_coded=True):
    first_batch_size = 0
    batch_count = 0 
    total_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            images, bool_masked_pos, decode_masked_pos = batch
            images = images.to(device, non_blocking=True)
            coded_image = coded_layer(images)
            # nomalize use imagenety
            coded_image = (coded_image - K710_SSV2_MEAN) /  K710_SSV2_STD
            if patch_norm_coded:
                long_exposed_image = images.mean(dim=2, keepdim=True)
                long_exposed_image = (long_exposed_image - K710_SSV2_MEAN) / K710_SSV2_STD
                coded_image = patch_norm_coded_image(coded_image, long_exposed_image)
            loss = decorrelation_loss(coded_image, L1=True)
            loss_value = loss.item()

            torch.cuda.synchronize()

            if step % 10 == 0:
                print(f"Val Step [{step}/{len(data_loader)}]\t Loss: {loss_value}")

            if first_batch_size == 0:
                first_batch_size = coded_image.size(0)
            total_loss += loss_value * coded_image.size(0) / first_batch_size
            batch_count += coded_image.size(0) / first_batch_size


    return total_loss / batch_count




def val_one_epoch_with_matrix(data_loader: Iterable,
                              device: torch.device,
                              coded_layer: torch.nn.Module,
                              patch_norm_coded=True):
    first_batch_size = 0
    batch_count = 0 
    total_loss = 0
    avg_corr_matrix = None
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            images, bool_masked_pos, decode_masked_pos = batch
            images = images.to(device, non_blocking=True)
            coded_image = coded_layer(images)
            # normalize using imagenet stats
            coded_image = (coded_image - K710_SSV2_MEAN) / K710_SSV2_STD
            if patch_norm_coded:
                long_exposed_image = images.mean(dim=2, keepdim=True)
                long_exposed_image = (long_exposed_image - K710_SSV2_MEAN) / K710_SSV2_STD
                coded_image = patch_norm_coded_image(coded_image, long_exposed_image)
            
            # Use decorrelation_loss_with_matrix to get both loss and correlation matrix
            loss, corr_matrix = decorrelation_loss_with_matrix(coded_image, L1=True)
            loss_value = loss.item()

            # Accumulate correlation matrix
            if avg_corr_matrix is None:
                avg_corr_matrix = corr_matrix
            else:
                avg_corr_matrix += corr_matrix * coded_image.size(0) / first_batch_size

            torch.cuda.synchronize()

            if step % 10 == 0:
                print(f"Val Step [{step}/{len(data_loader)}]\t Loss: {loss_value}")

            if first_batch_size == 0:
                first_batch_size = coded_image.size(0)
            total_loss += loss_value * coded_image.size(0) / first_batch_size
            batch_count += coded_image.size(0) / first_batch_size

    # Average the correlation matrix
    avg_corr_matrix /= batch_count

    return total_loss / batch_count, avg_corr_matrix

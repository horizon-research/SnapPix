import torch
from torch.autograd import Variable
import time
import os
import sys
import numpy as np

from utils import AverageMeter, calculate_accuracy, save_gif, accuracy
from models.binarized_modules import binarizef


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, device):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # accuracies = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    input_mean = []

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        input_mean.extend([i.mean() for i in inputs.detach().cpu().numpy()])
        data_time.update(time.time() - end_time)

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        # accuracies.update(acc, inputs.size(0))
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()

        # https://github.com/itayhubara/BinaryNet.pytorch/blob/master/main_mnist.py#L113
        # for p in list(model.parameters()):
        #     if hasattr(p, 'org'):
        #         p.data.copy_(p.org)
        optimizer.step()
        # for p in list(model.parameters()):
        #     if hasattr(p, 'org'):
        #         p.org.copy_(p.data.clamp_(-1, 1))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'top1': top1.val,
            'top5': top5.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        sys.stdout.flush()
        sys.stdout.write('\rEpoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.sum:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t\t'
                         'len {len_mean},'
                         'mean {mean:.4f},'
                         'std {std:.4f},'
                         'min {min:.4f},'
                         'max {max:.4f}'
                         '\t\t'.format(
                             epoch,
                             i + 1,
                             len(data_loader),
                             batch_time=batch_time,
                             data_time=data_time,
                             loss=losses,
                             top1=top1,
                             top5=top5,
                             len_mean=len(input_mean),
                             mean=np.mean(input_mean),
                             std=np.std(input_mean),
                             min=np.min(input_mean),
                             max=np.max(input_mean),
                         ))
    sys.stdout.flush()
    print('\n[Train] Epoch{0}\t'
          'Time: {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
          'Data: {data_time.sum:.3f} ({data_time.avg:.3f})\t'
          'Loss: {loss.avg:.4f}\t'
          'Acc@1: {top1.avg:.3f}\t'
          'Acc@5: {top5.avg:.3f}'
          '\tlen {len_mean},'
          'mean {mean:.4f},'
          'std {std:.4f},'
          'min {min:.4f},'
          'max {max:.4f}'
          '\t\t'.format(
              epoch,
              batch_time=batch_time,
              data_time=data_time,
              loss=losses,
              top1=top1,
              top5=top5,
              len_mean=len(input_mean),
              mean=np.mean(input_mean),
              std=np.std(input_mean),
              min=np.min(input_mean),
              max=np.max(input_mean),
          ))
    print()

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'top1': top1.avg,
        'top5': top5.avg,
        'lr': optimizer.param_groups[0]['lr'],
        'batch_time': batch_time.sum,
        'data_time': data_time.sum,
    })
    # if hasattr(list(model.parameters())[0], 'org'):
    #     mask = binarize(
    #         list(model.parameters())[0].data,
    #         quant_mode='det'
    #     ).add_(1).div_(2).to('cpu').detach().numpy()
    if 'exp' in opt.model and not opt.load_path:
        mask = binarizef(
            list(model.parameters())[0]
        ).add_(1).div_(2).to('cpu').detach().numpy()
        print('max', mask.max())
        print('min', mask.min())
        mask = mask.reshape((opt.sample_duration, 8, 8, 1)).astype(np.uint8)
        assert mask.shape == (opt.sample_duration, 8, 8, 1)
        # save_file_path = os.path.join(opt.result_path,
        #                       'mask_{}.npy'.format(epoch))
        # np.save(save_file_path, mask)
        save_file_path = os.path.join(opt.result_path,
                                      'mask_{}.gif'.format(epoch))
        save_gif(mask, save_file_path, vmax=1, vmin=0)

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

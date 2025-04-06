import torch
from torch.autograd import Variable
import time
import sys
import numpy as np

from utils import AverageMeter, calculate_accuracy, accuracy


def val_epoch(epoch, data_loader, model, criterion, opt, logger, device):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # accuracies = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()
    input_mean = []

    end_time = time.time()
    print(len(data_loader))
    if opt.save_sample:
        res = []
        prd = []
        tgs = []
    N = 100
    for i, (inputs, targets) in enumerate(data_loader):
        if opt.save_sample:
            import os
            if i*opt.batch_size > N:
                break
            samples = inputs.to('cpu').detach().numpy()
            for j in range(opt.batch_size):
                n = i * opt.batch_size + j
                if n > N:
                    break
                if not opt.compress or opt.compress == 'reconstruct':
                    # save_file_path = os.path.join(opt.result_path, 'sample_%05d.npy' % n)
                    # np.save(save_file_path, samples[j])
                    from utils import save_gif
                    save_gif_path = os.path.join(opt.result_path, 'sample_%005d.gif' % n)
                    save_gif(samples[j].reshape(16, 112, 112, 1).astype(np.uint8), save_gif_path, vmax=255, vmin=0, interval=2000/16)
                elif opt.compress in ["one", "avg"]:
                    from PIL import Image
                    save_file_path = os.path.join(opt.result_path, 'sample_%05d.png' % n)
                    Image.fromarray(samples[j].reshape((112, 112)).astype(np.uint8)).resize((1120, 1120)).save(save_file_path)

        input_mean.extend([i.mean() for i in inputs.detach().cpu().numpy()])
        data_time.update(time.time() - end_time)

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        # accuracies.update(acc, inputs.size(0))
        # prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        prec1, prec3, prec5 = accuracy(outputs.data, targets, topk=(1, 3, 5))
        top1.update(prec1, inputs.size(0))
        top3.update(prec3, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        if opt.save_sample:
            for j in range(opt.batch_size):
                n = i * opt.batch_size + j
                out = outputs.to('cpu').detach().numpy()[j]
                tgt = targets.to('cpu').detach().numpy()[j]
                prd.append(out)
                res.append(out.argmax() == tgt)
                tgs.append(tgt)

        sys.stdout.flush()
        sys.stdout.write('\rEpoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.sum:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc@1 {top1.val:.4f} ({top1.avg:.4f})\t'
                         'Acc@3 {top3.val:.4f} ({top3.avg:.4f})\t\t'
                         'Acc@5 {top5.val:.4f} ({top5.avg:.4f})\t\t'
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
                             top3=top3,
                             top5=top5,
                             len_mean=len(input_mean),
                             mean=np.mean(input_mean),
                             std=np.std(input_mean),
                             min=np.min(input_mean),
                             max=np.max(input_mean),
                         ))
    sys.stdout.flush()
    print('\n[Val] Epoch{0}\t'
          'Time: {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
          'Data: {data_time.sum:.3f} ({data_time.avg:.3f})\t'
          'Loss: {loss.avg:.4f}\t'
          'Acc@1: {top1.avg:.4f}\t'
          'Acc@3: {top3.avg:.4f}\t'
          'Acc@5: {top5.avg:.4f}'
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
              top3=top3,
              top5=top5,
              len_mean=len(input_mean),
              mean=np.mean(input_mean),
              std=np.std(input_mean),
              min=np.min(input_mean),
              max=np.max(input_mean),
          ))
    print()
    if opt.save_sample:
        import json
        with open(opt.annotation_path, 'r') as f:
            labels = json.load(f)['labels']
        import pandas as pd
        save_file_path = os.path.join(opt.result_path, 'ans.csv')
        df = pd.DataFrame(prd, columns=labels)
        df["correct"] = res
        df["target"] = tgs
        df.to_csv(save_file_path, header=True, index=True)

    logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'top1': top1.avg,
        'top5': top5.avg,
    })

    return losses.avg

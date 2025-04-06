import os
import numpy as np # noqa
import matplotlib as mpl
mpl.use('svg')  # noqa

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import sys
# import json
import torch
from torch import nn
# from torch import optim
# from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, RGB2Gray, LowResolution)
from temporal_transforms import LoopPadding, TemporalCenterCrop
from spatio_temporal_transforms import Coded, Averaged, OneFrame, ToTemporal  # , ToRepeat
from target_transforms import ClassLabel  # , VideoID
from target_transforms import Compose as TargetCompos # noqa
from dataset import get_test_set
from utils import Logger
from utils import AverageMeter, accuracy


def test_eval(data_loader, model, criterion, opt, logger, device):
    print('eval')

    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    inst_results = []
    inst_targets = []
    if opt.save_sample:
        res = []
        prd = []
        tgs = []
    N = 200
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
                # if not opt.compress:
                #     save_file_path = os.path.join(opt.result_path, 'sample_%05d.npy' % n)
                #     np.save(save_file_path, samples[j])
                #     # from utils import save_gif
                #     # save_gif_path = os.path.join(opt.result_path, 'sample_%005d.gif' % n)
                #     # save_gif(samples[j].reshape(16, 112, 112, 1).astype(np.uint8), save_gif_path, vmax=255, vmin=0, interval=2000/16)
                # elif opt.compress in ["one", "avg"]:
                #     from PIL import Image
                #     save_file_path = os.path.join(opt.result_path, 'sample_%05d.png' % n)
                #     Image.fromarray(samples[j].reshape((112, 112)).astype(np.uint8)).resize((1120, 1120)).save(save_file_path)
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)

        inst_results.append(outputs.detach().cpu().numpy())
        inst_targets.append(targets.detach().cpu().numpy())

        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))

        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        if opt.save_sample:
            for j in range(opt.batch_size):
                n = i * opt.batch_size + j
                out = outputs.to('cpu').detach().numpy()[j]
                tgt = targets.to('cpu').detach().numpy()[j]
                prd.append(out)
                res.append(out.argmax() == tgt)
                tgs.append(tgt)

        sys.stdout.flush()
        sys.stdout.write('\rEVAL: [{:>6}/{:>6}]'.format(
            i + 1, len(data_loader)
        ))
    print()

    inst_results = np.concatenate(inst_results, axis=0)
    inst_targets = np.concatenate(inst_targets, axis=0)
    y_true, y_pred = to_instance(inst_targets, inst_results)

    inst_top1, inst_top5 = accuracy(
        torch.from_numpy(y_pred), torch.from_numpy(y_true), topk=(1, 5))

    actions = {
        'handclapping': 0,
        'boxing': 1,
        'handwaving': 2,
        'jogging': 3,
        'running': 4,
        'walking': 5,
    }
    labels = [
        'handclapping',
        'boxing',
        'handwaving',
        'jogging',
        'running',
        'walking',
    ]

    y_pred = y_pred.argmax(axis=1)
    # add
    if opt.save_sample:
        import pandas as pd
        save_file_path = os.path.join(opt.result_path, 'ans.csv')
        df = pd.DataFrame(prd, columns=labels)
        df["pred"] = np.array(prd).argmax(axis=1)
        df["correct"] = res
        df["target"] = tgs
        df.to_csv(save_file_path, header=True, index=True)
    # add end

    # score = evaluate(
    #     y_true, y_pred,
    #     labels,
    #     output_dir=opt.result_path,
    #     output_name='cm.svg')

    # logger.log({
    #     'inst_top1': inst_top1,
    #     'inst_top5': inst_top5,
    #     'clip_top1': top1.avg,
    #     'clip_top5': top5.avg,
    #     'score': score,
    # })
    # print('Loss {loss.avg:.4f}\t'
    #       'Acc(clip)@1 {top1.avg:.4f}\t'
    #       'Acc(clip)@5 {top5.avg:.4f}\t'
    #       'Acc(inst)@1 {inst_top1:.4f}\t'
    #       'Acc(inst)@5 {inst_top5:.4f}\n'
    #       'score: {score:.4f}\t'
    #       ''.format(
    #           loss=losses,
    #           top1=top1,
    #           top5=top5,
    #           inst_top1=inst_top1,
    #           inst_top5=inst_top5,
    #           score=score,
    #       ))


def to_instance(targets, results):
    tmp = []
    pi = 0
    pv = None
    for i, v in enumerate(targets):
        if pv is None:
            pv = v
        elif pv != v:
            tmp.append([pi, i-1, pv])
            pi = i
            pv = v
    else:
        tmp.append([pi, i, v])

    inst_pred = []
    inst_true = []
    for s, e, v in tmp:
        inst_pred.append(results[s:e+1].mean(axis=0))
        inst_true.append(v)
    inst_pred = np.array(inst_pred)
    inst_true = np.array(inst_true)
    return inst_true, inst_pred


def normalize(cm):
    new_cm = []
    for line in cm:
        sum_val = sum(line)
        if sum_val:
            new_array = [float(num)/float(sum_val) for num in line]
        else:
            new_array = [0 for num in line]
        new_cm.append(new_array)
    return new_cm


def evaluate(
        y_true, y_pred,
        labels,
        output_dir=None,
        output_name='cm.svg'):
    print(y_true)
    print(y_pred)
    N = len(labels)
    cm = confusion_matrix(y_true, y_pred, labels=range(N))
    cm = normalize(cm)
    print('cm:', np.array(cm).shape)
    score = sum(
        cm[i][i] for i in range(N)
    ) / len([1 for i in range(N) if sum(cm[i]) > 0])

    # plot confusion_matrix
    if output_dir:
        cm = (np.array(cm) * 100 + 0.5).astype(np.int8)
        fig = plt.figure(0, figsize=(10, 7))
        plt.clf()
        plt.matshow(cm, cmap=plt.cm.jet, fignum=False, vmin=0, vmax=100)
        ax = plt.axes()
        ax.set_xticks(range(N))
        ax.set_xticklabels(labels)
        ax.xaxis.set_ticks_position("bottom")
        ax.set_yticks(range(N))
        ax.set_yticklabels(labels)
        # ax2 = ax.twinx()
        # ax2.set_xticks(range(N))
        # ax2.set_xticklabels(labels)
        # ax2.xaxis.set_ticks_position("bottom")
        # ax2.set_yticks(range(N))
        # ax2.set_yticklabels(labels)
        if N > 50:
            plt.tick_params(axis='both', which='major', labelsize=3)
            plt.tick_params(axis='both', which='minor', labelsize=2)
        fig.autofmt_xdate()
        plt.title('')
        plt.colorbar()
        plt.xlabel('Predict class')
        plt.ylabel('True class')
        plt.grid(False)
        if N <= 50:
            width, height = np.array(cm).shape
            for x in range(width):
                for y in range(height):
                    ax.annotate(
                        str(cm[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
        plt.savefig(os.path.join(output_dir, output_name), dpi=1000)
    else:
        plt.show()
    return score


if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    # with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
    #     json.dump(vars(opt), opt_file, indent=2)
    if opt.dataset != 'kth2':
        raise
    if not opt.resume_path:
        raise

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    from torch.backends import cudnn
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu" if opt.no_cuda else "cuda")
    if not opt.no_cuda:
        criterion = criterion.to(device)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if opt.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            opt.scales, opt.sample_size, crop_positions=['c'])

    # if not opt.no_train:
    #     assert opt.train_crop in ['random', 'corner', 'center']
    #     if opt.train_crop == 'random':
    #         crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    #     elif opt.train_crop == 'corner':
    #         crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    #     elif opt.train_crop == 'center':
    #         crop_method = MultiScaleCornerCrop(
    #             opt.scales, opt.sample_size, crop_positions=['c'])
    #     if opt.dataset == 'gtea':
    #         spatial_transform = Compose([
    #             crop_method,
    #             RandomHorizontalFlip(),
    #             ToTensor(opt.norm_value), norm_method,
    #         ])
    #     else:
    #         spatial_transform = Compose([
    #             crop_method,
    #             RandomHorizontalFlip(),
    #             RGB2Gray(),
    #             ToTensor(opt.norm_value), norm_method,
    #         ])
    #     temporal_transform = TemporalRandomCrop(opt.sample_duration)
    #     if opt.compress == 'mask':
    #         spatio_temporal_transform = Coded(opt.mask_path)
    #     elif opt.compress == 'avg':
    #         spatio_temporal_transform = Averaged()
    #     elif opt.compress == 'one':
    #         spatio_temporal_transform = OneFrame()
    #     elif opt.compress == 'spatial':
    #         spatio_temporal_transform = Coded(opt.mask_path)
    #     elif opt.compress == 'avg':
    #         spatio_temporal_transform = Averaged()
    #     elif opt.compress == 'one':
    #         spatio_temporal_transform = OneFrame()
    #     elif opt.compress == 'spatial':
    #         if opt.dataset == 'gtea':
    #             spatial_transform = Compose([
    #                 crop_method,
    #                 RandomHorizontalFlip(),
    #                 LowResolution(opt.spatial_compress_size, use_cv2=opt.use_cv2),
    #                 ToTensor(opt.norm_value), norm_method,
    #             ])
    #         else:
    #             spatial_transform = Compose([
    #                 crop_method,
    #                 RandomHorizontalFlip(),
    #                 RGB2Gray(),
    #                 LowResolution(opt.spatial_compress_size, use_cv2=opt.use_cv2),
    #                 ToTensor(opt.norm_value), norm_method,
    #             ])
    #         spatio_temporal_transform = None
    #     elif opt.compress == 'temporal':
    #         spatio_temporal_transform = Compose([
    #             Coded(opt.mask_path),
    #             ToTemporal(opt.mask_path),
    #         ])
    #     elif opt.compress == 'mask_3d':
    #         spatio_temporal_transform = Compose([
    #             ToRepeat(Coded(opt.mask_path), opt.sample_duration),
    #         ])
    #     elif opt.compress == 'avg_3d':
    #         spatio_temporal_transform = Compose([
    #             ToRepeat(Averaged(), opt.sample_duration),
    #         ])
    #     elif opt.compress == 'one_3d':
    #         spatio_temporal_transform = Compose([
    #             ToRepeat(OneFrame(), opt.sample_duration),
    #         ])
    #     else:
    #         spatio_temporal_transform = None
    #     target_transform = ClassLabel()
    #     training_data = get_training_set(opt, spatial_transform,
    #                                      temporal_transform, target_transform,
    #                                      spatio_temporal_transform)
    #     train_loader = torch.utils.data.DataLoader(
    #         training_data,
    #         batch_size=opt.batch_size,
    #         shuffle=True,
    #         num_workers=opt.n_threads,
    #         pin_memory=True)
    #     train_logger = Logger(
    #         os.path.join(opt.result_path, 'train.log'),
    #         ['epoch', 'loss', 'top1', 'top5', 'lr', 'batch_time', 'data_time'])
    #     train_batch_logger = Logger(
    #         os.path.join(opt.result_path, 'train_batch.log'),
    #         ['epoch', 'batch', 'iter', 'loss', 'top1', 'top5', 'lr'])

    #     if opt.nesterov:
    #         dampening = 0
    #     else:
    #         dampening = opt.dampening
    #     if opt.optimizer == 'sgd':
    #         optimizer = optim.SGD(
    #             parameters,
    #             lr=opt.learning_rate,
    #             momentum=opt.momentum,
    #             dampening=dampening,
    #             weight_decay=opt.weight_decay,
    #             nesterov=opt.nesterov)
    #     elif opt.optimizer == 'adam':
    #         optimizer = optim.Adam(
    #             parameters,
    #             lr=opt.learning_rate)
    #     scheduler = lr_scheduler.ReduceLROnPlateau(
    #         optimizer, 'min', patience=opt.lr_patience)
    if not opt.no_val:
        if opt.dataset == 'gtea':
            spatial_transform = Compose([
                Scale(opt.sample_size),
                CenterCrop(opt.sample_size),
                ToTensor(opt.norm_value), norm_method,
            ])
        else:
            spatial_transform = Compose([
                Scale(opt.sample_size),
                CenterCrop(opt.sample_size),
                RGB2Gray(),
                ToTensor(opt.norm_value), norm_method,
            ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        if opt.compress == 'mask':
            spatio_temporal_transform = Coded(opt.mask_path)
        elif opt.compress == 'avg':
            spatio_temporal_transform = Averaged()
        elif opt.compress == 'one':
            spatio_temporal_transform = OneFrame()
        elif opt.compress == 'spatial':
            spatial_transform = Compose([
                crop_method,
                RandomHorizontalFlip(),
                RGB2Gray(),
                LowResolution(opt.spatial_compress_size, use_cv2=opt.use_cv2),
                ToTensor(opt.norm_value), norm_method,
            ])
            spatio_temporal_transform = None
            temporal_transform = TemporalCenterCrop(opt.sample_duration)
        elif opt.compress == 'temporal':
            spatio_temporal_transform = Compose([
                Coded(opt.mask_path),
                ToTemporal(opt.mask_path),
            ])
        else:
            spatio_temporal_transform = None
            temporal_transform = TemporalCenterCrop(opt.sample_duration)
        test_data = get_test_set(
            opt, spatial_transform, temporal_transform, target_transform,
            spatio_temporal_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test_logger = Logger(
            os.path.join(opt.result_path, 'test.log'),
            ['inst_top1', 'inst_top5', 'clip_top1', 'clip_top5', 'score'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # if not opt.no_train:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        raise

    # if opt.load_path:
    #     check = False
    #     print('loading checkpoint {}'.format(opt.load_path))
    #     checkpoint = torch.load(opt.load_path)
    #     assert opt.arch == checkpoint['arch']
    #     model.load_state_dict(checkpoint['state_dict'])

    #     for name, param in model.named_parameters():
    #         if name == 'module.exp.weight':
    #             param.requires_grad = False
    #             print('{}({}) is fixed'.format(name, param.shape))
    #             check = True
    #     if not check:
    #         raise

    # if opt.init_path:
    #     check = False
    #     print('initilize checkpoint {}'.format(opt.init_path))
    #     checkpoint = torch.load(opt.init_path)
    #     assert opt.arch == checkpoint['arch']

    #     for (name, param), i in zip(
    #             model.named_parameters(),
    #             range(opt.init_level)):
    #         for (prev_name, prev_param) in checkpoint['state_dict'].items():
    #             if name == prev_name:
    #                 param.data = prev_param.data
    #                 print('{}: {}({}) -> {}({})'.format(
    #                     i, prev_name, prev_param.shape, name, param.shape))
    #                 check = True
    #     if not check:
    #         raise

    model.to(device)
    test_eval(test_loader, model, criterion, opt, test_logger, device)

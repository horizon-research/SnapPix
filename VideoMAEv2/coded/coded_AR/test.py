import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import json
import numpy as np

from utils import AverageMeter


vids = {}
N = 10


def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = output_buffer
    average_scores = np.array(video_outputs).mean(axis=0)
    sorted_scores, locs = [], []
    for s in sorted(average_scores, reverse=True)[:N]:
        sorted_scores.append(s)
        locs.append((average_scores == s).argmax())


    rvids = {v: k for k,v in vids.items()}
    video_results = []
    for i in range(N):
        video_results.append({
            'label': class_names[int(locs[i])],
            'score': float(sorted_scores[i])
        })

    test_results['results'][rvids[int(video_id)]] = video_results


def test(data_loader, model, opt, class_names, device):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = -1
    test_results = {'results': {}}
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        for target in targets:
            if target not in vids:
                vids[target] = len(vids.keys())
        targets = [vids[t] for t in targets]

        inputs = inputs.to(device)
        targets = torch.Tensor(targets)
        targets = targets.to(device)
        outputs = model(inputs)
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs)

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and targets[j].detach().cpu().numpy() != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id,
                                        test_results, class_names)
                output_buffer = []
            output_buffer.append(outputs[j].data.detach().cpu().numpy())
            previous_video_id = targets[j].detach().cpu().numpy()

        if (i % 100) == 0:
            with open(
                    os.path.join(opt.result_path, '{}.json'.format(
                        opt.test_subset)), 'w') as f:
                json.dump(test_results, f)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        sys.stdout.flush()
        sys.stdout.write('\r[{}/{}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t\t\t'.format(
                             i + 1,
                             len(data_loader),
                             batch_time=batch_time,
                             data_time=data_time))
    with open(
            os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),
            'w') as f:
        json.dump(test_results, f, indent=2)

import csv


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k / batch_size)
    return res


import os
from itertools import combinations, chain, product

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation


def save_gif(frames, file_path, vmax=255, vmin=0, interval=3000/25):
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(
        left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ims = []
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    for frame in frames:
        m = plt.imshow(
            (frame).reshape(*frame.shape[:-1]).astype(np.uint8),
            cmap=plt.cm.gray, vmax=vmax, vmin=vmin)
        plt.axis('off')
        ims.append([m])
    ani = animation.ArtistAnimation(fig, ims, interval=interval, repeat=False)
    ani.save(file_path, writer="imagemagick")
    plt.close()


def gen_hama_photo(h, w):
    mask = np.zeros((8, 8, 1))
    if w.size:
        mask[:, w] = 1
    if h.size:
        mask[h, :] = 1
    return mask


def gen_hama_photo_patterns():
    index = list(chain(*[list(combinations(range(8), i)) for i in range(9)]))
    g = (gen_hama_photo(np.array(h), np.array(w)) for (h, w) in product(index, repeat=2))
    masks = np.array(list(g))
    # assert masks.shape[0] == 2**16
    mask_hama_photo = np.unique(masks, axis=0)
    return mask_hama_photo[1:-1]


def fit_hama_photo(raw, mask=gen_hama_photo_patterns()):
    res = np.array([
        mask[np.argmin(np.mean(np.square(mask - raw[i]), axis=(1, 2, 3)))]
        for i in range(16)])
    return res


def fit_hitomi(raw):
    agmx = raw.argmax(axis=0)
    res = np.array([np.ones((8, 8, 1)) * i == agmx for i in range(16)]).astype(np.uint8)
    return res


def fit_rand(raw, th=0):
    res = (raw>=th).astype(np.uint8)
    return res

# def fit_hama_photo_tensor(raw, mask=gen_hama_photo_patterns()):
#     from keras import backend as K
#     import tensorflow as tf
#     mask_tensor = tf.reshape(
#         tf.convert_to_tensor(mask, dtype=tf.float32), (-1, 8, 8, 1))
#     n = 16
#     i1 = tf.constant(1)
#     res0 = tf.reshape(mask_tensor[K.argmin(K.mean(K.square(
#         mask_tensor - raw[0]), axis=(1, 2, 3)))],
#         (1, 8, 8, 1))
#     c = lambda i, res: i < n
#     b = lambda i, res: (
#         i+1,
#         tf.concat([res, tf.reshape(mask_tensor[K.argmin(K.mean(K.square(
#             mask_tensor - raw[i]), axis=(1, 2, 3)))],
#             (1, 8, 8, 1))], axis=0))
#     _, res = tf.while_loop(
#         c, b, loop_vars=[i1, res0],
#         shape_invariants=[i1.get_shape(), tf.TensorShape((None, 8, 8, 1))])
#     return res
# 
# 
# def fit_hitomi_tensor(raw):
#     from keras import backend as K
#     import tensorflow as tf
#     i1 = tf.constant(1, dtype=tf.int64)
#     agmx = K.argmax(raw, axis=0)
#     res0 = tf.reshape(
#         K.cast(K.equal(agmx, 0), dtype=K.floatx()), (1, 8, 8, 1))
#     c = lambda i, res: i < 16
#     b = lambda i, res: (
#         i+1,
#         tf.concat([
#             res,
#             tf.reshape(
#                 K.cast(K.equal(agmx, i), dtype=K.floatx()), (1, 8, 8, 1))
#         ], axis=0)
#     )
#     _, res = tf.while_loop(
#         c, b, loop_vars=[i1, res0],
#         shape_invariants=[i1.get_shape(), tf.TensorShape((None, 8, 8, 1))])
#     return res
# 
# 
# def fit_rand_tensor(raw, th=0):
#     from keras import backend as K
#     res = K.cast(K.greater_equal(raw, th), K.floatx())
#     return res

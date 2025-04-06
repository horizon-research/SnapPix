import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
from multiprocessing import Pool
from itertools import chain
from functools import partial
from glob import glob
import numpy as np


def load_label(annotation_path):
    label_path = os.path.join(annotation_path)
    with open(label_path, "r") as f:
        labels = json.load(f)
    class_to_idx = {}
    for name, idx in labels.items():
        class_to_idx[name.replace(",", "").replace("'", "").replace(" ", "_").lower()] = idx

    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name
    return idx_to_class, class_to_idx

class REAL(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 spatio_temporal_transform=None,
                 target_transform=None,
                 sample_duration=16):
        self.root_path = root_path
        self.is_mask = os.path.basename(root_path) in ["mask", "mask2", "mask3", "mask3_selected", "mask_old"]
        self.is_reconstruct = os.path.basename(root_path) in ["mask3_reconstruct"]
        self.class_names, self.class_to_idx = load_label(annotation_path)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.spatio_temporal_transform = spatio_temporal_transform
        self.target_transform = target_transform
        self.g = self.gen_generator()

    def pre_process(self, X, y):
        clip = [Image.fromarray(arr) for arr in X]
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        if self.spatio_temporal_transform is not None:
            clip = self.spatio_temporal_transform(clip)
        else:
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = int(y)
        return clip, target

    def gen_generator(self):
        if self.is_mask:
            for action_dir in sorted(glob(os.path.join(self.root_path, "*"))):
                action = os.path.basename(action_dir)
                for clip_dir in sorted(glob(os.path.join(action_dir, "clip[0-9]"))):
                    for npy_file in sorted(glob(os.path.join(clip_dir, "npy/*.npy"))):
                        X = np.load(npy_file).reshape(1, 112, 112)
                        y = int(self.class_to_idx[action])
                        yield torch.from_numpy(X).float(), y
            # l = sorted(glob(os.path.join(self.root_path, "*/clip[0-9]/npy/*.npy")), key=lambda x: x.split('/'))
            # arr = np.load(os.path.join(self.root_path, "all.npy"))
            # # arr = np.load(os.path.join(self.root_path, "all_sub_black.npy"))
            # for path, a in zip(l, arr):
            #     action = path.split('/')[-4]
            #     X = a.reshape(1, 112, 112)
            #     y = int(self.class_to_idx[action])
            #     # sth: len 15488,mean 38.7060,std 13.4872,min 0.4102,max 84.3218
            #     # real: len 756,mean 91.7425,std 10.4236,min 70.6958,max 123.1428
            #     # sth_mean = 38.7060
            #     # sth_std = 13.4872
            #     # real_mean = 91.7425
            #     # real_std = 10.4236
            #     # yield torch.from_numpy(X).float().sub(real_mean).div(real_std).mul(sth_std).add(sth_mean), y
            #     yield torch.from_numpy(X).float(), y

            # for action_dir in sorted(glob(os.path.join(self.root_path, "*"))):
            #     action = os.path.basename(action_dir)
            #     for npy_file in sorted(glob(os.path.join(action_dir, "npy/*.npy"))):
            #         X = np.load(npy_file).reshape(1, 112, 112)
            #         y = int(self.class_to_idx[action])
            #         yield torch.from_numpy(X).div(255).div(1).float(), y
        elif self.is_reconstruct:
            for action_dir in sorted(glob(os.path.join(self.root_path, "*"))):
                action = os.path.basename(action_dir)
                for clip in sorted(glob(os.path.join(action_dir, "clip[0-9]/video/*.npy"))):
                    X = np.load(clip) * 255 / 2
                    y = self.class_to_idx[action]
                    yield torch.from_numpy(X.reshape(1, 16, 112, 112)).float(), int(y)
        else:
            for action_dir in sorted(glob(os.path.join(self.root_path, "*"))):
                action = os.path.basename(action_dir)
                for clip in sorted(glob(os.path.join(action_dir, "clip[0-9].npy"))):
                    X = np.load(clip)
                    y = self.class_to_idx[action]
                    yield self.pre_process(X, y)

    def __getitem__(self, index):
        return next(self.g)

    def __len__(self):
        if self.is_mask:
            # l = sorted(glob(os.path.join(self.root_path, "*/npy/*.npy")), key=lambda x: x.split('/'))
            l = sorted(glob(os.path.join(self.root_path, "*/clip[0-9]/npy/*.npy")), key=lambda x: x.split('/'))
            return len(l)
            # return 100
        else:
            return 100

# Based on S. Kumawat, T. Okawara, M. Yoshida, H. Nagahara and Y. Yagi, "Action Recognition From a Single Coded Image," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 4, pp. 4109-4121, 1 April 2023, doi: 10.1109/TPAMI.2022.3196350. keywords: {Image recognition;Videos;Image reconstruction;Cameras;Sensors;Task analysis;Computational modeling;Action recognition;coded exposure image;computational photography;knowledge distillation},
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import os
import torch.nn.functional as F


class Normal_BinarizeF(torch.autograd.Function): 

    @staticmethod
    def forward(ctx, _input):
        output = _input.new(_input.size())
        output[_input > 0] = 1
        output[_input <= 0] = 0
        # set the max idx along dim 1 to 1
        output.scatter_(1, _input.argmax(dim=1, keepdim=True), 1)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class CodedLayer(nn.Module):
    def __init__(self, t=16, c=1, s=8, init_pattern_path=None, init_v=0.03, opt_pattern=True):
        super().__init__()
        self.t = t
        self.s = s
        self.c = c
        self.binarizef = Normal_BinarizeF.apply
        self.coded_weight = Parameter(torch.zeros(self.c, self.t, self.s, self.s))
        self.coded_weight.requires_grad = True
        if init_pattern_path is not None:
            print("loading pattern_path", init_pattern_path)
            pattern = torch.load(init_pattern_path)
            # if not tensor, then it's a numpy array, make it tensor
            if not isinstance(pattern, torch.Tensor):
                pattern = torch.tensor(pattern)
            self.coded_weight.data = (pattern * 2 - 1) * init_v
        else:
            self.coded_weight.data.uniform_(-init_v, init_v)   
            # self.coded_weight.data = torch.ones(self.c, self.t, self.s, self.s) * init_v 
        self.sparse_random = False

        # generate a sparse pattern 1x16x8x8 but only 1 slot in 16 is 1
        self.sparse_random_pattern = torch.zeros(1, self.t, self.s, self.s)
        for i in range(self.s):
            for j in range(self.s):
                random_idx = torch.randint(0, self.t, (1,))
                self.sparse_random_pattern[0, random_idx, i, j] = 1
        self.opt_pattern = opt_pattern

    def forward(self, x):
        if self.sparse_random:
            binary_weight = self.sparse_random_pattern.repeat(1, 1, 14, 14).detach()
            # to device
            binary_weight = binary_weight.to(x.device)
        else:
            if self.opt_pattern:
                binary_weight = self.binarizef(self.coded_weight).repeat(1, 1, 14, 14)
            else:
                with torch.no_grad():
                    binary_weight = self.binarizef(self.coded_weight).repeat(1, 1, 14, 14)

        out = x * binary_weight.unsqueeze(0) # B, C, T, H, W
        out = out.sum(dim=2, keepdim=True)

        exposed_nums = binary_weight.unsqueeze(0).sum(dim=2, keepdim=True)
        exposed_nums[exposed_nums == 0] = 1 # there shouldn't be 0, but just in case
        norm_factor = 1 / exposed_nums

        out = out * norm_factor

        return out
    
    def get_pattern(self):
        if self.sparse_random:
            return self.sparse_random_pattern.detach().cpu().numpy()
        else:
            return self.binarizef(self.coded_weight).detach().cpu().numpy()
    


class CodedBaseline(nn.Module):
    def __init__(self, long_exposure=True):
        super().__init__()
        self.long_exposure = long_exposure
    def forward(self, x):
        if self.long_exposure:
            return x.mean(dim=2, keepdim=True)
        else:
            # exposose only the 7th frame
            return x[:, :, 7, :, :].unsqueeze(2)
    def get_pattern(self):
        if self.long_exposure:
            return  torch.ones(1, 16, 8, 8).cpu().numpy()
        else:
            zeros = torch.zeros(1, 16, 8, 8).cpu().numpy()
            zeros[0, 7,  :, :] = 1
            return zeros

    

        
class CodedAblatLayer(nn.Module):
    def __init__(self, t=16, c=1, s=8, init_pattern_path=None, init_v=0.03, global_pattern=False, pixel_wise_norm=False):
        super().__init__()
        self.t = t
        self.s = s
        self.c = c
        self.binarizef = Normal_BinarizeF.apply

        if not global_pattern:
            self.coded_weight = Parameter(torch.zeros(self.c, self.t, self.s, self.s))
        else:
            self.coded_weight = Parameter(torch.zeros(self.c, self.t, 112, 112))
        self.coded_weight.requires_grad = True


        if init_pattern_path is not None:
            assert not global_pattern, "global pattern not supported"
            print("loading pattern_path", init_pattern_path)
            pattern = torch.load(init_pattern_path)
            # if not tensor, then it's a numpy array, make it tensor
            if not isinstance(pattern, torch.Tensor):
                pattern = torch.tensor(pattern)
            self.coded_weight.data = (pattern * 2 - 1) * init_v
        else:
            self.coded_weight.data.uniform_(-init_v, init_v)   
        
        self.global_pattern = global_pattern
        self.pixel_wise_norm = pixel_wise_norm

    def forward(self, x):
        with torch.no_grad():
            if not self.global_pattern:
                binary_weight = self.binarizef(self.coded_weight).repeat(1, 1, 14, 14)
            else:
                binary_weight = self.binarizef(self.coded_weight).repeat(1, 1, 1, 1)

        out = x * binary_weight.unsqueeze(0) # B, C, T, H, W
        out = out.sum(dim=2, keepdim=True)

        exposed_nums = binary_weight.unsqueeze(0).sum(dim=2, keepdim=True)
        exposed_nums[exposed_nums == 0] = 1 # there shouldn't be 0, but just in case

        if self.pixel_wise_norm:
            norm_factor = 1. / exposed_nums
        else:
            norm_factor = 1. / 16.

        out = out * norm_factor

        return out
    
    def get_pattern(self):
        if self.sparse_random:
            return self.sparse_random_pattern.detach().cpu().numpy()
        else:
            return self.binarizef(self.coded_weight).detach().cpu().numpy()
    

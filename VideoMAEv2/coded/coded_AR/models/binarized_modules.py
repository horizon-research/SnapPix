import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        input = input.sign()
        return input

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


def binarize(tensor, quant_mode='det'):
    if quant_mode == 'det':
        return tensor.sign()
    elif quant_mode == 'hard_sigmoid':
        return tensor.add_(1).div_(2).clamp_(0, 1).round().mul_(2).add_(-1)
    else:
        return tensor.add_(1).div_(2).add_(
            torch.rand(tensor.size()).to('cuda').add(-0.5)
        ).clamp_(0, 1).round().mul_(2).add_(-1)


class Exposure(nn.Module):

    def __init__(self, t=16, c=1, s=8, binarize_type='full', kwargs={}):
        super().__init__()
        self.t = t
        self.s = s
        self.c = c
        self.noise_count = 100
        self.kwargs = kwargs
        self.binarize_type = binarize_type
        self.weight = Parameter(torch.Tensor(c, t, s, s))
        self.reset_parameters()
        self.act = BinActive()

    def reset_parameters(self):
        # nn.init.normal_(self.weight, 0.0, 1.0)
        nn.init.uniform_(self.weight, -0.5, 0.5)

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        # if self.noise_count > 0:
        #     p = max(self.noise_count, 0) / 100
        #     self.weight.org = self.weight.org.mul_(1-p).add_(
        #         torch.rand(self.weight.org.size()).to('cuda').add(-0.5).mul(p)
        #     )
        #     self.noise_count -= 1
        if self.binarize_type == 'full':
            self.weight.data = self.weight.org
        # out = input * binarize(
        #     self.weight, quant_mode='hard_sigmoid', **self.kwargs
        # ).add_(1).div_(2).repeat(1, 1, 14, 14)
        out = input * self.act(self.weight).add_(1).div_(2).repeat(1, 1, 14, 14)
        return out.mean(dim=2)

    def extra_repr(self):
        return 'binarize_typ={}, t={}, s={}'.format(
            self.binarize_type, self.t, self.s
        )


class BinarizeF(torch.autograd.Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input


binarizef = BinarizeF.apply


class Exposuref(nn.Module):

    def __init__(self, t=16, c=1, s=8, binarize_type='full', kwargs={}):
        super().__init__()
        self.t = t
        self.s = s
        self.c = c
        self.noise_count = 100
        self.kwargs = kwargs
        self.binarize_type = binarize_type
        self.weight = Parameter(torch.Tensor(c, t, s, s))
        self.reset_parameters()

    def reset_parameters(self):
        self.stdv = math.sqrt(1.5 / (self.s * self.s * self.t * 14 * 14))
        self.weight.data.uniform_(-self.stdv, self.stdv)
        self.weight.lr_scale = 1. / self.stdv

    def forward(self, input):
        # import ipdb; ipdb.set_trace()
        if self.training and self.noise_count > 0:
            p = max(self.noise_count, 0) / 100
            self.noise_count -= 1
            binary_weight = binarizef(
                self.weight * (1-p) +
                torch.rand(self.weight.data.size()).uniform_(-self.stdv, self.stdv).to('cuda').mul(p)
            ).add_(1).div_(2).repeat(1, 1, 14, 14)
        else:
            binary_weight = binarizef(self.weight).add_(1).div_(2).repeat(1, 1, 14, 14)
        out = input * binary_weight

        return out.mean(dim=2)
        # return out.mean(dim=2) / 255


    def extra_repr(self):
        return 'binarize_typ={}, t={}, s={}'.format(
            self.binarize_type, self.t, self.s
        )


class Exposuref3d(Exposuref):

    def __init__(self, t=16, c=1, s=8, binarize_type='full', kwargs={}):
        super().__init__(t, c, s, binarize_type, kwargs)

    def forward(self, input):
        return super().forward(input).view(-1, 1, 1, 112, 112).repeat(1, 1, 16, 1, 1)

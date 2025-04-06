import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


# class PatternConv(torch.nn.Module):
#     def __init__(
#             self, in_channels, out_channels, kernel_size,
#             bias=True, pattern_size=(8, 8)):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.bias = bias
#         self.pattern_size = pattern_size
#         hs, ws = self.pattern_size
#         self.convs = [[
#             nn.Conv2d(
#                 self.in_channels, self.out_channels,
#                 kernel_size=self.kernel_size,
#                 stride=(hs, ws), bias=self.bias
#             ).to("cuda")
#             for _ in range(ws)
#         ] for _ in range(hs)]
#         self.reset_parameters()
#
#     def forward(self, x):
#         hs, ws = self.pattern_size
#         paded = F.pad(x, (0, hs-1, 0, ws-1))
#         B, C, H, W = x.shape
#         out = torch.zeros(self.in_channels*B, self.out_channels*C, H, W).float().to('cuda')
#         for h in range(hs):
#             for w in range(ws):
#                 out[:, :, h::hs, w::ws] = self.convs[h][w](paded[:, :, h:H+h, w:W+w])
#         return out
#
#     def reset_parameters(self):
#         hs, ws = self.pattern_size
#         for h in range(hs):
#             for w in range(ws):
#                 init.kaiming_uniform_(self.convs[h][w].weight, a=math.sqrt(2))
#                 if self.convs[h][w].bias is not None:
#                     fan_in, _ = init._calculate_fan_in_and_fan_out(self.convs[h][w].weight)
#                     bound = 1 / math.sqrt(fan_in)
#                     init.uniform_(self.convs[h][w].bias, -bound, bound)
#
#     def extra_repr(self):
#         return 'pattern_size=%s, %s' % (repr(tuple(self.pattern_size)), repr(self.convs[0][0]))


class PatternConv(torch.nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size,
            bias=True, pattern_size=(8, 8), stride=1, padding=0, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.pattern_size = pattern_size
        self.stride = stride
        self.padding = padding

        hs, ws = self.pattern_size
        self.convs = [[
            nn.Conv2d(
                self.in_channels, self.out_channels,
                kernel_size=self.kernel_size,
                stride=(hs, ws),
                padding=self.padding,
                bias=self.bias, groups=groups
            ).to("cuda")
            for _ in range(ws//self.stride)
        ] for _ in range(hs//self.stride)]

        for h in range(hs//self.stride):
            for w in range(ws//self.stride):
                setattr(self, 'weight%s%s' % (h, w), self.convs[h][w])

        self.reset_parameters()

    def forward(self, x):
        hs, ws = self.pattern_size
        paded = F.pad(x, (0, hs-2+self.padding, 0, ws-2+self.padding))

        B, _, H, W = x.shape
        out = torch.zeros(B, self.out_channels, H//self.stride, W//self.stride).float().to('cuda')
        for h in range(hs//self.stride):
            for w in range(ws//self.stride):
                out[:, :, self.stride*h::hs, self.stride*w::ws] = self.convs[h][w](
                    paded[
                        :, :,
                        self.stride*h:H//self.stride+self.stride*h,
                        self.stride*w:W//self.stride+self.stride*w])
        return out

    def reset_parameters(self):
        hs, ws = self.pattern_size
        for h in range(hs//self.stride):
            for w in range(ws//self.stride):
                init.kaiming_uniform_(self.convs[h][w].weight, a=math.sqrt(2))
                if self.convs[h][w].bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(self.convs[h][w].weight)
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(self.convs[h][w].bias, -bound, bound)

    def extra_repr(self):
        return 'pattern_size=%s, %s' % (repr(tuple(self.pattern_size)), repr(self.convs[0][0]))


class PatternConvInit(PatternConv):
    def reset_parameters(self):
        H = 1e-4
        hs, ws = self.pattern_size
        for h in range(hs):
            for w in range(ws):
                init.uniform_(self.convs[h][w].weight, -H, H)
                self.convs[h][w].weight.data[:, :, 1, 1] += 1
                if self.convs[h][w].bias is not None:
                    self.convs[h][w].bias.data.zero_()


class PatternConv3d(torch.nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size,
            bias=True, pattern_size=(8, 8), groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.pattern_size = pattern_size
        hs, ws = self.pattern_size
        self.convs = [[
            nn.Conv3d(
                self.in_channels, self.out_channels,
                kernel_size=self.kernel_size,
                stride=(1, hs, ws), bias=self.bias, groups=groups
            ).to("cuda")
            for _ in range(ws)
        ] for _ in range(hs)]
        self.reset_parameters()

    def forward(self, x):
        hs, ws = self.pattern_size
        paded = F.pad(x, (hs-1, hs-1, ws-1, ws-1, 1, 1))
        B, C, T, H, W = x.shape
        out = torch.zeros(B, self.out_channels, T, H, W).float().to('cuda')
        for h in range(hs):
            for w in range(ws):
                out[:, :, :, h::hs, w::ws] = self.convs[h][w](paded[:, :, :, h:H+h, w:W+w])
        return out

    def reset_parameters(self):
        hs, ws = self.pattern_size
        for h in range(hs):
            for w in range(ws):
                init.kaiming_uniform_(self.convs[h][w].weight, a=math.sqrt(2))
                if self.convs[h][w].bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(self.convs[h][w].weight)
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(self.convs[h][w].bias, -bound, bound)

    def extra_repr(self):
        return 'pattern_size=%s, %s' % (repr(tuple(self.pattern_size)), repr(self.convs[0][0]))

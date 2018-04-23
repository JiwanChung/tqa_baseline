from __future__ import division

from basicBlock import BasicBlock
from downsampleH import DownSampleH

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ForgetGate(nn.Module):
    def __init__(self, config):
        super(ForgetGate, self).__init__()

        planes = 16

        self.res_conv_h = nn.Sequential(
            BasicBlock(planes, planes, stride=1, downsample=None),
            BasicBlock(planes, planes, stride=1, downsample=None),
            BasicBlock(planes, planes, stride=1, downsample=None),
            BasicBlock(planes, planes, stride=1, downsample=None),
            BasicBlock(planes, planes, stride=1, downsample=None),
            nn.Conv2d(planes, 1, 3, stride=1, padding=1))

        self.forget = Variable(torch.ones(1).cuda()) if config.cuda else Variable(torch.ones(1))
        self.batch_size = config.batch_size
        self.dim_words = config.dim_words

        self.downsample = DownSampleH(config)

    def forward(self, h, x):
        h_new_input = self.res_conv_h(x).squeeze()
        importance = self.downsample(h_new_input)

        forget = F.sigmoid(torch.mul(self.forget, importance)).unsqueeze(self.dim_words)
        h_new = h * (1 - forget) + F.softmax(h_new_input, dim=self.dim_words) * forget  # forget var should be learnable!

        return h_new

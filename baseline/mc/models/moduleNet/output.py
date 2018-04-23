from basicBlock import BasicBlock

import torch.nn as nn

class Output(nn.Module):
    def __init__(self, config):
        super(Output, self).__init__()

        planes = 16

        self.res_conv_o = nn.Sequential(
                BasicBlock(1, planes, stride=1, downsample=None),
                BasicBlock(planes, planes, stride=1, downsample=None),
                BasicBlock(planes, planes, stride=1, downsample=None),
                BasicBlock(planes, planes, stride=1, downsample=None),
                BasicBlock(planes, planes, stride=1, downsample=None),
                nn.Conv2d(planes, 1, 3, stride=1, padding=1))

    def forward(self, h):
        h = h.unsqueeze(1)
        o = self.res_conv_o(h)
        o = o.squeeze()

        return o

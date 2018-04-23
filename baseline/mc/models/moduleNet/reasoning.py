from basicBlock import BasicBlock

import torch
import torch.nn as nn
import torch.nn.functional as F

class Reasoning(nn.Module):
    def __init__(self, config):
        super(Reasoning, self).__init__()

        self.dim_words = config.dim_words
        self.keys = config.keys
        self.sizes = config.sizes
        planes = config.reasoning_planes

        self.res_conv_a = nn.Sequential(
                        BasicBlock(1, planes, stride=1, downsample=None),
                        BasicBlock(planes, planes, stride=1, downsample=None),
                        BasicBlock(planes, planes, stride=1, downsample=None))

        self.res_conv_x = nn.Sequential(
                        BasicBlock(planes, planes, stride=1, downsample=None),
                        BasicBlock(planes, planes, stride=1, downsample=None),
                        BasicBlock(planes, planes, stride=1, downsample=None))

        self.bm = nn.BatchNorm2d(planes)

        self.sample_down = {}
        for key in self.keys:
            self.sample_down[key] = nn.Linear(self.sizes[key], config.h_size)
            if config.cuda:
                self.sample_down[key] = self.sample_down[key].cuda()

    def forward(self, h, m):
        keys = self.keys
        dim_end = self.dim_words + 1

        # try to sample down
        m_high = [self.sample_down[key](m[key]).unsqueeze(dim_end) for key in keys]
        m_high = torch.cat(tuple(m_high), dim=dim_end)
        m_high = torch.sum(m_high, dim=dim_end) # sum retrieved memory along types: since types are supposed to be softmaxed, this makes sense

        # reasoning step
        a = torch.mul(h, m_high).unsqueeze(1) # add channel dimension
        a = self.res_conv_a(a)
        a = F.softmax(a, dim=self.dim_words)
        x = torch.mul(m_high.unsqueeze(1), a)
        x = self.res_conv_x(x) # x: batch_size, channel_size, embed_size, h_size

        x = self.bm(x)
        x = F.relu(x)

        return x

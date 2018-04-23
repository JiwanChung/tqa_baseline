from basicBlock import BasicBlock

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuestionAttend(nn.Module):
    def __init__(self, config):
        super(QuestionAttend, self).__init__()

        self.dim_words = config.dim_words
        planes = 16

        self.res_conv_x = nn.Sequential(
                                BasicBlock(planes, planes, stride=1, downsample=None),
                                BasicBlock(planes, planes, stride=1, downsample=None),
                                nn.Conv2d(planes, 1, 3, stride=1, padding=1),
                                nn.Linear(config.h_size, config.q_size))

    def forward(self, qa, x):

        # downsample & reason
        x_qa = self.res_conv_x(x).squeeze()
        # attend
        a = F.softmax(x_qa, dim=self.dim_words)
        qa = torch.mul(qa, a)

        return qa

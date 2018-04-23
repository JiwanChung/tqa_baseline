from downsampleH import DownSampleH

import torch.nn as nn
import torch.nn.functional as F

class Confidence(nn.Module):
    def __init__(self, config):
        super(Confidence, self).__init__()

        self.downsample = DownSampleH(config)

    def forward(self, h):

        conf = self.downsample(h)
        conf = F.sigmoid(conf)

        return conf

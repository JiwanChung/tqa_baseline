import torch
import torch.nn as nn
from torch.autograd import Variable

from simpleModule import SimpleModule


class Controller(nn.Module):
    def __init__(self, config):
        super(Controller, self).__init__()

        self.k = config.k  # MAX num of steps
        self.conf_threshold = config.conf

        self.batch_size = config.batch_size
        self.h_size = config.h_size
        self.emb = config.embed_size * config.bi

        self.init_h = lambda size_list: Variable(torch.zeros(*size_list)).cuda() if config.cuda else Variable(torch.zeros(*size_list))

        self.module = SimpleModule(config)


class POCController(Controller):
    def __init__(self, config):
        # hyperNetwork
        super(POCController, self).__init__(config)

    def forward(self, M, qa):
        current_batch_size = qa.size()[0]

        h = self.init_h([current_batch_size, self.emb, self.h_size])

        for i in range(self.k):
            # run module
            qa, h, o, conf = self.module(M, qa, h)
            # if conf > self.conf_threshold:

        return o, conf


class HyperController(Controller):
    def __init__(self, config):
        # hyperNetwork
        super(HyperController, self).__init__(config)

    def forward(self, M, qa):
        h = self.init_h([self.batch_size, self.emb, self.h_size])

        for i in range(self.k):
            # generate weight

            # inject weight

            # run module
            qa, h, o, conf = self.module(M, qa, h)
            if conf > self.conf_threshold:
                break

        return o, conf

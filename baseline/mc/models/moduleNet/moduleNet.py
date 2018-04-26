from encoder import Encoder
from controller import Controller, POCController
from decoder import Decoder

import torch
import torch.nn as nn
from torch.autograd import Variable


class ModuleNet(nn.Module):
    def __init__(self, config, vocab):
        # hyperNetwork
        super(ModuleNet, self).__init__()

        self.h_size = config.h_size
        self.dim_words = config.dim_words
        self.batch_size = config.batch_size

        self.debug = config.debug

        self.encoder = Encoder(config, vocab)
        self.controller = Controller(config) if config.hyper else POCController(config)
        self.init_h = lambda size_list: Variable(torch.zeros(*size_list))
        self.decoder = Decoder(config)

    def forward(self, CO, Q, A):

        # encoding layer
        c, q, A = self.encoder.forward(CO, Q, A)
        if self.debug:
            print('c', torch.mean(c))
        M = {'c': c}

        # reasoning layer
        MA = torch.sum(A, dim=(self.dim_words + 1)).squeeze()
        M['A'] = MA  # A to memory
        o, conf = self.controller(M, q)
        if self.debug:
            print('o', torch.mean(o))

        # decoder layer
        p = self.decoder(o, A)
        p = conf * p

        return p

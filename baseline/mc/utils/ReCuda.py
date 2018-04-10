import torch

class ReCuda():

    def __init__(self, config):
        self.if_cuda = config.cuda
        if self.if_cuda:
            self.torch = torch.cuda
        else:
            self.torch = torch
        if self.if_cuda:
            self.data_if_cuda = 0
        else:
            self.data_if_cuda = -1

    def var(self, variable):
        if self.if_cuda:
            return variable.cuda()
        else:
            return variable

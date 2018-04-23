import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class MemoryAttention(nn.Module):
    def __init__(self, config):
        super(MemoryAttention, self).__init__()

        self.dim_words = config.dim_words
        self.keys = config.keys
        self.recuda = config.recuda

        # linear mapping
        self.linear_map = {}
        for key in self.keys:
            self.linear_map[key] = nn.Linear(config.q_size * config.h_size, config.sizes[key])
            if config.cuda:
                self.linear_map[key] = self.linear_map[key].cuda()

    def forward(self, MO, qa, h):

        M = MO
        keys = self.keys

        size = {}
        for key in keys:
            size[key] = list(M[key].size())

        S = torch.matmul(qa.unsqueeze(3), h.unsqueeze(2))
        s_size = S.size()
        S = S.view(s_size[0], s_size[1], -1)
        # attention
        a = {}
        for key in keys:
            a[key] = torch.mul(M[key], self.linear_map[key](S))
            a[key] = F.softmax(a[key], dim=self.dim_words)
        # attention score
        scores = {}
        for key in keys:
            scores[key] = torch.norm(a[key], dim=self.dim_words)
            scores[key] = scores[key].unsqueeze(2)

        tuple_a = ()
        index_a = {}
        for i, key in enumerate(keys):
            tuple_a += (scores[key], )
            index_a[key] = i

        # score to softmax index of attending memory type
        score = torch.cat(tuple_a, dim=2)
        score = F.softmax(score, dim=self.dim_words)

        for key in keys:
            coeff = torch.index_select(score, self.dim_words, Variable(self.recuda.torch.LongTensor([index_a[key]])))
            a[key] = torch.matmul(a[key].unsqueeze(3), coeff.unsqueeze(2))
            a[key] = a[key].squeeze()

        m = {key : torch.mul(M[key], a[key]) for key in keys}

        return m

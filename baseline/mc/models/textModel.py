import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TextModel(nn.Module):
    def __init__(self, config, vocab):
        super(TextModel, self).__init__()

        self.config = config

        self.embed = nn.Embedding(len(vocab), config.emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)

        self.bi = config.bi

        self.embed_context = nn.GRU(config.emb_dim, self.config.embed_size, bidirectional=config.bi_gru)
        self.embed_question = nn.GRU(config.emb_dim, self.config.embed_size, bidirectional=config.bi_gru)
        self.embed_answer = nn.GRU(config.emb_dim, self.config.embed_size, bidirectional=config.bi_gru)

    def forward(self, context, question, answers):

        context = context[0]
        question = question[0]
        answers = answers[0]

        if not self.config.single_topic:
            context_shape = list(context.data.size())
            context_shape.append(self.config.emb_dim)
            context = context.view(-1, context.size()[2])

        context = self.embed(context)
        question = self.embed(question)

        if not self.config.large_topic:
            if not self.config.single_topic:
                context = context.view(*context_shape)
                context = torch.sum(context, 1) # sum along num of topics

        M, hm = self.embed_context(context) # P x embed_size
        U, hu = self.embed_question(question) # Q X embed_size

        M = M.permute(1,0,2)
        U = U.permute(1,2,0)
        S = torch.matmul(M, U)
        S, S_index = torch.max(S, dim=2)
        a = F.softmax(S).unsqueeze(0).permute(1,2,0)
        a = a.expand(M.data.size())
        m = torch.mul(a, M)
        m = torch.sum(m, 1).unsqueeze(0)

        origin_size = answers.data.size()
        answers = answers.view(-1, answers.size()[2])
        if self.config.verbose:
            if len(answers.data.size()) < 3:
                print(answers.data)
        answers = self.embed(answers)
        C, hc = self.embed_answer(answers) # A X embed_size
        C = C.unsqueeze(0).view(origin_size[0], origin_size[1], origin_size[2], self.bi*self.config.embed_size)
        c = torch.sum(C, dim=0)
        r = torch.matmul(m.permute(1,0,2), c.permute(1,2,0)).squeeze()

        return r

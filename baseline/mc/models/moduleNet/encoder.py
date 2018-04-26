import torch
import torch.nn as nn

import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, config, vocab):
        super(Encoder, self).__init__()

        self.bi = config.bi
        self.embed_size = config.embed_size
        self.batch_size = config.batch_size
        self.recuda = config.recuda

        self.debug = config.debug

        self.embed = nn.Embedding(len(vocab), config.emb_dim)
        self.vocabs = vocab.vectors.cuda() if config.cuda else vocab.vectors()
        self.embed.weight.data.copy_(self.vocabs)
        self.embed_context = nn.GRU(config.emb_dim, config.embed_size, bidirectional=config.bi_gru)
        self.embed_question = nn.GRU(config.emb_dim, config.embed_size, bidirectional=config.bi_gru)
        self.embed_answer = nn.GRU(config.emb_dim, config.embed_size, bidirectional=config.bi_gru)
        self.normalize_row = nn.Softmax(dim=2)  # normalize along words of topic

    def forward(self, CO, Q, A):

        context_shape = list(CO.data.size())
        context_shape.append((self.embed_size * self.bi))
        CO = CO.view(-1, CO.size()[2])

        answer_shape = list(A.data.size())
        answer_shape.append((self.embed_size * self.bi))
        A = A.view(-1, A.size()[2])

        CO = self.embed(CO)
        Q = self.embed(Q)
        A = self.embed(A)

        CO, hc = self.embed_context(CO) # P x embed_size
        CO = CO.view(*context_shape)
        Q, hq = self.embed_question(Q) # Q X embed_size
        A, ha = self.embed_answer(A)
        A = A.view(*answer_shape) # A X embed_size

        CO = CO.permute(1, 2, 0, 3) # topic_num, batch_size, words_topic, embed_size
        Q = Q.permute(1, 2, 0) # batch_size, embed_size, words_question
        A = A.permute(2, 3, 0, 1) # batch_size, embed_size, words_answer, answer_num

        C = CO # store data

        # <attention>
        S = torch.matmul(C, Q) # topic_num, batch_size, words_topic, words_question
        S = self.normalize_row(S) # attention practice based on QAnet (Google)
        Att = torch.matmul(S, Q.permute(0, 2, 1)) # Q: batch_size, words_question, embed_size
        Att = F.softmax(Att, dim=0)
        # Att: topic_num, batch_size, words_topic, embed_size
        C = F.normalize(C, dim=2) # normalize along words_topic, removing bias in total num of words
        C = torch.mul(Att, C) # apply attention
        C = torch.sum(torch.sum(C, 3), 2) # reduce dimension
        # </attention>

        maxval, argmax = torch.max(C, 0) # pick top 1 topic
        c = CO[argmax, torch.arange(0, argmax.size()[0]).type_as(argmax.data), :] # reduce based on top 1 indices
        c = c.permute(0, 2, 1)  # batch_size, embed_size, words_topic
        if self.debug:
            print('Q:', torch.mean(Q))
            print('C:', torch.mean(C))
            print('argmax:', argmax)

        return c, Q, A

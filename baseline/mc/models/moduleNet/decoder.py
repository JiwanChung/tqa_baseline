import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.dim_words = config.dim_words
        self.ans_k = config.ans_k

        self.debug = config.debug

        self.sampledown = nn.Sequential(
                            nn.Linear(config.a_size * config.h_size, config.hidden_size),
                            nn.ReLU(),
                            nn.Linear(config.hidden_size, config.a_size),
                            nn.ReLU()
                        )

        def self_attention(k, X):
            # self attention
            Y = torch.mul(X.unsqueeze(self.dim_words + 1), X.unsqueeze(self.dim_words + 2))
            Y = torch.sum(Y, self.dim_words + 2).squeeze()
            if k < 2:
                return torch.mul(X, F.softmax(Y))
            return torch.mul(self_attention(k - 1, X), F.softmax(Y))

        self.sa = self_attention

    def forward(self, o, A):
        # naive probability

        o = o.squeeze()
        x = torch.matmul(o.unsqueeze(3).unsqueeze(2), A.unsqueeze(3))
        s = x.size()
        x = x.view(s[0],s[1],-1,s[4])
        x = self.sampledown(x.permute(0,1,3,2))
        a = F.softmax(x.permute(0,1,3,2), dim=self.dim_words)
        oa = torch.mul(A, a)
        oa = oa.view(oa.size()[0], -1, oa.size()[3])
        oa = torch.sum(oa, 1)
        p = F.softmax(oa, dim=1)

        p_sum = torch.sum(p, dim=1, keepdim=True)

        # answer patchwise probability
        A = torch.mul(A, p.unsqueeze(1).unsqueeze(1))  # batch_size, embed_size, words_answer, answer_num
        A = A / p_sum.unsqueeze(1).unsqueeze(1)
        A = self.sa(self.ans_k, A)

        A = torch.sum(A, 1).squeeze()
        A = torch.sum(A, 1).squeeze()

        return A

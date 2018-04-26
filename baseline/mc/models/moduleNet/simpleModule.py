from memoryAttention import MemoryAttention
from reasoning import Reasoning
from questionAttend import QuestionAttend
from forgetGate import ForgetGate
from confidence import Confidence
from output import Output

import torch.nn as nn

class SimpleModule(nn.Module):
    def __init__(self, config):
        super(SimpleModule, self).__init__()

        self.memory_attention = MemoryAttention(config)
        self.reasoning = Reasoning(config)
        self.question_attend = QuestionAttend(config)
        self.forget_gate = ForgetGate(config)
        self.confidence = Confidence(config)
        self.output = Output(config)

    def forward(self, M, qa, h):
        '''
        M = [c, A],
        c: batch_size, embed_size, words_topic
        A: batch_size, embed_size, words_answer, answer_num
        qa: batch_size, embed_size, words_question
        h: batch_size, h_size
        '''
        m = self.memory_attention(M, qa, h)

        x = self.reasoning(h, m)

        qa = self.question_attend(qa, x)

        h = self.forget_gate(h, x)
        conf = self.confidence(h)
        o = self.output(h)

        return qa, h, o, conf

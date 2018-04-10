##
# coding: utf-8

import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

import os
import argparse
import sys
import glob
from tensorboard import Logger
from utils.ReCuda import ReCuda

from readData import get_data
from model import TextModel

if not os.path.isdir('logs'):
    os.mkdir('logs')
logger = Logger('./logs')

parser = argparse.ArgumentParser(description='TQA basline Text model')
parser.add_argument('--ckpt', '-c', default=None, help='ckpt epoch name')
parser.add_argument('--test', '-t', default=False, help='debug')
args = parser.parse_args()
args.source_dir = '/home/jiwan/tqa/prepro/data'
args.ckpt_dir = './ckpt'
args.emb_dim = 300
args.repeat = False
args.learning_rate = 0.001
args.if_pair = False
args.log_epoch = 4
args.bi_gru = True
args.batch_size = 36
args.verbose = False
args.end_epoch = 100

args.cuda = False
if not torch.cuda.is_available():
    args.cuda = False

config = args
config.recuda = ReCuda(config)
config.resume = False
if config.ckpt is not None:
    config.resume = True

config.recuda.torch.manual_seed(1)
##
data, iters, vocab = get_data(config)

if config.verbose:
    x = next(iter(iters['val']))
    print(x.answers[0].data)


##
# get net
def get_net(config, vocab):
    if config.resume:
        print('RESUME {}th epoch'.format(config.ckpt))
        assert os.path.isdir('ckpt'), 'Error: no dir'
        ckpt = torch.load(os.path.join(config.ckpt_dir, 'ckpt_{}.t7'.format(config.ckpt)))
        net = TextModel(vocab, config, 100)
        net.load_state_dict(ckpt['params'])
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']
    else:
        net = TextModel(vocab, config, 100)
        best_acc = 0
        start_epoch = 0
    net = config.recuda.var(net)
    return net, best_acc, start_epoch

net, best_acc, start_epoch = get_net(config, vocab)

# use CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

print("Let\'s start Training")

def train_epoch(args, data, train_iter, epoch):

    # train
    train_loss = 0
    for batch_index, data in tqdm(enumerate(train_iter)):
        net.zero_grad()
        answers_size = len(data.answers)
        answers = torch.stack(data.answers, dim=2)
        # batch_size = data.correct_answer.data.size()[0]
        target = Variable(data.correct_answer.data, requires_grad=False)
        target = config.recuda.var(target)
        # run
        if config.verbose:
            print('context:', data.topic.data)
        y = net.forward(data.question.data, data.topic.data, answers, answers_size)
        if config.verbose:
            print('y:', y.data)
            print('t:', target.data)
        loss = loss_fn(y, target)
        # count loss
        loss.backward()
        # optimize
        optimizer.step()

        train_loss += loss.data[0]
        loss_per = train_loss/(batch_index+1)
        print("Training {} epoch, loss: {}".format(epoch, loss_per))
        logger.scalar_summary('tr_loss', loss_per, epoch+1)


def validate_epoch(args, data, val_iter, epoch):
    # validate from time to time

    print("begin validation")
    correct = 0
    total = 0
    for index_v, data in tqdm(enumerate(val_iter)):
        answers_size = len(data.answers)
        answers = torch.stack(data.answers, dim=2)
        # run
        y = net.forward(data.question.data, data.topic.data, answers, answers_size)
        value, pred = torch.max(y, 1)
        check = torch.eq(data.correct_answer.data, pred.data)
        if config.verbose:
            print(torch.sum(check), check.size())
        correct += torch.sum(check)
        total += (check.size()[0])

    acc = 100.*correct/total
    print("Val {} epoch, acc: {}".format(epoch, acc))

    logger.scalar_summary('val_acc', acc, (epoch + 1))

    return acc


def save_net(net, epoch, acc):
    print('saving')
    state = {
        'params': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('ckpt'):
        os.mkdir('ckpt')
    torch.save(state, os.path.join(config.ckpt_dir, 'ckpt_{}.t7'.format(epoch)))


##
for epoch in range(start_epoch, config.end_epoch):
    print("{} epoch".format(epoch))
    train_epoch(args, data, iters['train'], epoch)
    acc = validate_epoch(args, data, iters['val'], epoch)

    save_net(net, epoch, acc)
##

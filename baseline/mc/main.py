##
# coding: utf-8

import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn

import os
from functools import partial
from collections import Counter
import pickle

from readData import get_data


def main(args):
    config = args

    print('loading data')
    data, iters, vocab, stats, Q_field = get_data(config)

    config.q_size = stats['question_size']
    config.a_size = stats['answer_size']
    config.c_size = stats['topic_size']
    config.keys = ['A', 'c']
    config.sizes = {'A': config.a_size, 'c': config.c_size}

    if config.verbose:
        x = next(iter(iters['val']))
        print(x.answers[0].data)

    print('loading model')
    net, best_acc, config.start_epoch = get_net(config, vocab)

    if config.train:
        print("Let\'s start Training")
        train_all(net, data, iters, config)
    else:
        print("Let\'s start Testing")
        test_and_save(net, data, iters[config.test_iter], config)

##
# get net
def get_net(config, vocab):
    if config.resume:
        assert os.path.isdir('ckpt'), 'Error: no dir'
        ckpt = torch.load(os.path.join(config.ckpt_dir, config.load_ckpt))
        net = config.model(config, vocab)
        net.load_state_dict(ckpt['params'])
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']
        print('RESUME {}th epoch'.format(start_epoch))
    else:
        net = config.model(config, vocab)
        best_acc = 0
        start_epoch = 0
    net = config.recuda.var(net)
    print('PARAMS: ', sum(p.numel() for p in net.parameters() if p.requires_grad))
    return net, best_acc, start_epoch

def prepare_data(config, field, permute_list):
    # permute for vocab Embedding
    return config.recuda.var(field.permute(*permute_list).contiguous())

##
def run_net(net, config, d):

    # run
    pd = partial(prepare_data, config)

    C = pd(d.t_word, [1, 0, 2])
    Q = pd(d.q_word, [0, 1])
    AS = pd(d.as_word, [1, 0, 2])

    CC = pd(d.t_char, [1, 0, 2, 3])  # topic_words, topic_num, batch_size, char_num
    QC = pd(d.q_char, [0, 1, 2])
    ASC = pd(d.as_char, [1, 0, 2, 3])

    return net.forward((C, CC), (Q, QC), (AS, ASC))


##
def train_epoch(net, config, data, train_iter, epoch):

    # train
    train_loss = 0
    for batch_index, data in tqdm(enumerate(train_iter)):
        net.zero_grad()
        target = Variable(data.ca.data, requires_grad=False)
        target = config.recuda.var(target)

        if config.verbose:
            if config.single_topic:
                print('context:', data.topic.data)
            else:
                print('context_list:', data.topic[0])

        # run
        y = run_net(net, config, data)

        if config.verbose:
            print('y:', y.data)
            print('t:', target.data)
        if config.debug:
            print('y:', y.data)

        loss = config.loss_fn(y, target)
        # count loss
        loss.backward()
        # optimize
        config.optimizer.step()

        train_loss += loss.data[0]
        loss_per = train_loss / (batch_index + 1)
        print("Training {} epoch, loss: {}".format(epoch, loss_per))
        config.logger.scalar_summary('tr_loss{}'.format(config.ckpt_name), loss_per, epoch + 1)


##
def validate_epoch(net, config, data, val_iter, epoch):
    # validate from time to time

    print("begin validation")
    correct = 0
    total = 0
    for index_v, data in tqdm(enumerate(val_iter)):
        # run
        y = run_net(net, config, data)

        value, pred = torch.max(y, 1)
        check = torch.eq(data.ca.data, pred.data)
        if config.verbose:
            print(torch.sum(check), check.size())
        correct += torch.sum(check)
        total += (check.size()[0])

    acc = 100.*correct/total
    print("Val {} epoch, acc: {}".format(epoch, acc))

    config.logger.scalar_summary('val_acc{}'.format(config.ckpt_name), acc, (epoch + 1))

    return acc


##
def save_net(net, config, epoch, acc):
    print('saving')
    state = {
        'params': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('ckpt'):
        os.mkdir('ckpt')
    if not os.path.isdir('ckpt/temp'):
        os.mkdir('ckpt/temp')
    torch.save(state, os.path.join(config.ckpt_dir, 'temp', 'ckpt{}_{}.t7'.format(config.ckpt_name,epoch)))


##
def train_all(net, data, iters, config):
    config.loss_fn = nn.CrossEntropyLoss()
    config.optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.l2)

    for epoch in range(config.start_epoch, config.end_epoch):
        print("{} epoch".format(epoch))
        train_epoch(net, config, data, iters['train'], epoch)
        acc = validate_epoch(net, config, data, iters['val'], epoch)

        save_net(net, config, epoch, acc)
##


def test(net, config, data, test_iter):
    test_net = Counter()
    net_dict = {}

    print("begin testing")
    for index_t, data in tqdm(enumerate(test_iter)):
        # run
        y = run_net(net, config, data)

        value, pred = torch.max(y, 1)
        check = torch.eq(data.ca.data, pred.data)
        for i in range(len(check)):
            test_net[data.id[i]] += int(check[i])
            net_dict[data.id[i]] = [pred.data[i], data.ca.data[i]]

    return test_net, net_dict


def test_and_save(net, data, test_iter, config):
    test_counter, test_dict = test(net, config, data, test_iter)

    with open(os.path.join(config.source_dir, 'correct_counter_{}{}.pickle'.format(config.test_iter, config.ckpt_name)), 'wb') as outfile:
        pickle.dump(test_counter, outfile)

    with open(os.path.join(config.source_dir, 'correct_dict_{}{}.pickle'.format(config.test_iter, config.ckpt_name)), 'wb') as outfile:
        pickle.dump(test_dict, outfile)

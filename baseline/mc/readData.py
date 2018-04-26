# -*- coding: utf-8 -*-

import json
import os
import sys
import torchtext

from utils.listField import ListField
from utils.idField import IdField
from utils.labelField import LabelField

from utils.tokenizer import Tokenizer

sys.path.append(os.path.dirname(os.path.expanduser('tqa/')))


def get_stat(config):

    files = ['train', 'val', 'test']
    stats = {}
    for name in files:
        stat = json.load(open(os.path.join(config.source_dir,'stat_{}_full.json'.format(name)), 'r'))
        for key in stat.keys():
            if str(key) in stats:
                stats[str(key)] = max(stats[str(key)], stat[key])
            else:
                stats[str(key)] = stat[key]

    return stats


def get_data(config):
    tokenizer = Tokenizer('revtok') if config.reversible else Tokenizer('spacy')

    stats = get_stat(config)

    Q_opts = {'sequential': True, 'tensor_type': config.recuda.torch.LongTensor, 'tokenize': tokenizer.tokenize, 'use_vocab': True}
    AS_opts = {'use_vocab': True}
    T_opts = {'use_vocab': True}
    if config.fix_length:
        Q_opts['fix_length'] = stats['question_size']
        AS_opts['fix_length'] = stats['answer_size']
        AS_opts['fix_list_length'] = stats['answer_num']
        T_opts['fix_length'] = stats['topic_size']
        T_opts['fix_list_length'] = stats['topic_num']

    if config.reversible:
        Q = torchtext.data.ReversibleField(**Q_opts)
    else:
        Q = torchtext.data.Field(**Q_opts)

    if not config.single_topic:
        T = ListField(Q, **T_opts)
        AS = ListField(Q, **AS_opts)
    else:
        T = Q
        AS = ListField(T, **AS_opts)

    CA = LabelField(use_vocab=False)
    ID = IdField(use_vocab=False)

    data_path = config.source_dir

    field_list = {'answers': [('as_word', AS)], 'correct_answer': [('ca', CA)], 'id': [('id', ID)], 'question': [('q_word', Q)], 'topic': [('t_word', T)]}

    if config.character_embedding:
        char_opts = {'fix_length': config.char_length, 'use_vocab': True}
        Nested_opts = {'tokenize': tokenizer.tokenize, 'use_vocab': True}

        Q_sub = torchtext.data.Field(**char_opts)
        Q_char = torchtext.data.NestedField(Q_sub, **Nested_opts)
        field_list['question'] = [('q_char', Q_char), field_list['question'][0]]

        if not config.single_topic:
            T_char = ListField(Q_char, **T_opts)
        else:
            T_char = Q_char

        AS_char = ListField(Q_char, **AS_opts)
        field_list['topic'] = [('t_char', T_char), field_list['topic'][0]]
        field_list['answers'] = [('as_char', AS_char), field_list['answers'][0]]

    if_test = ''
    full = ''
    sample = ''
    if config.test:
        if_test = '2'
    if not config.single_topic:
        full = '_full'
    tr_data = 'data_train{}{}{}.tsv'.format(if_test, full, sample)
    ts_data = 'data_test{}{}.tsv'.format(full, sample)
    val_data = 'data_val{}{}.tsv'.format(full, sample)


    print("loading {}, {}, {}".format(tr_data, val_data, ts_data))
    train, val, test = torchtext.data.TabularDataset.splits(path=data_path, train=tr_data, validation=val_data, test=ts_data, format='tsv', fields=field_list)

    def get_iter(data, name, size):
        if name == 'train':
            if_train = True
        else:
            if_train = False
        return torchtext.data.Iterator(
            data, sort=False, repeat=config.repeat, train=if_train,
            batch_size=size, device=config.recuda.data_if_cuda, shuffle=True)  # GPU

    Q.build_vocab(train, vectors="glove.6B.{}d".format(config.emb_dim))
    T.build_vocab(train, vectors="glove.6B.{}d".format(config.emb_dim))
    AS.build_vocab(train, vectors="glove.6B.{}d".format(config.emb_dim))
    ID.build_vocab(train, vectors="glove.6B.{}d".format(config.emb_dim))
    if config.character_embedding:
        Q_sub.build_vocab(train)
        Q_char.build_vocab(train)
        T_char.build_vocab(train)
        AS_char.build_vocab(train)


    if config.verbose:
        print('A:', len(AS.vocab.freqs), len(AS.vocab.itos), len(AS.vocab.stoi), len(AS.vocab.vectors))
        print('T:', len(T.vocab.freqs), len(T.vocab.itos), len(T.vocab.stoi), len(T.vocab.vectors))
    vocab = Q.vocab

    data = {'train':train, 'val':val, 'test':test}
    iters = {'train':get_iter(train, 'train', config.batch_size), 'val':get_iter(val, 'val', config.batch_size), 'test':get_iter(test, 'test', config.batch_size)}

    return data, iters, vocab, stats, Q

# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import torchtext
import torch

sys.path.append(os.path.dirname(os.path.expanduser('tqa/')))

from utils.listField import ListField
from utils.idField import IdField
from utils.labelField import LabelField
from utils.ReCuda import ReCuda

import re
import numpy as np

import spacy
spacy_en = spacy.load('en')

def tokenizer(text): # create a tokenizer function
    text = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ",
        str(text))
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\!+", "!", text)
    text = re.sub(r"\,+", ",", text)
    text = re.sub(r"\?+", "?", text)
    text = re.sub(r".$", "", text)

    if sys.version_info < (3, 0):
        text = unicode(text)

    print_list = [tok.text for tok in spacy_en.tokenizer(text) if tok.text != " "]

    return print_list

def get_data(config):
    X = torchtext.data.Field(sequential=True, tensor_type=config.recuda.torch.LongTensor, tokenize=tokenizer, use_vocab=True)
    if not config.single_topic:
        T = ListField(X, use_vocab=True)
        AS = ListField(X, use_vocab=True)
    else:
        T = X
        AS = ListField(T, use_vocab=True)

    CA = LabelField(use_vocab=False)
    ID = IdField(use_vocab=False)

    data_path = config.source_dir

    field_list = [('answers', AS), ('correct_answer', CA), ('id', ID), ('question', X), ('topic', T)]

    if_test = ''
    full = ''
    if config.test:
        if_test = '2'
    if not config.single_topic:
        full = '_full'
    tr_data = 'data_train{}{}.tsv'.format(if_test, full)
    ts_data = 'data_test{}.tsv'.format(full)
    val_data = 'data_val{}.tsv'.format(full)

    print("loading {}, {}, {}".format(tr_data, ts_data, val_data))
    train, val, test = torchtext.data.TabularDataset.splits( path=data_path, train=tr_data, validation=val_data, test=ts_data, format='tsv', skip_header=True, fields=field_list)

    def get_iter(data, name, size):
        if name == 'train':
            if_train = True
        else:
            if_train = False
        return torchtext.data.Iterator(
                data, sort=False, repeat=config.repeat, train= if_train,
                batch_size=size, shuffle=config.shuffle, device=config.recuda.data_if_cuda)#GPU

    X.build_vocab(train, vectors="glove.6B.{}d".format(config.emb_dim))
    T.build_vocab(train, vectors="glove.6B.{}d".format(config.emb_dim))
    AS.build_vocab(train, vectors="glove.6B.{}d".format(config.emb_dim))
    ID.build_vocab(train, vectors="glove.6B.{}d".format(config.emb_dim))

    if config.verbose:
        print('A:', len(AS.vocab.freqs), len(AS.vocab.itos), len(AS.vocab.stoi), len(AS.vocab.vectors))
        print('T:', len(T.vocab.freqs), len(T.vocab.itos), len(T.vocab.stoi), len(T.vocab.vectors))
    vocab = X.vocab

    data = {'train':train, 'val':val, 'test':test}
    iters = {'train':get_iter(train, 'train', config.batch_size), 'val':get_iter(val, 'val', config.batch_size), 'test':get_iter(test, 'test', config.batch_size)}

    return data, iters, vocab

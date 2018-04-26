import torch

from main import main as run

from models import TextModel, ModuleNet

import argparse
import os
from tensorboard import Logger
from utils.ReCuda import ReCuda


def cli():
    print("Getting configs...")
    args = setup()
    run(args)


def setup():
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    logger = Logger('./logs')

    parser = argparse.ArgumentParser(description='TQA model')
    parser.add_argument('--load-ckpt', '-c', default=None, help='ckpt name')
    parser.add_argument('--test', '-t', default=False, help='debug')
    parser.add_argument('--train', '-tr', default=True, help='train or test')
    args = parser.parse_args()
    args.source_dir = '/home/jiwan/tqa/prepro/data'
    args.ckpt_dir = './ckpt'
    args.emb_dim = 300
    args.repeat = False
    args.learning_rate = 0.001
    args.if_pair = False
    args.log_epoch = 4
    args.bi_gru = True
    args.batch_size = 18
    args.verbose = False
    args.end_epoch = 100
    args.single_topic = False
    args.embed_size = 100
    args.shuffle = True
    args.large_topic = False
    args.reversible = True
    args.fix_length = True
    args.reasoning_planes = 16
    args.k = 4
    args.conf = 0.7
    args.h_size = 128
    args.hyper = False
    args.hidden_size = 300
    args.dim_words = 2
    args.ans_k = 7
    args.l2 = 0.0005
    args.debug = True
    args.character_embedding = True
    args.char_length = 24

    args.bi = 2 if args.bi_gru else 1

    args.resume = False
    if args.load_ckpt is not None:
        args.resume = True

    args.test_iter = 'val'

    args.cuda = True
    if not torch.cuda.is_available():
        args.cuda = False

    config = args
    config.recuda = ReCuda(config)
    config.ckpt_name = '_single'
    if not config.single_topic:
        config.ckpt_name = '_all'
    if config.large_topic:
        config.ckpt_name = '_full'

    config.logger = logger

    config.recuda.torch.manual_seed(1)

    config.model = TextModel

    return config


if __name__ == "__main__":
    cli()

import os.path
import random

import numpy as np
import torch


def optimizer_opts(parser):
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--optimizer_name', type=str, default="AdamW")


def preprocess_data_opts(parser):
    parser.add_argument('--root', default='E:/Documents/影像学NLP/train', type=str)
    train_path = 'train.csv'
    test_path = 'preliminary_a_test.csv'

    parser.add_argument('--train_path', default=train_path, type=str)
    parser.add_argument('--test_path', default=test_path, type=str)


def train_opts(parser):
    parser.add_argument('--n_accumulate', default=1, type=int)
    parser.add_argument('--NUM_WORKERS', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=620)


def model_opts(parser):
    parser.add_argument('--model_name', default='facebook/bart-base', type=str)
    parser.add_argument('--encode_max_length', default=192, type=int)
    parser.add_argument('--decode_max_length', default=192, type=int)
    parser.add_argument('--result_dir', default='./models/', type=str)


def batch_size_opts(parser):
    parser.add_argument('--batch_size', type=int, default=4)


def scheduler_opts(parser):
    parser.add_argument('--lr_decay_func', type=str, default='cosine')
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--num_warmup_steps', type=int, default=20)


def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

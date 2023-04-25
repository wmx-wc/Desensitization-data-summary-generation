import torch

from data.mask_dataset import BartDataset, MyDataset
from data.metric_dataset import MetricDataset
from data.regular_dataset import RegularDataset
from data.valid_dataset import ValidDataset


def build_valid_dataloader(args, meta_data, shuffle=False):
    data_dataset = ValidDataset(meta_data, encode_max_length=args.encode_max_length,
                                decode_max_length=args.decode_max_length)
    data_loader = torch.utils.data.DataLoader(data_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              num_workers=args.NUM_WORKERS)
    return data_loader


def build_metric_dataloader(args, meta_data):
    data_dataset = MetricDataset(meta_data, encode_max_length=args.encode_max_length,
                                 decode_max_length=args.decode_max_length)
    data_loader = torch.utils.data.DataLoader(data_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=False,
                                              num_workers=args.NUM_WORKERS)
    return data_loader


def build_pre_bart_dataloader(args, meta_data, shuffle=True):
    data_dataset = BartDataset(meta_data, max_length=args.encode_max_length, vocab_size=args.vocab_size)
    if shuffle is False:
        args.batch_size = 16
    data_loader = torch.utils.data.DataLoader(data_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              num_workers=args.NUM_WORKERS)
    return data_loader


def build_my_dataloader(args, meta_data, shuffle=True):
    data_dataset = MyDataset(meta_data, encode_max_length=args.encode_max_length,
                             decode_max_length=args.decode_max_length, vocab_size=args.vocab_size)
    data_loader = torch.utils.data.DataLoader(data_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=False,
                                              num_workers=args.NUM_WORKERS)
    return data_loader


def build_regular_dataloader(args, meta_data, shuffle=True):
    data_dataset = RegularDataset(meta_data, encode_max_length=args.encode_max_length,
                                  decode_max_length=args.decode_max_length, vocab_size=args.vocab_size)
    data_loader = torch.utils.data.DataLoader(data_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=False,
                                              num_workers=args.NUM_WORKERS)
    return data_loader

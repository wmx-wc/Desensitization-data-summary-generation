import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from random import sample
from data.process_data import precess_data


def mask_span_tokens(inputs: torch.Tensor, special_tokens_mask: torch.Tensor, vocab_size=1400, lm_prob=0.8,
                     valid_len=None):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()
    prob_matrix = torch.full(labels.shape, lm_prob)  # 填充0.15
    prob_matrix.masked_fill_(special_tokens_mask.bool(), value=0.0)  # 特殊tokens的索引不参与预测
    # 掩蔽的索引,从伯努利分布中提取二进制随机数（0或1）
    mask_indices = torch.bernoulli(prob_matrix).bool()
    i = 0

    while i < valid_len:
        if mask_indices[i]:
            # 决定是否掩蔽下一个
            p = random.uniform(0, 1)
            if p <= 0.1:
                # mask 3 tokens
                step = 3
            elif p <= 0.4:
                # mask 2 tokens
                step = 2
            else:
                # mask 1 token
                step = 1
            j = 0
            while i < valid_len and j < step:
                mask_indices[i] = True
                i += 1
                j += 1
            if i < mask_indices.shape[0]:
                mask_indices[i] = False
            i += 1
        else:
            i += 1
    labels[~mask_indices] = -100  # 未参与预测的tokens的labels设置为-100 间接不参与损失函数的计算
    # 选择掩蔽的索引的百分之80的索引
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & mask_indices
    # 将上述索引替换为 mask token ids
    inputs[indices_replaced] = 4
    # 百分之10的时间替换为随机 token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & mask_indices & ~indices_replaced
    # 随机挑选单词的索引形状和 labels一样
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


class BartDataset(Dataset):
    def __init__(self, meta_data: list, max_length=192, vocab_size=1400):
        self.meta_data = meta_data
        # 序列的最大长度
        self.enc_max_length = max_length
        self.dec_max_length = max_length
        self.vocab_size = vocab_size
        # 影像医学描述
        self.des = []
        # 特殊 tokens mask
        self.des_st_masks = []
        # attention_masks,输入的有效长度
        self.des_attention_masks = []
        self.valid_lens = []
        for i in range(len(meta_data)):
            description = meta_data[i][1]
            if len(meta_data[i]) > 2:
                report = meta_data[i][2]
                len1 = len(description)
                len2 = len(report)
                description = description + [5] + report
                if len(description) > max_length:
                    description = description[: max_length]
                if len1 + len2 < self.enc_max_length:
                    self.des_st_masks.append([0] * len1 + [1] + [0] * len2 + [1] *
                                             (self.enc_max_length - len1 - len2 - 1))
                else:
                    self.des_st_masks.append([0] * len1 + [1] + [0] *
                                             (self.enc_max_length - len1 - 1))
            else:
                self.des_st_masks.append([0] * len(description) + [1] *
                                         (self.enc_max_length - len(description)))
            self.des.append(description + [0] * (self.enc_max_length - len(description)))

            self.des_attention_masks.append([1] * len(description) + [0] *
                                            (self.enc_max_length - len(description)))
            self.valid_lens.append(len(description))

    def __len__(self):
        return len(self.des)

    def __getitem__(self, index):
        enc_inputs = torch.tensor(self.des[index], dtype=torch.long)
        enc_st_mask = torch.tensor(self.des_st_masks[index], dtype=torch.long)
        enc_attention_mask = torch.tensor(self.des_attention_masks[index], dtype=torch.long)
        valid_len = self.valid_lens[index]
        enc_inputs, labels = mask_span_tokens(enc_inputs, enc_st_mask, self.vocab_size, lm_prob=0.15,
                                              valid_len=valid_len)
        dec_inputs = torch.cat([torch.tensor([2]), enc_inputs[: -1]], dim=0)
        return enc_inputs, dec_inputs, labels, enc_attention_mask


class MyDataset(Dataset):
    def __init__(self, meta_data: list, encode_max_length=160, decode_max_length=96, vocab_size=1400):
        self.meta_data = meta_data
        # 序列的最大长度
        self.enc_max_length = encode_max_length
        self.dec_max_length = decode_max_length
        self.vocab_size = vocab_size
        # 影像医学描述
        self.des = []

        # attention_masks,输入的有效长度
        self.des_attention_masks = []
        # 诊断报告
        self.repo_inputs = []
        self.repo_labels = []
        # 解码器 attention_masks,输入的有效长度
        self.repo_attention_masks = []

        for i in range(len(meta_data)):
            description = meta_data[i][1]
            report = meta_data[i][2]
            self.des.append(description + [0] * (self.enc_max_length - len(description)))
            self.des_attention_masks.append([1] * len(description) + [0] *
                                            (self.enc_max_length - len(description)))

            self.repo_inputs.append([2] + report + [1] + [0] * (self.dec_max_length - len(report) - 2))
            self.repo_labels.append(report + [1] + [0] * (self.dec_max_length - len(report) - 1))
            self.repo_attention_masks.append([1] * (len(report) + 1) + [0] * (self.dec_max_length - len(report) - 1))

    def __len__(self):
        return len(self.des)

    def __getitem__(self, index):
        enc_inputs = torch.tensor(self.des[index], dtype=torch.long)
        enc_attention_mask = torch.tensor(self.des_attention_masks[index], dtype=torch.long)
        dec_inputs = torch.tensor(self.repo_inputs[index], dtype=torch.long)
        dec_masks = torch.tensor(self.repo_attention_masks[index], dtype=torch.long)
        dec_labels = torch.tensor(self.repo_labels[index], dtype=torch.long)
        return enc_inputs, enc_attention_mask, dec_inputs, dec_labels, dec_masks


if __name__ == '__main__':
    data = precess_data('E:/Documents/影像学NLP/train', 'train.csv', mode='train')

    print()
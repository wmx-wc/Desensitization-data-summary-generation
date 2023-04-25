import torch
from torch.utils.data import Dataset

from data.mask_dataset import mask_span_tokens


class RegularDataset(Dataset):
    def __init__(self, meta_data: list, encode_max_length=160, decode_max_length=96, vocab_size=1400):
        self.meta_data = meta_data
        # 序列的最大长度
        self.enc_max_length = encode_max_length
        self.dec_max_length = decode_max_length
        self.vocab_size = vocab_size
        # 影像医学描述
        self.des = []
        # 特殊 tokens mask
        self.des_st_masks = []
        # attention_masks,输入的有效长度
        self.des_attention_masks = []
        # 诊断报告
        self.repo_inputs = []
        self.repo_labels = []
        # 特殊 tokens mask
        self.repo_st_masks = []
        # 解码器 attention_masks,输入的有效长度
        self.repo_attention_masks = []
        self.valid_lens = []
        # 记录id
        self.ids = []
        for i in range(len(meta_data)):
            self.ids.append(meta_data[i][0])
            description = meta_data[i][1]
            report = meta_data[i][2]
            self.des.append(description + [0] * (self.enc_max_length - len(description)))
            self.des_st_masks.append([0] * len(description) + [1] *
                                     (self.enc_max_length - len(description)))
            self.des_attention_masks.append([1] * len(description) + [0] *
                                            (self.enc_max_length - len(description)))

            self.repo_inputs.append([2] + report + [1] + [0] * (self.dec_max_length - len(report) - 2))

            self.repo_st_masks.append([1] + [0] * len(report))

            self.repo_labels.append(report + [1] + [0] * (self.dec_max_length - len(report) - 1))
            self.repo_attention_masks.append([1] * (len(report) + 1) + [0] * (self.dec_max_length - len(report) - 1))
            self.valid_lens.append(len(description))

    def __len__(self):
        return len(self.des)

    def __getitem__(self, index):
        # 获取编码器的输入，特殊tokens的掩码，attention mask
        enc_inputs = torch.tensor(self.des[index], dtype=torch.long)
        enc_st_mask = torch.tensor(self.des_st_masks[index], dtype=torch.long)
        enc_attention_mask = torch.tensor(self.des_attention_masks[index], dtype=torch.long)

        valid_len = self.valid_lens[index]

        enc_inputs, enc_labels = mask_span_tokens(enc_inputs, enc_st_mask, self.vocab_size, lm_prob=0.15,
                                                  valid_len=valid_len)

        dec_inputs = torch.tensor(self.repo_inputs[index], dtype=torch.long)

        dec_st_mask = self.repo_st_masks[index]
        dec_valid_len = len(dec_st_mask)
        dec_st_mask = torch.tensor(dec_st_mask + [1] * (self.dec_max_length - dec_valid_len), dtype=torch.long)

        dec_inputs, _ = mask_span_tokens(dec_inputs, dec_st_mask, self.vocab_size, lm_prob=0.15, valid_len=dec_valid_len)

        dec_masks = torch.tensor(self.repo_attention_masks[index], dtype=torch.long)
        dec_labels = torch.tensor(self.repo_labels[index], dtype=torch.long)

        return enc_inputs, enc_labels, enc_attention_mask, dec_inputs, dec_labels, dec_masks
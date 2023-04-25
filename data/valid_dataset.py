import torch
from torch.utils.data import Dataset


class ValidDataset(Dataset):
    def __init__(self, meta_data: list, encode_max_length=160, decode_max_length=96):
        self.meta_data = meta_data
        # 序列的最大长度
        self.enc_max_length = encode_max_length
        self.dec_max_length = decode_max_length
        # 影像医学描述
        self.des = []
        # attention_masks,输入的有效长度
        self.des_attention_masks = []
        # 诊断报告
        self.repo_inputs = []
        # 解码器 attention_masks,输入的有效长度
        self.repo_attention_masks = []
        # 解码器输出的标签
        self.repo_labels = []
        for i in range(len(meta_data)):
            description = meta_data[i][1]
            report = meta_data[i][2]
            self.des.append(description + [0] * (self.enc_max_length - len(description)))
            self.des_attention_masks.append([1] * len(description) + [0] * (self.enc_max_length - len(description)))
            self.repo_inputs.append([2] + report + [1] + [0] * (self.dec_max_length - len(report) - 2))
            self.repo_labels.append(report + [1] + [0] * (self.dec_max_length - len(report) - 1))
            self.repo_attention_masks.append([1] * (len(report) + 1) + [0] * (self.dec_max_length - len(report) - 1))

    def __len__(self):
        return len(self.des)

    def __getitem__(self, index):
        enc_inputs = torch.tensor(self.des[index], dtype=torch.long)
        enc_attention_mask = torch.tensor(self.des_attention_masks[index], dtype=torch.long)
        dec_inputs = torch.tensor(self.repo_inputs[index], dtype=torch.long)
        dec_labels = torch.tensor(self.repo_labels[index], dtype=torch.long)
        dec_masks = torch.tensor(self.repo_attention_masks[index], dtype=torch.long)
        return enc_inputs, enc_attention_mask, dec_inputs, dec_labels, dec_masks

